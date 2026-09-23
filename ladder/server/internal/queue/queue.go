package queue

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"ladder/server/internal/rating"
	"ladder/server/internal/runner"
	"ladder/server/internal/store"
)

const pollInterval = 250 * time.Millisecond

// Queue executes pending batches with a bounded pool of polling workers.
type Queue struct {
	store   *store.Store
	runner  *runner.Runner
	rater   rating.Rater
	workers int

	mu     sync.Mutex
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func New(s *store.Store, r *runner.Runner, workers int, rater rating.Rater) *Queue {
	if workers < 1 {
		workers = 2
	}
	if rater == nil {
		rater = rating.NoopRater{}
	}
	return &Queue{store: s, runner: r, workers: workers, rater: rater}
}

// Start launches the worker pool. Calling Start more than once is ignored.
func (q *Queue) Start(ctx context.Context) {
	q.mu.Lock()
	if q.cancel != nil {
		q.mu.Unlock()
		return
	}
	workerCtx, cancel := context.WithCancel(ctx)
	q.cancel = cancel
	q.wg.Add(q.workers)
	q.mu.Unlock()

	for i := 0; i < q.workers; i++ {
		go q.worker(workerCtx)
	}
}

// Stop cancels workers and waits for them to release their claimed batch.
func (q *Queue) Stop() {
	q.mu.Lock()
	cancel := q.cancel
	q.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	q.wg.Wait()
}

func (q *Queue) worker(ctx context.Context) {
	defer q.wg.Done()
	for {
		if ctx.Err() != nil {
			return
		}
		batch, matches, ok, err := q.store.ClaimPendingBatch(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			if !waitPoll(ctx) {
				return
			}
			continue
		}
		if !ok {
			select {
			case <-ctx.Done():
				return
			case <-time.After(pollInterval):
			}
			continue
		}
		q.processBatch(ctx, batch, matches)
	}
}

func (q *Queue) processBatch(ctx context.Context, batch store.Batch, matches []store.Match) {
	failedCount := 0
	doneCount := 0
	var firstError string
	for _, match := range matches {
		if ctx.Err() != nil {
			return
		}
		switch match.Status {
		case "done":
			doneCount++
			continue
		case "failed":
			failedCount++
			if firstError == "" {
				firstError = match.Error
			}
			continue
		}
		if err := q.processMatch(ctx, match); err != nil {
			if isCancellationError(ctx, err) {
				_ = q.store.ResetMatchPending(context.Background(), match.ID)
				return
			}
			failedCount++
			if firstError == "" {
				firstError = err.Error()
			}
			failCtx := ctx
			if failCtx.Err() != nil {
				failCtx = context.Background()
			}
			_ = q.store.FailMatch(failCtx, match.ID, err.Error())
			continue
		}
		doneCount++
	}
	if ctx.Err() != nil {
		return
	}

	status := "done"
	errText := ""
	if failedCount > 0 {
		if doneCount == 0 {
			status = "failed"
		} else {
			status = "partial"
		}
		if firstError == "" {
			firstError = "existing failed match"
		}
		errText = fmt.Sprintf("%d/%d matches failed: %s", failedCount, len(matches), firstError)
	}
	_ = q.store.FinishBatch(ctx, batch.ID, status, errText)
}

func (q *Queue) processMatch(ctx context.Context, match store.Match) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	agentA, err := q.store.GetAgent(ctx, match.AgentAID)
	if err != nil {
		return fmt.Errorf("load agent %d: %w", match.AgentAID, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	agentB, err := q.store.GetAgent(ctx, match.AgentBID)
	if err != nil {
		return fmt.Errorf("load agent %d: %w", match.AgentBID, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := q.store.MarkMatchRunning(ctx, match.ID); err != nil {
		return err
	}
	output, err := q.runner.Run(ctx, runner.Spec{
		Agents: []runner.AgentRef{
			{Name: agentA.Name, Path: runnerPath(agentA)},
			{Name: agentB.Name, Path: runnerPath(agentB)},
		},
		Games: match.Games,
		Seed:  match.Seed,
	})
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	games := make([]store.Game, 0, len(output.Games))
	ratingResults := make([]rating.GameResult, 0, len(output.Games))
	winA, draw, lossA := 0, 0, 0
	durationMs := 0
	for _, game := range output.Games {
		if err := ctx.Err(); err != nil {
			return err
		}
		scoreA, scoreB, err := scoresForMatch(game, agentA.Name, agentB.Name)
		if err != nil {
			return err
		}
		result := "draw"
		switch {
		case scoreA > scoreB:
			result = "win"
			winA++
		case scoreA < scoreB:
			result = "loss"
			lossA++
		default:
			draw++
		}
		if game.DurationMs > 0 {
			durationMs += game.DurationMs
		}
		ratingResults = append(ratingResults, rating.GameResult{
			MatchID:  match.ID,
			GameIdx:  game.Index,
			AgentIDs: []int64{agentA.ID, agentB.ID},
			Scores:   []float64{scoreA, scoreB},
		})
		games = append(games, store.Game{
			Index:      game.Index,
			Seed:       game.Seed,
			ScoreA:     scoreA,
			ScoreB:     scoreB,
			Result:     result,
			Turns:      game.Turns,
			DurationMs: game.DurationMs,
		})
	}
	if len(games) == 0 {
		return fmt.Errorf("arena returned no games for match %d", match.ID)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := q.store.RecordMatchResult(ctx, match.ID, games, winA, draw, lossA, durationMs); err != nil {
		return err
	}
	// Ratings are applied only after the match is committed so a retried
	// cancelled match cannot apply them twice.
	for _, result := range ratingResults {
		if err := q.rater.Update(context.WithoutCancel(ctx), result); err != nil {
			log.Printf("update rating for match %d game %d: %v", match.ID, result.GameIdx, err)
		}
	}
	return nil
}

func isCancellationError(ctx context.Context, err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ProcessState == nil || ctx.Err() == nil {
		return false
	}
	waitStatus, ok := exitErr.ProcessState.Sys().(syscall.WaitStatus)
	return ok && waitStatus.Signaled()
}

func runnerPath(agent store.Agent) string {
	if agent.Kind == "builtin" {
		return agent.BuiltinName
	}
	return agent.FilePath
}

func scoresForMatch(game runner.Game, nameA, nameB string) (float64, float64, error) {
	if scoreA, ok := game.Result[nameA]; ok {
		if scoreB, ok := game.Result[nameB]; ok {
			return scoreA, scoreB, nil
		}
	}
	if len(game.Players) >= 2 {
		scoreA, okA := game.Result[game.Players[0]]
		scoreB, okB := game.Result[game.Players[1]]
		if okA && okB {
			return scoreA, scoreB, nil
		}
	}
	return 0, 0, fmt.Errorf("arena returned scores without both match agents %q and %q", nameA, nameB)
}

func waitPoll(ctx context.Context) bool {
	timer := time.NewTimer(pollInterval)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
