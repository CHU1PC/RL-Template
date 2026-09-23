package queue

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"ladder/server/internal/db"
	"ladder/server/internal/rating"
	"ladder/server/internal/runner"
	"ladder/server/internal/store"
)

const queueTestArenaOutput = `{"games":[{"index":0,"seed":1,"players":["a","b"],"result":{"a":1,"b":0},"turns":1,"duration_ms":1}],"summary":{}}`

const queueTestArenaThreeGamesOutput = `{"games":[{"index":0,"seed":1,"players":["a","b"],"result":{"a":1,"b":0},"turns":1,"duration_ms":1},{"index":1,"seed":2,"players":["a","b"],"result":{"a":0,"b":1},"turns":2,"duration_ms":2},{"index":2,"seed":3,"players":["a","b"],"result":{"a":0,"b":0},"turns":3,"duration_ms":3}],"summary":{}}`

func TestProcessBatchStatusesAndCounts(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	database, err := db.Open(filepath.Join(t.TempDir(), "ladder.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	s := store.New(database)
	agents := make([]store.Agent, 0, 5)
	for _, name := range []string{"alpha", "beta", "gamma", "fail", "bad"} {
		agent, err := s.CreateAgent(ctx, store.Agent{Name: name, Kind: "builtin", BuiltinName: "random"})
		if err != nil {
			t.Fatal(err)
		}
		agents = append(agents, agent)
	}
	arenaPath := filepath.Join(t.TempDir(), "fake-arena.sh")
	script := "#!/bin/sh\ncase \"$*\" in\n  *fail*|*bad*) exit 7 ;;\nesac\nprintf '%s\\n' '" + queueTestArenaOutput + "'\n"
	if err := os.WriteFile(arenaPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	q := New(s, runner.New(arenaPath), 1, rating.NoopRater{})
	q.Start(ctx)
	t.Cleanup(q.Stop)

	doneBatch, err := s.CreateBatch(ctx, 1, [][2]int64{{agents[0].ID, agents[1].ID}})
	if err != nil {
		t.Fatal(err)
	}
	assertBatchStatus(t, s, doneBatch.ID, "done", 0, 1)

	partialBatch, err := s.CreateBatch(ctx, 1, [][2]int64{{agents[0].ID, agents[1].ID}, {agents[0].ID, agents[3].ID}})
	if err != nil {
		t.Fatal(err)
	}
	assertBatchStatus(t, s, partialBatch.ID, "partial", 1, 1)

	failedBatch, err := s.CreateBatch(ctx, 1, [][2]int64{{agents[3].ID, agents[4].ID}})
	if err != nil {
		t.Fatal(err)
	}
	assertBatchStatus(t, s, failedBatch.ID, "failed", 1, 0)
}

func TestCancellationRequeuesBatchAndMatches(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	database, err := db.Open(filepath.Join(t.TempDir(), "ladder.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	s := store.New(database)
	alpha, err := s.CreateAgent(ctx, store.Agent{Name: "alpha", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	beta, err := s.CreateAgent(ctx, store.Agent{Name: "beta", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	gamma, err := s.CreateAgent(ctx, store.Agent{Name: "gamma", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	slowArena := filepath.Join(t.TempDir(), "slow-arena.sh")
	slowScript := "#!/bin/sh\nsleep 10\nprintf '%s\\n' '" + queueTestArenaOutput + "'\n"
	if err := os.WriteFile(slowArena, []byte(slowScript), 0o755); err != nil {
		t.Fatal(err)
	}
	q := New(s, runner.New(slowArena), 1, rating.NoopRater{})
	q.Start(ctx)
	defer q.Stop()

	batch, err := s.CreateBatch(ctx, 1, [][2]int64{{alpha.ID, beta.ID}, {alpha.ID, gamma.ID}})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		_, matches, err := s.GetBatch(context.Background(), batch.ID)
		if err != nil {
			t.Fatal(err)
		}
		if len(matches) == 2 && matches[0].Status == "running" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	_, matches, err := s.GetBatch(context.Background(), batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 2 || matches[0].Status != "running" || matches[1].Status != "pending" {
		t.Fatalf("matches before cancellation = %#v, want running then pending", matches)
	}

	cancel()
	q.Stop()
	gotBatch, matches, err := s.GetBatch(context.Background(), batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if gotBatch.Status != "pending" && gotBatch.Status != "running" {
		t.Fatalf("batch after cancellation = %#v, want pending or running", gotBatch)
	}
	for _, match := range matches {
		if match.Status != "pending" {
			t.Fatalf("match after cancellation = %#v, want pending", matches)
		}
	}

	if _, err := s.RequeueRunningBatches(context.Background()); err != nil {
		t.Fatal(err)
	}
	fastArena := filepath.Join(t.TempDir(), "fast-arena.sh")
	fastScript := "#!/bin/sh\nprintf '%s\\n' '" + queueTestArenaOutput + "'\n"
	if err := os.WriteFile(fastArena, []byte(fastScript), 0o755); err != nil {
		t.Fatal(err)
	}
	q2 := New(s, runner.New(fastArena), 1, rating.NoopRater{})
	q2.Start(context.Background())
	defer q2.Stop()
	assertBatchStatus(t, s, batch.ID, "done", 0, 2)
}

func TestIsCancellationError(t *testing.T) {
	exit3 := wrappedProcessError(t, exec.Command("sh", "-c", "exit 3"))
	signal := wrappedProcessError(t, exec.Command("sh", "-c", "kill -KILL $$"))
	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name string
		ctx  context.Context
		err  error
		want bool
	}{
		{name: "exit3_live", ctx: context.Background(), err: exit3, want: false},
		{name: "exit3_cancelled", ctx: cancelledCtx, err: exit3, want: false},
		{name: "signal_live", ctx: context.Background(), err: signal, want: false},
		{name: "signal_cancelled", ctx: cancelledCtx, err: signal, want: true},
		{name: "context_cancelled", ctx: context.Background(), err: fmt.Errorf("wrapped: %w", context.Canceled), want: true},
		{name: "plain_error", ctx: cancelledCtx, err: errors.New("plain error"), want: false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isCancellationError(test.ctx, test.err); got != test.want {
				t.Fatalf("isCancellationError() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestExitFailureDuringCancelIsFailed(t *testing.T) {
	flagPath := filepath.Join(t.TempDir(), "cancel.flag")
	store, q, _, alpha, beta := newQueueTestFixture(t, "#!/bin/sh\ntouch "+flagPath+"\nexit 3\n", rating.NoopRater{})
	ctx := flagContext{Context: context.Background(), flagPath: flagPath}
	batch, err := store.CreateBatch(context.Background(), 1, [][2]int64{{alpha.ID, beta.ID}})
	if err != nil {
		t.Fatal(err)
	}
	claimedBatch, matches, ok, err := store.ClaimPendingBatch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("ClaimPendingBatch() returned no batch")
	}
	if claimedBatch.ID != batch.ID || len(matches) != 1 {
		t.Fatalf("claimed batch = %#v, matches = %#v", claimedBatch, matches)
	}
	q.processBatch(ctx, claimedBatch, matches)

	_, gotMatches, err := store.GetBatch(context.Background(), batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotMatches) != 1 {
		t.Fatalf("matches = %#v, want one match", gotMatches)
	}
	if gotMatches[0].Status != "failed" {
		t.Fatalf("match status = %q, want failed", gotMatches[0].Status)
	}
	if gotMatches[0].Error == "" {
		t.Fatal("match error is empty, want arena failure text")
	}
}

func TestRaterNotAppliedWhenCancelled(t *testing.T) {
	flagPath := filepath.Join(t.TempDir(), "cancel.flag")
	script := "#!/bin/sh\nprintf '%s\\n' '" + queueTestArenaThreeGamesOutput + "'\ntouch " + flagPath + "\n"
	rater := &recordingRater{}
	store, q, _, alpha, beta := newQueueTestFixture(t, script, rater)
	ctx := flagContext{Context: context.Background(), flagPath: flagPath}
	batch, err := store.CreateBatch(context.Background(), 3, [][2]int64{{alpha.ID, beta.ID}})
	if err != nil {
		t.Fatal(err)
	}
	claimedBatch, matches, ok, err := store.ClaimPendingBatch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("ClaimPendingBatch() returned no batch")
	}
	q.processBatch(ctx, claimedBatch, matches)

	_, gotMatches, err := store.GetBatch(context.Background(), batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotMatches) != 1 || gotMatches[0].Status != "pending" {
		t.Fatalf("matches = %#v, want one pending match", gotMatches)
	}
	if got := rater.resultsSnapshot(); len(got) != 0 {
		t.Fatalf("rating results = %#v, want none", got)
	}
}

func TestRaterAppliedOnlyAfterCommit(t *testing.T) {
	rater := &recordingRater{}
	store, q, ctx, alpha, beta := newQueueTestFixture(t, "#!/bin/sh\nprintf '%s\\n' '"+queueTestArenaThreeGamesOutput+"'\n", rater)
	batch, err := store.CreateBatch(ctx, 3, [][2]int64{{alpha.ID, beta.ID}})
	if err != nil {
		t.Fatal(err)
	}
	claimedBatch, matches, ok, err := store.ClaimPendingBatch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("ClaimPendingBatch() returned no batch")
	}
	q.processBatch(ctx, claimedBatch, matches)

	gotBatch, gotMatches, err := store.GetBatch(ctx, batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if gotBatch.Status != "done" || len(gotMatches) != 1 || gotMatches[0].Status != "done" {
		t.Fatalf("batch = %#v, matches = %#v, want done", gotBatch, gotMatches)
	}
	results := rater.resultsSnapshot()
	if len(results) != 3 {
		t.Fatalf("rating results = %#v, want three results", results)
	}
	for i, result := range results {
		if result.MatchID != gotMatches[0].ID || result.GameIdx != i {
			t.Fatalf("rating result %d = %#v, want match %d game %d", i, result, gotMatches[0].ID, i)
		}
	}
}

func TestRaterErrorKeepsMatchDone(t *testing.T) {
	rater := &recordingRater{err: errors.New("rater unavailable")}
	store, q, ctx, alpha, beta := newQueueTestFixture(t, "#!/bin/sh\nprintf '%s\\n' '"+queueTestArenaThreeGamesOutput+"'\n", rater)
	batch, err := store.CreateBatch(ctx, 3, [][2]int64{{alpha.ID, beta.ID}})
	if err != nil {
		t.Fatal(err)
	}
	claimedBatch, matches, ok, err := store.ClaimPendingBatch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("ClaimPendingBatch() returned no batch")
	}
	q.processBatch(ctx, claimedBatch, matches)

	gotBatch, gotMatches, err := store.GetBatch(ctx, batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if gotBatch.Status != "done" || len(gotMatches) != 1 || gotMatches[0].Status != "done" {
		t.Fatalf("batch = %#v, matches = %#v, want done", gotBatch, gotMatches)
	}
	if got := len(rater.resultsSnapshot()); got != 3 {
		t.Fatalf("rating result count = %d, want 3", got)
	}
}

type flagContext struct {
	context.Context
	flagPath string
}

func (c flagContext) Done() <-chan struct{} { return nil }

func (c flagContext) Err() error {
	if _, err := os.Stat(c.flagPath); err == nil {
		return context.Canceled
	}
	return nil
}

type recordingRater struct {
	mu      sync.Mutex
	results []rating.GameResult
	err     error
}

func (r *recordingRater) Update(_ context.Context, result rating.GameResult) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.results = append(r.results, result)
	return r.err
}

func (r *recordingRater) resultsSnapshot() []rating.GameResult {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]rating.GameResult(nil), r.results...)
}

func newQueueTestFixture(t *testing.T, script string, rater rating.Rater) (*store.Store, *Queue, context.Context, store.Agent, store.Agent) {
	t.Helper()
	ctx := flagContext{Context: context.Background(), flagPath: filepath.Join(t.TempDir(), "never-created.flag")}
	database, err := db.Open(filepath.Join(t.TempDir(), "ladder.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	s := store.New(database)
	alpha, err := s.CreateAgent(ctx, store.Agent{Name: "alpha", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	beta, err := s.CreateAgent(ctx, store.Agent{Name: "beta", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	arenaPath := filepath.Join(t.TempDir(), "fake-arena.sh")
	if err := os.WriteFile(arenaPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return s, New(s, runner.New(arenaPath), 1, rater), ctx, alpha, beta
}

func wrappedProcessError(t *testing.T, cmd *exec.Cmd) error {
	t.Helper()
	if err := cmd.Run(); err == nil {
		t.Fatal("command unexpectedly succeeded")
	} else {
		return fmt.Errorf("arena failed: %w", err)
	}
	return nil
}

func assertBatchStatus(t *testing.T, s *store.Store, id int64, wantStatus string, wantFailed, wantDone int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		batch, matches, err := s.GetBatch(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if batch.Status == wantStatus {
			if batch.FailedCount != wantFailed || batch.DoneCount != wantDone {
				t.Fatalf("batch = %#v, want counts failed=%d done=%d", batch, wantFailed, wantDone)
			}
			if len(matches) != wantFailed+wantDone {
				t.Fatalf("matches = %#v", matches)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	batch, _, err := s.GetBatch(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	t.Fatalf("batch status = %q, want %q", batch.Status, wantStatus)
}
