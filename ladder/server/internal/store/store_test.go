package store

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"testing"

	"ladder/server/internal/db"
)

func testStore(t *testing.T) (*Store, *sql.DB) {
	t.Helper()
	database, err := db.Open(filepath.Join(t.TempDir(), "ladder.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	return New(database), database
}

func TestStoreCRUDAndResults(t *testing.T) {
	ctx := context.Background()
	s, _ := testStore(t)
	a, err := s.CreateAgent(ctx, Agent{Name: "alpha", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.CreateAgent(ctx, Agent{Name: "beta", Kind: "onnx", FilePath: "/tmp/beta.onnx"})
	if err != nil {
		t.Fatal(err)
	}
	if a.RatingMu != 600 || a.RatingSigma != 200 || b.RatingMu != 600 || b.RatingSigma != 200 {
		t.Fatalf("initial ratings = %v, %v", a, b)
	}
	if _, err := s.CreateAgent(ctx, Agent{Name: "alpha", Kind: "builtin"}); !errors.Is(err, ErrDuplicate) {
		t.Fatalf("duplicate error = %v", err)
	}
	group, err := s.CreateGroup(ctx, "all")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.AddAgentToGroup(ctx, group.ID, a.ID); err != nil {
		t.Fatal(err)
	}
	if err := s.AddAgentToGroup(ctx, group.ID, a.ID); err != nil {
		t.Fatal(err)
	}
	if err := s.AddAgentToGroup(ctx, group.ID, b.ID); err != nil {
		t.Fatal(err)
	}
	groupAgents, err := s.ListGroupAgents(ctx, group.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(groupAgents) != 2 {
		t.Fatalf("group agents = %d, want 2", len(groupAgents))
	}

	batch, err := s.CreateBatch(ctx, 3, [][2]int64{{a.ID, b.ID}})
	if err != nil {
		t.Fatal(err)
	}
	claimed, matches, ok, err := s.ClaimPendingBatch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || claimed.ID != batch.ID || claimed.Status != "running" || len(matches) != 1 {
		t.Fatalf("claim = %#v, %#v, %v", claimed, matches, ok)
	}
	if matches[0].AgentAName != "alpha" || matches[0].AgentBName != "beta" {
		t.Fatalf("joined names = %#v", matches[0])
	}
	_, _, ok, err = s.ClaimPendingBatch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("second claim unexpectedly succeeded")
	}

	matchID := matches[0].ID
	games := []Game{
		{Index: 0, Seed: 10, ScoreA: 1, ScoreB: 0, Result: "win", Turns: 5, DurationMs: 2, ReplayPath: "r0"},
		{Index: 1, Seed: 11, ScoreA: 0.5, ScoreB: 0.5, Result: "draw", Turns: 6, DurationMs: 3, ReplayPath: "r1"},
		{Index: 2, Seed: 12, ScoreA: 1, ScoreB: 0, Result: "win", Turns: 7, DurationMs: 4, ReplayPath: "r2"},
	}
	if err := s.RecordMatchResult(ctx, matchID, games, 2, 1, 0, 9); err != nil {
		t.Fatal(err)
	}
	if err := s.FinishBatch(ctx, batch.ID, "done", ""); err != nil {
		t.Fatal(err)
	}
	ranking, err := s.Ranking(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ranking) != 2 {
		t.Fatalf("ranking length = %d", len(ranking))
	}
	if ranking[0].Name != "alpha" || ranking[0].Games != 3 || ranking[0].Win != 2 || ranking[0].Draw != 1 || ranking[0].Loss != 0 || ranking[0].WinRate != 5.0/6.0 {
		t.Fatalf("alpha ranking = %#v", ranking[0])
	}
	if ranking[1].Name != "beta" || ranking[1].Games != 3 || ranking[1].Win != 0 || ranking[1].Draw != 1 || ranking[1].Loss != 2 || ranking[1].WinRate != 1.0/6.0 {
		t.Fatalf("beta ranking = %#v", ranking[1])
	}
	gotMatch, gotGames, err := s.GetMatch(ctx, matchID)
	if err != nil {
		t.Fatal(err)
	}
	if gotMatch.Status != "done" || gotMatch.WinA != 2 || gotMatch.Draw != 1 || gotMatch.LossA != 0 || gotMatch.DurationMs != 9 || len(gotGames) != 3 {
		t.Fatalf("match = %#v games = %#v", gotMatch, gotGames)
	}
	if gotGames[2].ReplayPath != "r2" || gotGames[2].Index != 2 {
		t.Fatalf("games = %#v", gotGames)
	}
}

func TestRequeueRunningBatches(t *testing.T) {
	ctx := context.Background()
	s, database := testStore(t)
	alpha, err := s.CreateAgent(ctx, Agent{Name: "alpha", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	beta, err := s.CreateAgent(ctx, Agent{Name: "beta", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := s.CreateBatch(ctx, 1, [][2]int64{{alpha.ID, beta.ID}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `UPDATE batches SET status = 'running', started_at = 'started' WHERE id = ?`, batch.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `UPDATE matches SET status = 'running' WHERE batch_id = ?`, batch.ID); err != nil {
		t.Fatal(err)
	}

	count, err := s.RequeueRunningBatches(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("requeued count = %d, want 1", count)
	}
	gotBatch, matches, err := s.GetBatch(ctx, batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if gotBatch.Status != "pending" || len(matches) != 1 || matches[0].Status != "pending" {
		t.Fatalf("requeued batch = %#v, matches = %#v", gotBatch, matches)
	}
}

func TestConfiguredAgentRatingsArePersisted(t *testing.T) {
	ctx := context.Background()
	database, err := db.Open(filepath.Join(t.TempDir(), "ladder.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	s := NewWithRatingDefaults(database, 712.5, 44.25)
	agent, err := s.CreateAgent(ctx, Agent{Name: "configured", Kind: "builtin", BuiltinName: "random"})
	if err != nil {
		t.Fatal(err)
	}
	if agent.RatingMu != 712.5 || agent.RatingSigma != 44.25 {
		t.Fatalf("created agent ratings = %#v", agent)
	}
	persisted, err := s.GetAgent(ctx, agent.ID)
	if err != nil {
		t.Fatal(err)
	}
	if persisted.RatingMu != 712.5 || persisted.RatingSigma != 44.25 {
		t.Fatalf("persisted agent ratings = %#v", persisted)
	}
}
