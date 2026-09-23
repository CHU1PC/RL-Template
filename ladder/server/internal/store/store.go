package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"time"

	"ladder/server/internal/pairing"
)

var ErrNotFound = errors.New("not found")
var ErrDuplicate = errors.New("duplicate")

type Store struct {
	db           *sql.DB
	ratingMu0    float64
	ratingSigma0 float64
}

type Agent struct {
	ID          int64
	Name        string
	Kind        string
	FilePath    string
	BuiltinName string
	RatingMu    float64
	RatingSigma float64
	CreatedAt   time.Time
}

type Group struct {
	ID        int64
	Name      string
	CreatedAt time.Time
}

type Batch struct {
	ID          int64
	Status      string
	Games       int
	Error       string
	CreatedAt   time.Time
	StartedAt   time.Time
	FinishedAt  time.Time
	FailedCount int
	DoneCount   int
}

type Match struct {
	ID         int64
	BatchID    int64
	AgentAID   int64
	AgentBID   int64
	AgentAName string
	AgentBName string
	Games      int
	Seed       int64
	Status     string
	Error      string
	WinA       int
	Draw       int
	LossA      int
	DurationMs int
	CreatedAt  time.Time
}

type Game struct {
	ID         int64
	MatchID    int64
	Index      int
	Seed       int64
	ScoreA     float64
	ScoreB     float64
	Result     string
	Turns      int
	DurationMs int
	ReplayPath string
}

type RankingRow struct {
	AgentID int64
	Name    string
	Mu      float64
	Sigma   float64
	Games   int
	Win     int
	Draw    int
	Loss    int
	WinRate float64
}

func New(db *sql.DB) *Store {
	return NewWithRatingDefaults(db, 600, 200)
}

func NewWithRatingDefaults(db *sql.DB, ratingMu0, ratingSigma0 float64) *Store {
	return &Store{db: db, ratingMu0: ratingMu0, ratingSigma0: ratingSigma0}
}

func (s *Store) CreateAgent(ctx context.Context, a Agent) (Agent, error) {
	createdAt := a.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}
	createdAt = createdAt.UTC()
	result, err := s.db.ExecContext(ctx, `
		INSERT INTO agents(name, kind, file_path, builtin_name, rating_mu, rating_sigma, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		a.Name, a.Kind, a.FilePath, a.BuiltinName, s.ratingMu0, s.ratingSigma0, formatTime(createdAt))
	if err != nil {
		return Agent{}, wrapDBError("create agent", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return Agent{}, fmt.Errorf("create agent: get id: %w", err)
	}
	a.ID = id
	a.RatingMu = s.ratingMu0
	a.RatingSigma = s.ratingSigma0
	a.CreatedAt = createdAt
	return a, nil
}

func (s *Store) ListAgents(ctx context.Context) ([]Agent, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, name, kind, file_path, builtin_name, rating_mu, rating_sigma, created_at
		FROM agents ORDER BY id ASC`)
	if err != nil {
		return nil, fmt.Errorf("list agents: %w", err)
	}
	defer rows.Close()
	agents := make([]Agent, 0)
	for rows.Next() {
		agent, err := scanAgent(rows)
		if err != nil {
			return nil, fmt.Errorf("list agents: %w", err)
		}
		agents = append(agents, agent)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list agents: %w", err)
	}
	return agents, nil
}

func (s *Store) GetAgent(ctx context.Context, id int64) (Agent, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, name, kind, file_path, builtin_name, rating_mu, rating_sigma, created_at
		FROM agents WHERE id = ?`, id)
	agent, err := scanAgent(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Agent{}, fmt.Errorf("get agent %d: %w", id, ErrNotFound)
	}
	if err != nil {
		return Agent{}, fmt.Errorf("get agent %d: %w", id, err)
	}
	return agent, nil
}

func (s *Store) CreateGroup(ctx context.Context, name string) (Group, error) {
	createdAt := time.Now().UTC()
	result, err := s.db.ExecContext(ctx, `
		INSERT INTO "groups"(name, created_at) VALUES (?, ?)`, name, formatTime(createdAt))
	if err != nil {
		return Group{}, wrapDBError("create group", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return Group{}, fmt.Errorf("create group: get id: %w", err)
	}
	return Group{ID: id, Name: name, CreatedAt: createdAt}, nil
}

func (s *Store) GetGroup(ctx context.Context, id int64) (Group, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, name, created_at FROM "groups" WHERE id = ?`, id)
	group, err := scanGroup(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Group{}, fmt.Errorf("get group %d: %w", id, ErrNotFound)
	}
	if err != nil {
		return Group{}, fmt.Errorf("get group %d: %w", id, err)
	}
	return group, nil
}

func (s *Store) ListGroups(ctx context.Context) ([]Group, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, name, created_at FROM "groups" ORDER BY id ASC`)
	if err != nil {
		return nil, fmt.Errorf("list groups: %w", err)
	}
	defer rows.Close()
	groups := make([]Group, 0)
	for rows.Next() {
		group, err := scanGroup(rows)
		if err != nil {
			return nil, fmt.Errorf("list groups: %w", err)
		}
		groups = append(groups, group)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list groups: %w", err)
	}
	return groups, nil
}

func (s *Store) AddAgentToGroup(ctx context.Context, groupID, agentID int64) error {
	if _, err := s.db.ExecContext(ctx, `
		INSERT OR IGNORE INTO agent_groups(agent_id, group_id) VALUES (?, ?)`, agentID, groupID); err != nil {
		return fmt.Errorf("add agent to group: %w", err)
	}
	return nil
}

func (s *Store) ListGroupAgents(ctx context.Context, groupID int64) ([]Agent, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT a.id, a.name, a.kind, a.file_path, a.builtin_name, a.rating_mu, a.rating_sigma, a.created_at
		FROM agents a
		JOIN agent_groups ag ON ag.agent_id = a.id
		JOIN "groups" g ON g.id = ag.group_id
		WHERE g.id = ?
		ORDER BY a.id ASC`, groupID)
	if err != nil {
		return nil, fmt.Errorf("list group agents: %w", err)
	}
	defer rows.Close()
	agents := make([]Agent, 0)
	for rows.Next() {
		agent, err := scanAgent(rows)
		if err != nil {
			return nil, fmt.Errorf("list group agents: %w", err)
		}
		agents = append(agents, agent)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list group agents: %w", err)
	}
	return agents, nil
}

func (s *Store) CreateBatch(ctx context.Context, games int, pairs [][2]int64) (Batch, error) {
	if games < 1 {
		return Batch{}, fmt.Errorf("create batch: games must be at least 1")
	}
	createdAt := time.Now().UTC()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return Batch{}, fmt.Errorf("create batch: begin transaction: %w", err)
	}
	rollback := func(cause error) (Batch, error) {
		_ = tx.Rollback()
		return Batch{}, cause
	}
	result, err := tx.ExecContext(ctx, `
		INSERT INTO batches(status, games, error, created_at, started_at, finished_at)
		VALUES ('pending', ?, '', ?, '', '')`, games, formatTime(createdAt))
	if err != nil {
		return rollback(fmt.Errorf("create batch: insert batch: %w", err))
	}
	batchID, err := result.LastInsertId()
	if err != nil {
		return rollback(fmt.Errorf("create batch: get id: %w", err))
	}
	for _, pair := range pairs {
		seed := rand.Int64()
		if seed == 0 {
			seed = 1
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO matches(batch_id, agent_a_id, agent_b_id, games, seed, status, error, win_a, draw, loss_a, duration_ms, created_at)
			VALUES (?, ?, ?, ?, ?, 'pending', '', 0, 0, 0, 0, ?)`,
			batchID, pair[0], pair[1], games, seed, formatTime(createdAt)); err != nil {
			return rollback(fmt.Errorf("create batch: insert match: %w", err))
		}
	}
	if err := tx.Commit(); err != nil {
		return Batch{}, fmt.Errorf("create batch: commit transaction: %w", err)
	}
	return Batch{ID: batchID, Status: "pending", Games: games, CreatedAt: createdAt}, nil
}

func (s *Store) SetAgentFilePath(ctx context.Context, id int64, filePath string) error {
	result, err := s.db.ExecContext(ctx, `UPDATE agents SET file_path = ? WHERE id = ?`, filePath, id)
	if err != nil {
		return fmt.Errorf("set agent file path: %w", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("set agent file path: rows affected: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("set agent file path %d: %w", id, ErrNotFound)
	}
	return nil
}

func (s *Store) ListBatches(ctx context.Context, limit int) ([]Batch, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT b.id, b.status, b.games, b.error, b.created_at, b.started_at, b.finished_at,
		       (SELECT COUNT(*) FROM matches m WHERE m.batch_id = b.id AND m.status = 'failed'),
		       (SELECT COUNT(*) FROM matches m WHERE m.batch_id = b.id AND m.status = 'done')
		FROM batches b ORDER BY b.id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, fmt.Errorf("list batches: %w", err)
	}
	defer rows.Close()
	batches := make([]Batch, 0)
	for rows.Next() {
		batch, err := scanBatch(rows)
		if err != nil {
			return nil, fmt.Errorf("list batches: %w", err)
		}
		batches = append(batches, batch)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list batches: %w", err)
	}
	return batches, nil
}

func (s *Store) GetBatch(ctx context.Context, id int64) (Batch, []Match, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT b.id, b.status, b.games, b.error, b.created_at, b.started_at, b.finished_at,
		       (SELECT COUNT(*) FROM matches m WHERE m.batch_id = b.id AND m.status = 'failed'),
		       (SELECT COUNT(*) FROM matches m WHERE m.batch_id = b.id AND m.status = 'done')
		FROM batches b WHERE b.id = ?`, id)
	batch, err := scanBatch(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Batch{}, nil, fmt.Errorf("get batch %d: %w", id, ErrNotFound)
	}
	if err != nil {
		return Batch{}, nil, fmt.Errorf("get batch %d: %w", id, err)
	}
	matches, err := s.matchesForBatch(ctx, id)
	if err != nil {
		return Batch{}, nil, err
	}
	return batch, matches, nil
}

func (s *Store) ClaimPendingBatch(ctx context.Context) (b Batch, ms []Match, ok bool, err error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return Batch{}, nil, false, fmt.Errorf("claim pending batch: begin transaction: %w", err)
	}
	rollback := func(cause error) (Batch, []Match, bool, error) {
		_ = tx.Rollback()
		return Batch{}, nil, false, cause
	}
	var id int64
	err = tx.QueryRowContext(ctx, `
		SELECT id FROM batches WHERE status = 'pending' ORDER BY created_at ASC, id ASC LIMIT 1`).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		_ = tx.Rollback()
		return Batch{}, nil, false, nil
	}
	if err != nil {
		return rollback(fmt.Errorf("claim pending batch: find batch: %w", err))
	}
	startedAt := time.Now().UTC()
	result, err := tx.ExecContext(ctx, `
		UPDATE batches SET status = 'running', started_at = ? WHERE id = ? AND status = 'pending'`,
		formatTime(startedAt), id)
	if err != nil {
		return rollback(fmt.Errorf("claim pending batch: update batch: %w", err))
	}
	count, err := result.RowsAffected()
	if err != nil {
		return rollback(fmt.Errorf("claim pending batch: rows affected: %w", err))
	}
	if count != 1 {
		return rollback(fmt.Errorf("claim pending batch: %w", ErrNotFound))
	}
	row := tx.QueryRowContext(ctx, `
		SELECT b.id, b.status, b.games, b.error, b.created_at, b.started_at, b.finished_at,
		       (SELECT COUNT(*) FROM matches m WHERE m.batch_id = b.id AND m.status = 'failed'),
		       (SELECT COUNT(*) FROM matches m WHERE m.batch_id = b.id AND m.status = 'done')
		FROM batches b WHERE b.id = ?`, id)
	b, err = scanBatch(row)
	if err != nil {
		return rollback(fmt.Errorf("claim pending batch: read batch: %w", err))
	}
	matches, err := matchesForBatchTx(ctx, tx, id)
	if err != nil {
		return rollback(err)
	}
	if err := tx.Commit(); err != nil {
		return Batch{}, nil, false, fmt.Errorf("claim pending batch: commit transaction: %w", err)
	}
	return b, matches, true, nil
}

// RequeueRunningBatches makes batches and matches interrupted by a process
// termination eligible for the next queue worker.
func (s *Store) RequeueRunningBatches(ctx context.Context) (int64, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("requeue running batches: begin transaction: %w", err)
	}
	rollback := func(cause error) (int64, error) {
		_ = tx.Rollback()
		return 0, cause
	}
	if _, err := tx.ExecContext(ctx, `
		UPDATE matches SET status = 'pending', error = ''
		WHERE status = 'running'
		  AND batch_id IN (SELECT id FROM batches WHERE status = 'running')`); err != nil {
		return rollback(fmt.Errorf("requeue running batches: reset matches: %w", err))
	}
	result, err := tx.ExecContext(ctx, `
		UPDATE batches SET status = 'pending', error = '', started_at = '', finished_at = ''
		WHERE status = 'running'`)
	if err != nil {
		return rollback(fmt.Errorf("requeue running batches: reset batches: %w", err))
	}
	count, err := result.RowsAffected()
	if err != nil {
		return rollback(fmt.Errorf("requeue running batches: rows affected: %w", err))
	}
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("requeue running batches: commit transaction: %w", err)
	}
	return count, nil
}

// MarkMatchRunning claims a pending match for the current queue worker.
func (s *Store) MarkMatchRunning(ctx context.Context, id int64) error {
	result, err := s.db.ExecContext(ctx, `
		UPDATE matches SET status = 'running', error = '' WHERE id = ? AND status = 'pending'`, id)
	if err != nil {
		return fmt.Errorf("mark match running: %w", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("mark match running: rows affected: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("mark match running %d: %w", id, ErrNotFound)
	}
	return nil
}

// ResetMatchPending returns an interrupted match to the queue.
func (s *Store) ResetMatchPending(ctx context.Context, id int64) error {
	if _, err := s.db.ExecContext(ctx, `
		UPDATE matches SET status = 'pending', error = '' WHERE id = ? AND status = 'running'`, id); err != nil {
		return fmt.Errorf("reset match pending: %w", err)
	}
	return nil
}

func (s *Store) FinishBatch(ctx context.Context, id int64, status string, errText string) error {
	result, err := s.db.ExecContext(ctx, `
		UPDATE batches SET status = ?, error = ?, finished_at = ? WHERE id = ?`,
		status, errText, formatTime(time.Now().UTC()), id)
	if err != nil {
		return fmt.Errorf("finish batch: %w", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("finish batch: rows affected: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("finish batch %d: %w", id, ErrNotFound)
	}
	return nil
}

func (s *Store) ListMatches(ctx context.Context, agentID int64, limit int) ([]Match, error) {
	if limit <= 0 {
		limit = 100
	}
	query := `
		SELECT m.id, m.batch_id, m.agent_a_id, m.agent_b_id, a.name, b.name,
		       m.games, m.seed, m.status, m.error, m.win_a, m.draw, m.loss_a,
		       m.duration_ms, m.created_at
		FROM matches m
		JOIN agents a ON a.id = m.agent_a_id
		JOIN agents b ON b.id = m.agent_b_id`
	args := []any{}
	if agentID != 0 {
		query += " WHERE m.agent_a_id = ? OR m.agent_b_id = ?"
		args = append(args, agentID, agentID)
	}
	query += " ORDER BY m.id DESC LIMIT ?"
	args = append(args, limit)
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list matches: %w", err)
	}
	defer rows.Close()
	matches := make([]Match, 0)
	for rows.Next() {
		match, err := scanMatch(rows)
		if err != nil {
			return nil, fmt.Errorf("list matches: %w", err)
		}
		matches = append(matches, match)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list matches: %w", err)
	}
	return matches, nil
}

func (s *Store) GetMatch(ctx context.Context, id int64) (Match, []Game, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT m.id, m.batch_id, m.agent_a_id, m.agent_b_id, a.name, b.name,
		       m.games, m.seed, m.status, m.error, m.win_a, m.draw, m.loss_a,
		       m.duration_ms, m.created_at
		FROM matches m
		JOIN agents a ON a.id = m.agent_a_id
		JOIN agents b ON b.id = m.agent_b_id
		WHERE m.id = ?`, id)
	match, err := scanMatch(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Match{}, nil, fmt.Errorf("get match %d: %w", id, ErrNotFound)
	}
	if err != nil {
		return Match{}, nil, fmt.Errorf("get match %d: %w", id, err)
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, match_id, idx, seed, score_a, score_b, result, turns, duration_ms, replay_path
		FROM games WHERE match_id = ? ORDER BY idx ASC`, id)
	if err != nil {
		return Match{}, nil, fmt.Errorf("get match %d games: %w", id, err)
	}
	defer rows.Close()
	games := make([]Game, 0)
	for rows.Next() {
		game, err := scanGame(rows)
		if err != nil {
			return Match{}, nil, fmt.Errorf("get match %d games: %w", id, err)
		}
		games = append(games, game)
	}
	if err := rows.Err(); err != nil {
		return Match{}, nil, fmt.Errorf("get match %d games: %w", id, err)
	}
	return match, games, nil
}

func (s *Store) RecordMatchResult(ctx context.Context, matchID int64, games []Game, winA, draw, lossA, durationMs int) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("record match result: begin transaction: %w", err)
	}
	rollback := func(cause error) error {
		_ = tx.Rollback()
		return cause
	}
	for _, game := range games {
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO games(match_id, idx, seed, score_a, score_b, result, turns, duration_ms, replay_path)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			matchID, game.Index, game.Seed, game.ScoreA, game.ScoreB, game.Result,
			game.Turns, game.DurationMs, game.ReplayPath); err != nil {
			return rollback(fmt.Errorf("record match result: insert game: %w", err))
		}
	}
	result, err := tx.ExecContext(ctx, `
		UPDATE matches
		SET status = 'done', error = '', win_a = ?, draw = ?, loss_a = ?, duration_ms = ?
		WHERE id = ?`, winA, draw, lossA, durationMs, matchID)
	if err != nil {
		return rollback(fmt.Errorf("record match result: update match: %w", err))
	}
	count, err := result.RowsAffected()
	if err != nil {
		return rollback(fmt.Errorf("record match result: rows affected: %w", err))
	}
	if count == 0 {
		return rollback(fmt.Errorf("record match result %d: %w", matchID, ErrNotFound))
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("record match result: commit transaction: %w", err)
	}
	return nil
}

func (s *Store) FailMatch(ctx context.Context, matchID int64, errText string) error {
	result, err := s.db.ExecContext(ctx, `
		UPDATE matches SET status = 'failed', error = ? WHERE id = ?`, errText, matchID)
	if err != nil {
		return fmt.Errorf("fail match: %w", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("fail match: rows affected: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("fail match %d: %w", matchID, ErrNotFound)
	}
	return nil
}

func (s *Store) Ranking(ctx context.Context) ([]RankingRow, error) {
	rows, err := s.db.QueryContext(ctx, `
		WITH participant_results AS (
			SELECT agent_a_id AS agent_id, games, win_a AS win, draw, loss_a AS loss
			FROM matches WHERE status = 'done'
			UNION ALL
			SELECT agent_b_id AS agent_id, games, loss_a AS win, draw, win_a AS loss
			FROM matches WHERE status = 'done'
		), stats AS (
			SELECT agent_id, SUM(games) AS games, SUM(win) AS win, SUM(draw) AS draw, SUM(loss) AS loss
			FROM participant_results GROUP BY agent_id
		)
		SELECT a.id, a.name, a.rating_mu, a.rating_sigma,
		       COALESCE(stats.games, 0), COALESCE(stats.win, 0),
		       COALESCE(stats.draw, 0), COALESCE(stats.loss, 0),
		       CASE WHEN COALESCE(stats.games, 0) = 0 THEN 0.0
		            ELSE (COALESCE(stats.win, 0) + 0.5 * COALESCE(stats.draw, 0)) / stats.games
		       END AS win_rate
		FROM agents a
		LEFT JOIN stats ON stats.agent_id = a.id
		ORDER BY a.rating_mu DESC, win_rate DESC, a.name ASC`)
	if err != nil {
		return nil, fmt.Errorf("ranking: %w", err)
	}
	defer rows.Close()
	ranking := make([]RankingRow, 0)
	for rows.Next() {
		var row RankingRow
		if err := rows.Scan(&row.AgentID, &row.Name, &row.Mu, &row.Sigma, &row.Games, &row.Win, &row.Draw, &row.Loss, &row.WinRate); err != nil {
			return nil, fmt.Errorf("ranking: %w", err)
		}
		ranking = append(ranking, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("ranking: %w", err)
	}
	return ranking, nil
}

func (s *Store) AgentRatings(ctx context.Context) ([]pairing.AgentRating, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id, rating_mu FROM agents ORDER BY id ASC`)
	if err != nil {
		return nil, fmt.Errorf("agent ratings: %w", err)
	}
	defer rows.Close()
	ratings := make([]pairing.AgentRating, 0)
	for rows.Next() {
		var rating pairing.AgentRating
		if err := rows.Scan(&rating.ID, &rating.Mu); err != nil {
			return nil, fmt.Errorf("agent ratings: %w", err)
		}
		ratings = append(ratings, rating)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("agent ratings: %w", err)
	}
	return ratings, nil
}

type scanner interface {
	Scan(dest ...any) error
}

func scanAgent(row scanner) (Agent, error) {
	var agent Agent
	var createdAt string
	err := row.Scan(&agent.ID, &agent.Name, &agent.Kind, &agent.FilePath, &agent.BuiltinName,
		&agent.RatingMu, &agent.RatingSigma, &createdAt)
	if err != nil {
		return Agent{}, err
	}
	agent.CreatedAt, err = parseTime(createdAt)
	if err != nil {
		return Agent{}, err
	}
	return agent, nil
}

func scanGroup(row scanner) (Group, error) {
	var group Group
	var createdAt string
	err := row.Scan(&group.ID, &group.Name, &createdAt)
	if err != nil {
		return Group{}, err
	}
	group.CreatedAt, err = parseTime(createdAt)
	if err != nil {
		return Group{}, err
	}
	return group, nil
}

func scanBatch(row scanner) (Batch, error) {
	var batch Batch
	var createdAt, startedAt, finishedAt string
	err := row.Scan(&batch.ID, &batch.Status, &batch.Games, &batch.Error, &createdAt, &startedAt, &finishedAt,
		&batch.FailedCount, &batch.DoneCount)
	if err != nil {
		return Batch{}, err
	}
	batch.CreatedAt, err = parseTime(createdAt)
	if err != nil {
		return Batch{}, err
	}
	batch.StartedAt, err = parseTime(startedAt)
	if err != nil {
		return Batch{}, err
	}
	batch.FinishedAt, err = parseTime(finishedAt)
	if err != nil {
		return Batch{}, err
	}
	return batch, nil
}

func scanMatch(row scanner) (Match, error) {
	var match Match
	var createdAt string
	err := row.Scan(&match.ID, &match.BatchID, &match.AgentAID, &match.AgentBID,
		&match.AgentAName, &match.AgentBName, &match.Games, &match.Seed, &match.Status,
		&match.Error, &match.WinA, &match.Draw, &match.LossA, &match.DurationMs, &createdAt)
	if err != nil {
		return Match{}, err
	}
	match.CreatedAt, err = parseTime(createdAt)
	if err != nil {
		return Match{}, err
	}
	return match, nil
}

func scanGame(row scanner) (Game, error) {
	var game Game
	err := row.Scan(&game.ID, &game.MatchID, &game.Index, &game.Seed, &game.ScoreA, &game.ScoreB,
		&game.Result, &game.Turns, &game.DurationMs, &game.ReplayPath)
	return game, err
}

func (s *Store) matchesForBatch(ctx context.Context, batchID int64) ([]Match, error) {
	return matchesForBatchDB(ctx, s.db, batchID)
}

type queryer interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
}

func matchesForBatchDB(ctx context.Context, q queryer, batchID int64) ([]Match, error) {
	rows, err := q.QueryContext(ctx, `
		SELECT m.id, m.batch_id, m.agent_a_id, m.agent_b_id, a.name, b.name,
		       m.games, m.seed, m.status, m.error, m.win_a, m.draw, m.loss_a,
		       m.duration_ms, m.created_at
		FROM matches m
		JOIN agents a ON a.id = m.agent_a_id
		JOIN agents b ON b.id = m.agent_b_id
		WHERE m.batch_id = ? ORDER BY m.id ASC`, batchID)
	if err != nil {
		return nil, fmt.Errorf("get batch %d matches: %w", batchID, err)
	}
	defer rows.Close()
	matches := make([]Match, 0)
	for rows.Next() {
		match, err := scanMatch(rows)
		if err != nil {
			return nil, fmt.Errorf("get batch %d matches: %w", batchID, err)
		}
		matches = append(matches, match)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("get batch %d matches: %w", batchID, err)
	}
	return matches, nil
}

func matchesForBatchTx(ctx context.Context, tx *sql.Tx, batchID int64) ([]Match, error) {
	return matchesForBatchDB(ctx, tx, batchID)
}

func formatTime(value time.Time) string {
	return value.UTC().Format(time.RFC3339Nano)
}

func parseTime(value string) (time.Time, error) {
	if value == "" {
		return time.Time{}, nil
	}
	return time.Parse(time.RFC3339Nano, value)
}

func wrapDBError(operation string, err error) error {
	if strings.Contains(strings.ToLower(err.Error()), "unique constraint failed") {
		return fmt.Errorf("%s: %w: %v", operation, ErrDuplicate, err)
	}
	return fmt.Errorf("%s: %w", operation, err)
}
