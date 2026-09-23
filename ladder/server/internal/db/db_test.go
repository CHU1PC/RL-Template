package db_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"ladder/server/internal/db"
	"ladder/server/internal/store"
	_ "modernc.org/sqlite"
)

const legacyInitSQL = `
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK(kind IN ('onnx','builtin')),
    file_path TEXT NOT NULL DEFAULT '',
    builtin_name TEXT NOT NULL DEFAULT '',
    rating_mu REAL NOT NULL DEFAULT 600,
    rating_sigma REAL NOT NULL DEFAULT 200,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "groups" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_groups (
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    group_id INTEGER NOT NULL REFERENCES "groups"(id) ON DELETE CASCADE,
    PRIMARY KEY(agent_id, group_id)
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL CHECK(status IN ('pending','running','done','failed')),
    games INTEGER NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    agent_a_id INTEGER NOT NULL REFERENCES agents(id),
    agent_b_id INTEGER NOT NULL REFERENCES agents(id),
    games INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','done','failed')),
    error TEXT NOT NULL DEFAULT '',
    win_a INTEGER NOT NULL DEFAULT 0,
    draw INTEGER NOT NULL DEFAULT 0,
    loss_a INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    score_a REAL NOT NULL,
    score_b REAL NOT NULL,
    result TEXT NOT NULL CHECK(result IN ('win','draw','loss')),
    turns INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    replay_path TEXT NOT NULL DEFAULT '',
    UNIQUE(match_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_matches_batch_id ON matches(batch_id);
CREATE INDEX IF NOT EXISTS idx_matches_agent_a_id ON matches(agent_a_id);
CREATE INDEX IF NOT EXISTS idx_matches_agent_b_id ON matches(agent_b_id);
CREATE INDEX IF NOT EXISTS idx_games_match_id ON games(match_id);
`

const testSQLitePragmas = "?_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)"

func TestOpenIsIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "ladder.db")
	first, err := db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()

	second, err := db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	var count int
	if err := second.QueryRowContext(context.Background(), "SELECT count(*) FROM schema_migrations").Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("migration count = %d, want 2", count)
	}
	var foreignKeys, busyTimeout int
	var journalMode string
	if err := second.QueryRowContext(context.Background(), "PRAGMA foreign_keys").Scan(&foreignKeys); err != nil {
		t.Fatal(err)
	}
	if err := second.QueryRowContext(context.Background(), "PRAGMA busy_timeout").Scan(&busyTimeout); err != nil {
		t.Fatal(err)
	}
	if err := second.QueryRowContext(context.Background(), "PRAGMA journal_mode").Scan(&journalMode); err != nil {
		t.Fatal(err)
	}
	if foreignKeys != 1 || busyTimeout != 5000 || journalMode != "wal" {
		t.Fatalf("pragmas = foreign_keys:%d busy_timeout:%d journal_mode:%q", foreignKeys, busyTimeout, journalMode)
	}
}

func TestExistingDatabaseGetsPartialStatusMigration(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "legacy.db")
	legacy, err := sql.Open("sqlite", path+testSQLitePragmas)
	if err != nil {
		t.Fatal(err)
	}
	legacy.SetMaxOpenConns(1)
	if _, err := legacy.ExecContext(ctx, legacyInitSQL); err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.ExecContext(ctx, `CREATE TABLE schema_migrations (
		name TEXT PRIMARY KEY,
		applied_at TEXT NOT NULL
	)`); err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.ExecContext(ctx, `INSERT INTO schema_migrations(name, applied_at) VALUES ('0001_init.sql', '2026-09-22T00:00:00Z')`); err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.ExecContext(ctx, `
		INSERT INTO agents(id, name, kind, created_at) VALUES
			(1, 'alpha', 'builtin', '2026-09-22T00:00:00Z'),
			(2, 'beta', 'builtin', '2026-09-22T00:00:00Z')`); err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.ExecContext(ctx, `
		INSERT INTO batches(id, status, games, created_at) VALUES
			(1, 'pending', 1, '2026-09-22T00:00:00Z')`); err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.ExecContext(ctx, `
		INSERT INTO matches(id, batch_id, agent_a_id, agent_b_id, games, seed, status, created_at) VALUES
			(1, 1, 1, 2, 1, 123, 'pending', '2026-09-22T00:00:00Z')`); err != nil {
		t.Fatal(err)
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}

	migrated, err := db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer migrated.Close()
	if err := store.New(migrated).FinishBatch(ctx, 1, "partial", "one match failed"); err != nil {
		t.Fatalf("finish partial batch: %v", err)
	}
	var status string
	if err := migrated.QueryRowContext(ctx, `SELECT status FROM batches WHERE id = 1`).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "partial" {
		t.Fatalf("batch status = %q, want partial", status)
	}
	var matchCount int
	if err := migrated.QueryRowContext(ctx, `SELECT COUNT(*) FROM matches WHERE batch_id = 1`).Scan(&matchCount); err != nil {
		t.Fatal(err)
	}
	if matchCount != 1 {
		t.Fatalf("match count = %d, want 1", matchCount)
	}
	var foreignKeys int
	if err := migrated.QueryRowContext(ctx, "PRAGMA foreign_keys").Scan(&foreignKeys); err != nil {
		t.Fatal(err)
	}
	if foreignKeys != 1 {
		t.Fatalf("foreign_keys = %d, want 1", foreignKeys)
	}
	rows, err := migrated.QueryContext(ctx, "PRAGMA foreign_key_check")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if rows.Next() {
		var table, rowID, parent, foreignKey any
		if err := rows.Scan(&table, &rowID, &parent, &foreignKey); err != nil {
			t.Fatal(err)
		}
		t.Fatalf("foreign key violation: table=%v row=%v parent=%v fk=%v", table, rowID, parent, foreignKey)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
}
