package db

import (
	"context"
	"database/sql"
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	_ "modernc.org/sqlite"
)

//go:embed migrations/*.sql
var migrationFiles embed.FS

const sqlitePragmas = "?_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)"

// Open opens the SQLite database at path, applies all embedded migrations and returns it.
func Open(path string) (*sql.DB, error) {
	if path != ":memory:" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, fmt.Errorf("create database directory: %w", err)
		}
	}

	dsn := path + sqlitePragmas
	if path == ":memory:" {
		dsn = "file::memory:?mode=memory&cache=shared&_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)"
	}
	database, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite database: %w", err)
	}
	// SQLite permits one writer at a time. A single pooled connection keeps
	// migrations and the small application database deterministic.
	// The PRAGMA foreign_keys OFF/ON toggle in applyMigrations is safe only
	// with this single-connection pool; a larger pool requires moving it onto
	// a dedicated *sql.Conn.
	database.SetMaxOpenConns(1)
	database.SetMaxIdleConns(1)
	if err := database.PingContext(context.Background()); err != nil {
		database.Close()
		return nil, fmt.Errorf("ping sqlite database: %w", err)
	}
	if err := applyMigrations(context.Background(), database); err != nil {
		database.Close()
		return nil, err
	}
	return database, nil
}

func applyMigrations(ctx context.Context, database *sql.DB) error {
	entries, err := migrationFiles.ReadDir("migrations")
	if err != nil {
		return fmt.Errorf("read migrations: %w", err)
	}
	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			files = append(files, entry.Name())
		}
	}
	sort.Strings(files)
	if _, err := database.ExecContext(ctx, "PRAGMA foreign_keys = OFF"); err != nil {
		return fmt.Errorf("disable foreign keys for migrations: %w", err)
	}
	foreignKeysRestored := false
	restoreForeignKeys := func() error {
		if foreignKeysRestored {
			return nil
		}
		foreignKeysRestored = true
		if _, err := database.ExecContext(ctx, "PRAGMA foreign_keys = ON"); err != nil {
			return fmt.Errorf("restore foreign keys after migrations: %w", err)
		}
		return nil
	}

	tx, err := database.BeginTx(ctx, nil)
	if err != nil {
		_ = restoreForeignKeys()
		return fmt.Errorf("begin migration transaction: %w", err)
	}
	rollback := func(cause error) error {
		_ = tx.Rollback()
		if err := restoreForeignKeys(); err != nil {
			return fmt.Errorf("%w; %v", cause, err)
		}
		return cause
	}
	if _, err := tx.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS schema_migrations (
		name TEXT PRIMARY KEY,
		applied_at TEXT NOT NULL
	)`); err != nil {
		return rollback(fmt.Errorf("create schema_migrations: %w", err))
	}

	applied := make(map[string]struct{}, len(files))
	rows, err := tx.QueryContext(ctx, "SELECT name FROM schema_migrations")
	if err != nil {
		return rollback(fmt.Errorf("read schema_migrations: %w", err))
	}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			rows.Close()
			return rollback(fmt.Errorf("scan schema_migrations: %w", err))
		}
		applied[name] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return rollback(fmt.Errorf("iterate schema_migrations: %w", err))
	}
	if err := rows.Close(); err != nil {
		return rollback(fmt.Errorf("close schema_migrations: %w", err))
	}

	for _, name := range files {
		if _, ok := applied[name]; ok {
			continue
		}
		contents, err := migrationFiles.ReadFile("migrations/" + name)
		if err != nil {
			return rollback(fmt.Errorf("read migration %s: %w", name, err))
		}
		if _, err := tx.ExecContext(ctx, string(contents)); err != nil {
			return rollback(fmt.Errorf("apply migration %s: %w", name, err))
		}
		if _, err := tx.ExecContext(ctx, "INSERT INTO schema_migrations(name, applied_at) VALUES (?, ?)", name, time.Now().UTC().Format(time.RFC3339Nano)); err != nil {
			return rollback(fmt.Errorf("record migration %s: %w", name, err))
		}
	}
	if err := tx.Commit(); err != nil {
		_ = restoreForeignKeys()
		return fmt.Errorf("commit migrations: %w", err)
	}
	if err := restoreForeignKeys(); err != nil {
		return err
	}
	return nil
}
