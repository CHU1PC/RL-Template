CREATE TABLE batches_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL CHECK(status IN ('pending','running','done','partial','failed')),
    games INTEGER NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT ''
);

INSERT INTO batches_new (id, status, games, error, created_at, started_at, finished_at)
SELECT id, status, games, error, created_at, started_at, finished_at
FROM batches;

DROP TABLE batches;
ALTER TABLE batches_new RENAME TO batches;
