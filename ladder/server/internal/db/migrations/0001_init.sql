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
