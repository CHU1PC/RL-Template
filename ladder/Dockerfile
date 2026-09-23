# Reproducibility comes from lockfiles (Cargo.lock, go.sum, bun.lock), not image tags.
FROM oven/bun:1 AS web

WORKDIR /src/ladder/web
COPY ladder/web/package.json ladder/web/bun.lock ./
RUN bun install --frozen-lockfile
COPY ladder/web/ ./
RUN bun run build

FROM golang:1 AS server

WORKDIR /src/ladder/server
COPY ladder/server/go.mod ladder/server/go.sum ./
RUN go mod download
COPY ladder/server/ ./
RUN CGO_ENABLED=0 go build -o /out/server ./cmd/server

FROM rust:1 AS arena

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
RUN cargo build --release --locked -p rl-arena

FROM debian:bookworm-slim AS runtime

WORKDIR /app
COPY --from=server /out/server /app/server
COPY --from=arena /src/target/release/rl-arena /app/arena
COPY --from=web /src/ladder/web/dist/ /app/web/

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --no-create-home app \
    && mkdir -p /app/data \
    && chown -R app:app /app

EXPOSE 8080

# server は web の dist、永続データ、arena バイナリを環境変数で参照する。
ENV LADDER_WEB_DIR=/app/web
ENV LADDER_DATA_DIR=/app/data
ENV LADDER_ARENA_BIN=/app/arena
USER app
ENTRYPOINT ["/app/server"]
