package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"ladder/server/internal/api"
	"ladder/server/internal/db"
	"ladder/server/internal/queue"
	"ladder/server/internal/rating"
	"ladder/server/internal/runner"
	"ladder/server/internal/store"
)

type config struct {
	Addr         string
	DataDir      string
	DBPath       string
	ArenaBin     string
	Workers      int
	WebDir       string
	RatingMu0    float64
	RatingSigma0 float64
}

func configFromEnv(getenv func(string) string) config {
	dataDir := getenv("LADDER_DATA_DIR")
	if dataDir == "" {
		dataDir = "./data"
	}
	dbPath := getenv("LADDER_DB_PATH")
	if dbPath == "" {
		dbPath = filepath.Join(dataDir, "ladder.db")
	}
	addr := getenv("LADDER_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	arenaBin := getenv("LADDER_ARENA_BIN")
	if arenaBin == "" {
		arenaBin = "arena"
	}
	workers := 2
	if parsed, err := strconv.Atoi(getenv("LADDER_WORKERS")); err == nil && parsed >= 1 {
		workers = parsed
	}
	ratingMu0 := 600.0
	if parsed, err := strconv.ParseFloat(getenv("LADDER_RATING_MU0"), 64); err == nil {
		ratingMu0 = parsed
	}
	ratingSigma0 := 200.0
	if parsed, err := strconv.ParseFloat(getenv("LADDER_RATING_SIGMA0"), 64); err == nil {
		ratingSigma0 = parsed
	}
	return config{Addr: addr, DataDir: dataDir, DBPath: dbPath, ArenaBin: arenaBin, Workers: workers,
		WebDir: getenv("LADDER_WEB_DIR"), RatingMu0: ratingMu0, RatingSigma0: ratingSigma0}
}

func main() {
	openAPIPath := flag.String("openapi", "", "write the OpenAPI specification to a file")
	flag.Parse()

	router := chi.NewRouter()
	router.Use(api.UploadLimitMiddleware)
	apiConfig := huma.DefaultConfig("ladder", "0.0.0")
	apiConfig.Servers = []*huma.Server{{URL: "/api"}}
	var humaAPI huma.API
	router.Route("/api", func(apiRouter chi.Router) {
		humaAPI = humachi.New(apiRouter, apiConfig)
	})

	if *openAPIPath != "" {
		api.New(humaAPI, nil, nil, "")
		specification, err := humaAPI.OpenAPI().YAML()
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to generate OpenAPI specification: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(*openAPIPath, specification, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write OpenAPI specification to %q: %v\n", *openAPIPath, err)
			os.Exit(1)
		}
		return
	}

	cfg := configFromEnv(os.Getenv)
	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "failed to create data directory %q: %v\n", cfg.DataDir, err)
		os.Exit(1)
	}
	database, err := db.Open(cfg.DBPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open database: %v\n", err)
		os.Exit(1)
	}
	defer database.Close()

	ladderStore := store.NewWithRatingDefaults(database, cfg.RatingMu0, cfg.RatingSigma0)
	requeued, err := ladderStore.RequeueRunningBatches(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to requeue running batches: %v\n", err)
		os.Exit(1)
	}
	slog.Info("requeued running batches", "count", requeued)
	ladderRunner := runner.New(cfg.ArenaBin)
	rater := rating.NoopRater{} // Swap this one line when the rating policy is decided.
	ladderQueue := queue.New(ladderStore, ladderRunner, cfg.Workers, rater)
	api.New(humaAPI, ladderStore, ladderQueue, cfg.DataDir)
	if cfg.WebDir != "" {
		router.NotFound(staticHandler(cfg.WebDir).ServeHTTP)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	ladderQueue.Start(ctx)
	server := &http.Server{Addr: cfg.Addr, Handler: router}
	serverErrors := make(chan error, 1)
	go func() {
		slog.Info("ladder server started", "addr", cfg.Addr, "data_dir", cfg.DataDir, "db_path", cfg.DBPath, "arena_bin", cfg.ArenaBin, "workers", cfg.Workers, "web_dir", cfg.WebDir)
		serverErrors <- server.ListenAndServe()
	}()

	select {
	case err := <-serverErrors:
		if !errors.Is(err, http.ErrServerClosed) {
			slog.Error("ladder server stopped", "error", err)
		}
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_ = server.Shutdown(shutdownCtx)
		cancel()
	}
	ladderQueue.Stop()
}

func staticHandler(webDir string) http.Handler {
	root, err := filepath.Abs(webDir)
	if err != nil {
		root = webDir
	}
	fileServer := http.FileServer(http.Dir(root))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api" || strings.HasPrefix(r.URL.Path, "/api/") {
			http.NotFound(w, r)
			return
		}
		cleanPath := filepath.Clean(r.URL.Path)
		if cleanPath == "." || cleanPath == string(filepath.Separator) {
			cleanPath = "/"
		}
		candidate := filepath.Join(root, filepath.FromSlash(strings.TrimPrefix(cleanPath, "/")))
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			fileServer.ServeHTTP(w, r)
			return
		}
		indexPath := filepath.Join(root, "index.html")
		info, err := os.Stat(indexPath)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		indexFile, err := os.Open(indexPath)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		defer indexFile.Close()
		http.ServeContent(w, r, "index.html", info.ModTime(), indexFile)
	})
}
