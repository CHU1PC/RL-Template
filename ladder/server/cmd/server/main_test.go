package main

import (
	"testing"
)

func TestConfigFromEnvDefaults(t *testing.T) {
	cfg := configFromEnv(func(string) string { return "" })
	if cfg.Addr != ":8080" || cfg.DataDir != "./data" || cfg.DBPath != "data/ladder.db" || cfg.ArenaBin != "arena" || cfg.Workers != 2 || cfg.WebDir != "" || cfg.RatingMu0 != 600 || cfg.RatingSigma0 != 200 {
		t.Fatalf("defaults = %#v", cfg)
	}
}

func TestConfigFromEnvRatingValues(t *testing.T) {
	values := map[string]string{
		"LADDER_RATING_MU0":    "712.5",
		"LADDER_RATING_SIGMA0": "44.25",
	}
	cfg := configFromEnv(func(key string) string { return values[key] })
	if cfg.RatingMu0 != 712.5 || cfg.RatingSigma0 != 44.25 {
		t.Fatalf("rating config = %#v", cfg)
	}

	values["LADDER_RATING_MU0"] = "invalid"
	values["LADDER_RATING_SIGMA0"] = ""
	cfg = configFromEnv(func(key string) string { return values[key] })
	if cfg.RatingMu0 != 600 || cfg.RatingSigma0 != 200 {
		t.Fatalf("invalid rating config = %#v", cfg)
	}
}

func TestConfigFromEnvInvalidWorkers(t *testing.T) {
	values := map[string]string{"LADDER_DATA_DIR": "/tmp/ladder", "LADDER_WORKERS": "0"}
	cfg := configFromEnv(func(key string) string { return values[key] })
	if cfg.Workers != 2 || cfg.DBPath != "/tmp/ladder/ladder.db" {
		t.Fatalf("config = %#v", cfg)
	}
}
