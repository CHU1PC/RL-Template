package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"ladder/server/internal/db"
	"ladder/server/internal/queue"
	"ladder/server/internal/rating"
	"ladder/server/internal/runner"
	"ladder/server/internal/store"
)

const fakeArenaOutput = `{"games":[{"index":0,"seed":1,"players":["a","b"],"result":{"a":1.0,"b":0.0},"turns":5,"duration_ms":3},{"index":1,"seed":2,"players":["a","b"],"result":{"a":0.5,"b":0.5},"turns":7,"duration_ms":4}],"summary":{"a":{"win":1,"draw":1,"loss":0},"b":{"win":0,"draw":1,"loss":1}}}`

func TestHTTPBatchFlowSeed(t *testing.T) {
	server, cancel := testHTTPServer(t, fakeArenaOutput, true)
	defer cancel()

	firstID := postBuiltin(t, server, "stat_v1")
	secondID := postBuiltin(t, server, "stat_v2")
	if firstID == secondID {
		t.Fatalf("agent IDs are equal: %d", firstID)
	}
	batchResponse := postJSON(t, server, "/api/batches/auto", `{"games":2}`)
	var created struct {
		ID int64 `json:"id"`
	}
	decodeJSON(t, batchResponse, &created)

	deadline := time.Now().Add(3 * time.Second)
	var detail struct {
		Status      string `json:"status"`
		FailedCount int    `json:"failed_count"`
		DoneCount   int    `json:"done_count"`
		Matches     []struct {
			ID   int64  `json:"id"`
			Seed string `json:"seed"`
		} `json:"matches"`
	}
	for time.Now().Before(deadline) {
		response := get(t, server, "/api/batches/"+itoa(created.ID))
		decodeJSON(t, response, &detail)
		if detail.Status == "done" {
			break
		}
		if detail.Status == "failed" {
			t.Fatalf("batch failed")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if detail.Status != "done" || detail.FailedCount != 0 || detail.DoneCount != 1 {
		t.Fatalf("batch = %#v, want done with failed_count=0 and done_count=1", detail)
	}
	if len(detail.Matches) != 1 || detail.Matches[0].Seed == "" {
		t.Fatalf("match seeds = %#v, want one non-empty JSON string", detail.Matches)
	}
	matchResponse := get(t, server, "/api/matches/"+itoa(detail.Matches[0].ID))
	var matchDetail struct {
		Match struct {
			Seed string `json:"seed"`
		} `json:"match"`
		Games []struct {
			Seed string `json:"seed"`
		} `json:"games"`
	}
	decodeJSON(t, matchResponse, &matchDetail)
	if matchDetail.Match.Seed == "" || len(matchDetail.Games) != 2 || matchDetail.Games[0].Seed == "" || matchDetail.Games[1].Seed == "" {
		t.Fatalf("match detail seeds = %#v, want JSON strings", matchDetail)
	}

	listResponse := get(t, server, "/api/batches")
	var listed struct {
		Batches []struct {
			FailedCount int `json:"failed_count"`
			DoneCount   int `json:"done_count"`
		} `json:"batches"`
	}
	decodeJSON(t, listResponse, &listed)
	if len(listed.Batches) != 1 || listed.Batches[0].FailedCount != 0 || listed.Batches[0].DoneCount != 1 {
		t.Fatalf("listed batches = %#v, want failed_count=0 and done_count=1", listed)
	}

	rankingResponse := get(t, server, "/api/ranking")
	var ranking struct {
		Ranking []struct {
			Agent string `json:"agent"`
			Games int    `json:"games"`
		} `json:"ranking"`
	}
	decodeJSON(t, rankingResponse, &ranking)
	if len(ranking.Ranking) != 2 {
		t.Fatalf("ranking length = %d, want 2", len(ranking.Ranking))
	}
	for _, row := range ranking.Ranking {
		if row.Games != 2 {
			t.Fatalf("%s games = %d, want 2", row.Agent, row.Games)
		}
	}
}

func TestMissingArenaFailsBatchAndKeepsServing(t *testing.T) {
	server, cancel := testHTTPServer(t, "", false)
	defer cancel()
	postBuiltin(t, server, "stat_v1")
	postBuiltin(t, server, "stat_v2")
	batchResponse := postJSON(t, server, "/api/batches/auto", `{"games":1}`)
	var created struct {
		ID int64 `json:"id"`
	}
	decodeJSON(t, batchResponse, &created)

	deadline := time.Now().Add(3 * time.Second)
	var detail struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	for time.Now().Before(deadline) {
		response := get(t, server, "/api/batches/"+itoa(created.ID))
		decodeJSON(t, response, &detail)
		if detail.Status == "failed" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if detail.Status != "failed" || detail.Error == "" {
		t.Fatalf("batch = %#v, want failed with an error", detail)
	}
	if response := get(t, server, "/api/agents"); response.StatusCode != http.StatusOK {
		t.Fatalf("agents after failed batch status = %d", response.StatusCode)
	}
}

func TestGroupAgentPathParameter(t *testing.T) {
	server, cancel := testHTTPServer(t, "", false)
	defer cancel()

	agentID := postBuiltin(t, server, "stat_v1")
	groupResponse := postJSON(t, server, "/api/groups", `{"name":"alpha"}`)
	var group struct {
		ID int64 `json:"id"`
	}
	decodeJSON(t, groupResponse, &group)

	assignedResponse := postJSON(t, server, "/api/groups/"+itoa(group.ID)+"/agents", `{"agent_id":`+itoa(agentID)+`}`)
	var assigned struct {
		Agents []struct {
			ID int64 `json:"id"`
		} `json:"agents"`
	}
	decodeJSON(t, assignedResponse, &assigned)
	if len(assigned.Agents) != 1 || assigned.Agents[0].ID != agentID {
		t.Fatalf("assigned agents = %#v, want agent %d", assigned.Agents, agentID)
	}

	request, err := http.NewRequest(http.MethodPost, server.URL+"/api/groups/"+itoa(group.ID+1)+"/agents", bytes.NewBufferString(`{"agent_id":`+itoa(agentID)+`}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	missingResponse, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer missingResponse.Body.Close()
	if missingResponse.StatusCode != http.StatusNotFound {
		data, _ := io.ReadAll(missingResponse.Body)
		t.Fatalf("missing group status = %d, body = %s", missingResponse.StatusCode, data)
	}
}

func testHTTPServer(t *testing.T, output string, arenaExists bool) (*httptest.Server, func()) {
	t.Helper()
	database, err := db.Open(filepath.Join(t.TempDir(), "ladder.db"))
	if err != nil {
		t.Fatal(err)
	}
	ladderStore := store.New(database)
	arenaBin := filepath.Join(t.TempDir(), "arena")
	if arenaExists {
		script := "#!/bin/sh\nprintf '%s\\n' '" + output + "'\n"
		if err := os.WriteFile(arenaBin, []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	ladderQueue := queue.New(ladderStore, runner.New(arenaBin), 1, rating.NoopRater{})
	ctx, cancelContext := context.WithCancel(context.Background())
	ladderQueue.Start(ctx)

	router := chi.NewRouter()
	router.Use(UploadLimitMiddleware)
	router.Route("/api", func(apiRouter chi.Router) {
		config := huma.DefaultConfig("ladder", "test")
		api := humachi.New(apiRouter, config)
		New(api, ladderStore, ladderQueue, t.TempDir())
	})
	server := httptest.NewServer(router)
	cancel := func() {
		server.Close()
		cancelContext()
		ladderQueue.Stop()
		database.Close()
	}
	t.Cleanup(cancel)
	return server, cancel
}

func postBuiltin(t *testing.T, server *httptest.Server, name string) int64 {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for key, value := range map[string]string{"name": name, "kind": "builtin", "builtin_name": "random"} {
		if err := writer.WriteField(key, value); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, server.URL+"/api/agents", &body)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		data, _ := io.ReadAll(response.Body)
		t.Fatalf("create agent status = %d, body = %s", response.StatusCode, data)
	}
	var result struct {
		ID int64 `json:"id"`
	}
	decodeJSON(t, response, &result)
	return result.ID
}

func postJSON(t *testing.T, server *httptest.Server, path, body string) *http.Response {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, server.URL+path, bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		data, _ := io.ReadAll(response.Body)
		response.Body.Close()
		t.Fatalf("POST %s status = %d, body = %s", path, response.StatusCode, data)
	}
	return response
}

func get(t *testing.T, server *httptest.Server, path string) *http.Response {
	t.Helper()
	response, err := http.Get(server.URL + path)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(response.Body)
		response.Body.Close()
		t.Fatalf("GET %s status = %d, body = %s", path, response.StatusCode, data)
	}
	return response
}

func decodeJSON(t *testing.T, response *http.Response, value any) {
	t.Helper()
	defer response.Body.Close()
	if err := json.NewDecoder(response.Body).Decode(value); err != nil {
		t.Fatal(err)
	}
}

func itoa(value int64) string {
	return strconv.FormatInt(value, 10)
}
