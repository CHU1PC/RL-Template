package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"ladder/server/internal/pairing"
	"ladder/server/internal/queue"
	"ladder/server/internal/store"
)

const maxUploadBytes int64 = 512 * 1024 * 1024

type Server struct {
	store   *store.Store
	queue   *queue.Queue
	dataDir string
}

// New registers the ladder API on an already configured Huma API.
func New(humaAPI huma.API, s *store.Store, q *queue.Queue, dataDir string) *Server {
	server := &Server{store: s, queue: q, dataDir: dataDir}
	server.register(humaAPI)
	return server
}

// UploadLimitMiddleware caps the request body before Huma parses multipart data.
func UploadLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/agents") && strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
			r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes+1024*1024)
		}
		next.ServeHTTP(w, r)
	})
}

type AgentBody struct {
	ID          int64   `json:"id"`
	Name        string  `json:"name"`
	Kind        string  `json:"kind"`
	FilePath    string  `json:"file_path"`
	BuiltinName string  `json:"builtin_name"`
	RatingMu    float64 `json:"rating_mu"`
	RatingSigma float64 `json:"rating_sigma"`
	CreatedAt   string  `json:"created_at"`
}

type AgentResponse struct{ Body AgentBody }
type AgentListBody struct {
	Agents []AgentBody `json:"agents"`
}
type AgentListResponse struct{ Body AgentListBody }

type GroupBody struct {
	ID        int64  `json:"id"`
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
}

type GroupResponse struct{ Body GroupBody }
type GroupListBody struct {
	Groups []GroupBody `json:"groups"`
}
type GroupListResponse struct{ Body GroupListBody }
type GroupAgentsBody struct {
	Group  GroupBody   `json:"group"`
	Agents []AgentBody `json:"agents"`
}
type GroupAgentsResponse struct{ Body GroupAgentsBody }

type MatchBody struct {
	ID         int64  `json:"id"`
	BatchID    int64  `json:"batch_id"`
	AgentAID   int64  `json:"agent_a_id"`
	AgentBID   int64  `json:"agent_b_id"`
	AgentAName string `json:"agent_a_name"`
	AgentBName string `json:"agent_b_name"`
	Games      int    `json:"games"`
	Seed       int64  `json:"seed,string"`
	Status     string `json:"status"`
	Error      string `json:"error"`
	WinA       int    `json:"win_a"`
	Draw       int    `json:"draw"`
	LossA      int    `json:"loss_a"`
	DurationMs int    `json:"duration_ms"`
	CreatedAt  string `json:"created_at"`
}

type GameBody struct {
	ID         int64   `json:"id"`
	MatchID    int64   `json:"match_id"`
	Index      int     `json:"index"`
	Seed       int64   `json:"seed,string"`
	ScoreA     float64 `json:"score_a"`
	ScoreB     float64 `json:"score_b"`
	Result     string  `json:"result"`
	Turns      int     `json:"turns"`
	DurationMs int     `json:"duration_ms"`
	ReplayPath string  `json:"replay_path"`
}

func (MatchBody) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	if seed := schema.Properties["seed"]; seed != nil {
		seed.Type = huma.TypeString
		seed.Format = "int64"
	}
	return schema
}

func (GameBody) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	if seed := schema.Properties["seed"]; seed != nil {
		seed.Type = huma.TypeString
		seed.Format = "int64"
	}
	return schema
}

type BatchBody struct {
	ID          int64       `json:"id"`
	Status      string      `json:"status"`
	Games       int         `json:"games"`
	Error       string      `json:"error"`
	CreatedAt   string      `json:"created_at"`
	StartedAt   string      `json:"started_at"`
	FinishedAt  string      `json:"finished_at"`
	FailedCount int         `json:"failed_count"`
	DoneCount   int         `json:"done_count"`
	Matches     []MatchBody `json:"matches,omitempty"`
}

type BatchResponse struct{ Body BatchBody }
type BatchListBody struct {
	Batches []BatchBody `json:"batches"`
}
type BatchListResponse struct{ Body BatchListBody }
type MatchListBody struct {
	Matches []MatchBody `json:"matches"`
}
type MatchListResponse struct{ Body MatchListBody }
type MatchDetailBody struct {
	Match MatchBody  `json:"match"`
	Games []GameBody `json:"games"`
}
type MatchDetailResponse struct{ Body MatchDetailBody }

type RankingRowBody struct {
	Agent   string  `json:"agent"`
	AgentID int64   `json:"agent_id"`
	Mu      float64 `json:"mu"`
	Sigma   float64 `json:"sigma"`
	Games   int     `json:"games"`
	Win     int     `json:"win"`
	Draw    int     `json:"draw"`
	Loss    int     `json:"loss"`
	WinRate float64 `json:"win_rate"`
}
type RankingListBody struct {
	Ranking []RankingRowBody `json:"ranking"`
}
type RankingResponse struct{ Body RankingListBody }

type CreateAgentForm struct {
	Name        string        `form:"name" required:"true"`
	Kind        string        `form:"kind" required:"true" enum:"onnx,builtin"`
	BuiltinName string        `form:"builtin_name" required:"false"`
	File        huma.FormFile `form:"file" contentType:"application/octet-stream" required:"false"`
}
type CreateAgentInput struct {
	RawBody huma.MultipartFormFiles[CreateAgentForm]
}

type CreateGroupBody struct {
	Name string `json:"name" required:"true"`
}
type CreateGroupInput struct{ Body CreateGroupBody }
type GroupAgentBody struct {
	AgentID int64 `json:"agent_id" required:"true"`
}
type GroupAgentInput struct {
	ID   int64 `path:"id"`
	Body GroupAgentBody
}
type CreateBatchBody struct {
	AgentIDs []int64 `json:"agent_ids" required:"true" minItems:"2"`
	Games    int     `json:"games" required:"true" minimum:"1"`
}
type CreateBatchInput struct{ Body CreateBatchBody }
type CreateAutoBatchBody struct {
	Games int `json:"games" required:"true" minimum:"1"`
}
type CreateAutoBatchInput struct{ Body CreateAutoBatchBody }
type IDInput struct {
	ID int64 `path:"id"`
}
type ListBatchesInput struct {
	Limit int `query:"limit" default:"100" minimum:"1"`
}
type ListMatchesInput struct {
	AgentID int64 `query:"agent_id"`
	Limit   int   `query:"limit" default:"100" minimum:"1"`
}

func (s *Server) register(api huma.API) {
	huma.Register(api, huma.Operation{OperationID: "create-agent", Method: http.MethodPost, Path: "/agents", DefaultStatus: http.StatusCreated, Errors: []int{http.StatusConflict, http.StatusUnprocessableEntity}}, s.createAgent)
	huma.Register(api, huma.Operation{OperationID: "list-agents", Method: http.MethodGet, Path: "/agents"}, s.listAgents)
	huma.Register(api, huma.Operation{OperationID: "get-agent", Method: http.MethodGet, Path: "/agents/{id}", Errors: []int{http.StatusNotFound}}, s.getAgent)
	huma.Register(api, huma.Operation{OperationID: "create-group", Method: http.MethodPost, Path: "/groups", DefaultStatus: http.StatusCreated, Errors: []int{http.StatusConflict, http.StatusUnprocessableEntity}}, s.createGroup)
	huma.Register(api, huma.Operation{OperationID: "list-groups", Method: http.MethodGet, Path: "/groups"}, s.listGroups)
	huma.Register(api, huma.Operation{OperationID: "add-group-agent", Method: http.MethodPost, Path: "/groups/{id}/agents", Errors: []int{http.StatusNotFound, http.StatusUnprocessableEntity}}, s.addGroupAgent)
	huma.Register(api, huma.Operation{OperationID: "create-batch", Method: http.MethodPost, Path: "/batches", DefaultStatus: http.StatusCreated, Errors: []int{http.StatusNotFound, http.StatusUnprocessableEntity}}, s.createBatch)
	huma.Register(api, huma.Operation{OperationID: "create-auto-batch", Method: http.MethodPost, Path: "/batches/auto", DefaultStatus: http.StatusCreated, Errors: []int{http.StatusUnprocessableEntity}}, s.createAutoBatch)
	huma.Register(api, huma.Operation{OperationID: "list-batches", Method: http.MethodGet, Path: "/batches"}, s.listBatches)
	huma.Register(api, huma.Operation{OperationID: "get-batch", Method: http.MethodGet, Path: "/batches/{id}", Errors: []int{http.StatusNotFound}}, s.getBatch)
	huma.Register(api, huma.Operation{OperationID: "list-matches", Method: http.MethodGet, Path: "/matches"}, s.listMatches)
	huma.Register(api, huma.Operation{OperationID: "get-match", Method: http.MethodGet, Path: "/matches/{id}", Errors: []int{http.StatusNotFound}}, s.getMatch)
	huma.Register(api, huma.Operation{OperationID: "get-ranking", Method: http.MethodGet, Path: "/ranking"}, s.ranking)
}

func (s *Server) createAgent(ctx context.Context, input *CreateAgentInput) (*AgentResponse, error) {
	if s.store == nil {
		return nil, fmt.Errorf("store is not configured")
	}
	form := input.RawBody.Data()
	if form == nil {
		return nil, huma.Error422UnprocessableEntity("agent form is required")
	}
	if form.File.IsSet {
		defer form.File.Close()
	}
	if form.Kind != "onnx" && form.Kind != "builtin" {
		return nil, huma.Error422UnprocessableEntity("kind must be onnx or builtin")
	}
	if strings.TrimSpace(form.Name) == "" {
		return nil, huma.Error422UnprocessableEntity("name is required")
	}
	if form.Kind == "builtin" {
		if strings.TrimSpace(form.BuiltinName) == "" {
			return nil, huma.Error422UnprocessableEntity("builtin_name is required for builtin agents")
		}
		agent, err := s.store.CreateAgent(ctx, store.Agent{Name: form.Name, Kind: form.Kind, BuiltinName: form.BuiltinName})
		if err != nil {
			return nil, duplicateError(err, "agent name already exists")
		}
		return &AgentResponse{Body: agentView(agent)}, nil
	}
	if !form.File.IsSet || form.File.Filename == "" {
		return nil, huma.Error422UnprocessableEntity("file is required for onnx agents")
	}
	if form.File.Size > maxUploadBytes {
		_ = form.File.Close()
		return nil, huma.Error413RequestEntityTooLarge("upload exceeds 512 MiB")
	}
	agent, err := s.store.CreateAgent(ctx, store.Agent{Name: form.Name, Kind: form.Kind})
	if err != nil {
		return nil, duplicateError(err, "agent name already exists")
	}
	path, err := s.saveUpload(agent.ID, form.File)
	if err != nil {
		return nil, huma.Error422UnprocessableEntity(err.Error())
	}
	if err := s.store.SetAgentFilePath(ctx, agent.ID, path); err != nil {
		_ = os.Remove(path)
		return nil, fmt.Errorf("save agent file path: %w", err)
	}
	agent.FilePath = path
	return &AgentResponse{Body: agentView(agent)}, nil
}

func (s *Server) saveUpload(agentID int64, file huma.FormFile) (string, error) {
	name := file.Filename
	if name == "." || name == ".." || name == "" || filepath.Base(name) != name || strings.ContainsAny(name, `/\\`) {
		_ = file.Close()
		return "", fmt.Errorf("file name must be a base name")
	}
	dataDir, err := filepath.Abs(s.dataDir)
	if err != nil {
		_ = file.Close()
		return "", fmt.Errorf("resolve data directory: %w", err)
	}
	agentDir := filepath.Join(dataDir, "agents", strconv.FormatInt(agentID, 10))
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		_ = file.Close()
		return "", fmt.Errorf("create agent upload directory: %w", err)
	}
	target := filepath.Join(agentDir, filepath.Base(name))
	if filepath.Dir(target) != agentDir {
		_ = file.Close()
		return "", fmt.Errorf("file name escapes agent directory")
	}
	output, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		_ = file.Close()
		return "", fmt.Errorf("create upload: %w", err)
	}
	_, copyErr := io.Copy(output, io.LimitReader(file, maxUploadBytes+1))
	closeErr := output.Close()
	fileCloseErr := file.Close()
	if copyErr != nil {
		_ = os.Remove(target)
		return "", fmt.Errorf("write upload: %w", copyErr)
	}
	if closeErr != nil || fileCloseErr != nil {
		_ = os.Remove(target)
		return "", fmt.Errorf("close upload: %w", firstError(closeErr, fileCloseErr))
	}
	if info, err := os.Stat(target); err != nil {
		_ = os.Remove(target)
		return "", fmt.Errorf("stat upload: %w", err)
	} else if info.Size() > maxUploadBytes {
		_ = os.Remove(target)
		return "", fmt.Errorf("upload exceeds 512 MiB")
	}
	abs, err := filepath.Abs(target)
	if err != nil {
		_ = os.Remove(target)
		return "", fmt.Errorf("resolve upload path: %w", err)
	}
	return abs, nil
}

func firstError(first, second error) error {
	if first != nil {
		return first
	}
	return second
}

func (s *Server) listAgents(ctx context.Context, _ *struct{}) (*AgentListResponse, error) {
	agents, err := s.store.ListAgents(ctx)
	if err != nil {
		return nil, err
	}
	result := make([]AgentBody, 0, len(agents))
	for _, agent := range agents {
		result = append(result, agentView(agent))
	}
	return &AgentListResponse{Body: AgentListBody{Agents: result}}, nil
}

func (s *Server) getAgent(ctx context.Context, input *IDInput) (*AgentResponse, error) {
	agent, err := s.store.GetAgent(ctx, input.ID)
	if err != nil {
		return nil, notFoundError(err)
	}
	return &AgentResponse{Body: agentView(agent)}, nil
}

func (s *Server) createGroup(ctx context.Context, input *CreateGroupInput) (*GroupResponse, error) {
	if strings.TrimSpace(input.Body.Name) == "" {
		return nil, huma.Error422UnprocessableEntity("name is required")
	}
	group, err := s.store.CreateGroup(ctx, input.Body.Name)
	if err != nil {
		return nil, duplicateError(err, "group name already exists")
	}
	return &GroupResponse{Body: groupView(group)}, nil
}

func (s *Server) listGroups(ctx context.Context, _ *struct{}) (*GroupListResponse, error) {
	groups, err := s.store.ListGroups(ctx)
	if err != nil {
		return nil, err
	}
	result := make([]GroupBody, 0, len(groups))
	for _, group := range groups {
		result = append(result, groupView(group))
	}
	return &GroupListResponse{Body: GroupListBody{Groups: result}}, nil
}

func (s *Server) addGroupAgent(ctx context.Context, input *GroupAgentInput) (*GroupAgentsResponse, error) {
	group, err := s.store.GetGroup(ctx, input.ID)
	if err != nil {
		return nil, notFoundError(err)
	}
	if _, err := s.store.GetAgent(ctx, input.Body.AgentID); err != nil {
		return nil, notFoundError(err)
	}
	if err := s.store.AddAgentToGroup(ctx, group.ID, input.Body.AgentID); err != nil {
		return nil, err
	}
	agents, err := s.store.ListGroupAgents(ctx, group.ID)
	if err != nil {
		return nil, err
	}
	result := make([]AgentBody, 0, len(agents))
	for _, agent := range agents {
		result = append(result, agentView(agent))
	}
	return &GroupAgentsResponse{Body: GroupAgentsBody{Group: groupView(group), Agents: result}}, nil
}

func (s *Server) createBatch(ctx context.Context, input *CreateBatchInput) (*BatchResponse, error) {
	if input.Body.Games < 1 {
		return nil, huma.Error422UnprocessableEntity("games must be at least 1")
	}
	if len(input.Body.AgentIDs) < 2 {
		return nil, huma.Error422UnprocessableEntity("at least two agents are required")
	}
	seen := make(map[int64]struct{}, len(input.Body.AgentIDs))
	for _, id := range input.Body.AgentIDs {
		if _, ok := seen[id]; ok {
			return nil, huma.Error422UnprocessableEntity("agent_ids must be distinct")
		}
		seen[id] = struct{}{}
		if _, err := s.store.GetAgent(ctx, id); err != nil {
			return nil, notFoundError(err)
		}
	}
	pairs := make([][2]int64, 0, len(input.Body.AgentIDs)*(len(input.Body.AgentIDs)-1)/2)
	for i := 0; i < len(input.Body.AgentIDs); i++ {
		for j := i + 1; j < len(input.Body.AgentIDs); j++ {
			pairs = append(pairs, [2]int64{input.Body.AgentIDs[i], input.Body.AgentIDs[j]})
		}
	}
	batch, err := s.store.CreateBatch(ctx, input.Body.Games, pairs)
	if err != nil {
		return nil, err
	}
	return &BatchResponse{Body: batchView(batch, nil)}, nil
}

func (s *Server) createAutoBatch(ctx context.Context, input *CreateAutoBatchInput) (*BatchResponse, error) {
	if input.Body.Games < 1 {
		return nil, huma.Error422UnprocessableEntity("games must be at least 1")
	}
	agents, err := s.store.AgentRatings(ctx)
	if err != nil {
		return nil, err
	}
	if len(agents) < 2 {
		return nil, huma.Error422UnprocessableEntity("at least two agents are required")
	}
	pairs := pairing.Pair(agents, 0)
	if len(pairs) == 0 {
		return nil, huma.Error422UnprocessableEntity("at least two agents are required")
	}
	batch, err := s.store.CreateBatch(ctx, input.Body.Games, pairs)
	if err != nil {
		return nil, err
	}
	return &BatchResponse{Body: batchView(batch, nil)}, nil
}

func (s *Server) listBatches(ctx context.Context, input *ListBatchesInput) (*BatchListResponse, error) {
	batches, err := s.store.ListBatches(ctx, input.Limit)
	if err != nil {
		return nil, err
	}
	result := make([]BatchBody, 0, len(batches))
	for _, batch := range batches {
		result = append(result, batchView(batch, nil))
	}
	return &BatchListResponse{Body: BatchListBody{Batches: result}}, nil
}

func (s *Server) getBatch(ctx context.Context, input *IDInput) (*BatchResponse, error) {
	batch, matches, err := s.store.GetBatch(ctx, input.ID)
	if err != nil {
		return nil, notFoundError(err)
	}
	return &BatchResponse{Body: batchView(batch, matches)}, nil
}

func (s *Server) listMatches(ctx context.Context, input *ListMatchesInput) (*MatchListResponse, error) {
	matches, err := s.store.ListMatches(ctx, input.AgentID, input.Limit)
	if err != nil {
		return nil, err
	}
	result := make([]MatchBody, 0, len(matches))
	for _, match := range matches {
		result = append(result, matchView(match))
	}
	return &MatchListResponse{Body: MatchListBody{Matches: result}}, nil
}

func (s *Server) getMatch(ctx context.Context, input *IDInput) (*MatchDetailResponse, error) {
	match, games, err := s.store.GetMatch(ctx, input.ID)
	if err != nil {
		return nil, notFoundError(err)
	}
	gameViews := make([]GameBody, 0, len(games))
	for _, game := range games {
		gameViews = append(gameViews, gameView(game))
	}
	return &MatchDetailResponse{Body: MatchDetailBody{Match: matchView(match), Games: gameViews}}, nil
}

func (s *Server) ranking(ctx context.Context, _ *struct{}) (*RankingResponse, error) {
	ranking, err := s.store.Ranking(ctx)
	if err != nil {
		return nil, err
	}
	result := make([]RankingRowBody, 0, len(ranking))
	for _, row := range ranking {
		result = append(result, RankingRowBody{
			Agent: row.Name, AgentID: row.AgentID, Mu: row.Mu, Sigma: row.Sigma,
			Games: row.Games, Win: row.Win, Draw: row.Draw, Loss: row.Loss, WinRate: row.WinRate,
		})
	}
	return &RankingResponse{Body: RankingListBody{Ranking: result}}, nil
}

func agentView(agent store.Agent) AgentBody {
	// Agent names follow stat_v<n>/rl_v<n> by convention; the server intentionally does not validate that.
	return AgentBody{ID: agent.ID, Name: agent.Name, Kind: agent.Kind, FilePath: agent.FilePath,
		BuiltinName: agent.BuiltinName, RatingMu: agent.RatingMu, RatingSigma: agent.RatingSigma,
		CreatedAt: formatTime(agent.CreatedAt)}
}

func groupView(group store.Group) GroupBody {
	return GroupBody{ID: group.ID, Name: group.Name, CreatedAt: formatTime(group.CreatedAt)}
}

func matchView(match store.Match) MatchBody {
	return MatchBody{ID: match.ID, BatchID: match.BatchID, AgentAID: match.AgentAID, AgentBID: match.AgentBID,
		AgentAName: match.AgentAName, AgentBName: match.AgentBName, Games: match.Games, Seed: match.Seed,
		Status: match.Status, Error: match.Error, WinA: match.WinA, Draw: match.Draw, LossA: match.LossA,
		DurationMs: match.DurationMs, CreatedAt: formatTime(match.CreatedAt)}
}

func gameView(game store.Game) GameBody {
	return GameBody{ID: game.ID, MatchID: game.MatchID, Index: game.Index, Seed: game.Seed, ScoreA: game.ScoreA,
		ScoreB: game.ScoreB, Result: game.Result, Turns: game.Turns, DurationMs: game.DurationMs, ReplayPath: game.ReplayPath}
}

func batchView(batch store.Batch, matches []store.Match) BatchBody {
	result := BatchBody{ID: batch.ID, Status: batch.Status, Games: batch.Games, Error: batch.Error,
		CreatedAt: formatTime(batch.CreatedAt), StartedAt: formatTime(batch.StartedAt), FinishedAt: formatTime(batch.FinishedAt),
		FailedCount: batch.FailedCount, DoneCount: batch.DoneCount}
	if matches != nil {
		result.Matches = make([]MatchBody, 0, len(matches))
		for _, match := range matches {
			result.Matches = append(result.Matches, matchView(match))
		}
	}
	return result
}

func formatTime(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	return value.UTC().Format(time.RFC3339Nano)
}

func duplicateError(err error, message string) error {
	if errors.Is(err, store.ErrDuplicate) {
		return huma.Error409Conflict(message)
	}
	return err
}

func notFoundError(err error) error {
	if errors.Is(err, store.ErrNotFound) {
		return huma.Error404NotFound("resource not found")
	}
	return err
}
