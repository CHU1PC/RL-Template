/*
Package runner implements the contract between the ladder and arena.

arena match --agent <name>=<path> --agent <name>=<path> --games <N> --seed <S> --json

stdout is exactly one JSON object:

	{"games":[{"index":0,"seed":123,"players":["a","b"],"result":{"a":1.0,"b":0.0},"turns":57,"duration_ms":12}],
	 "summary":{"a":{"win":3,"draw":1,"loss":1},"b":{"win":1,"draw":1,"loss":3}}}

result values are per player: 1.0 win, 0.5 draw, 0.0 loss. N-player generalises to one score per player.

Go structs (exact names and JSON tags):
*/
package runner

// Output is the complete stdout object from arena.
type Output struct {
	Games   []Game            `json:"games"`
	Summary map[string]Record `json:"summary"`
}

// Game is one game's result.
type Game struct {
	Index      int                `json:"index"`
	Seed       int64              `json:"seed"`
	Players    []string           `json:"players"`
	Result     map[string]float64 `json:"result"`
	Turns      int                `json:"turns"`
	DurationMs int                `json:"duration_ms"`
}

// Record is a player's aggregate record.
type Record struct {
	Win  int `json:"win"`
	Draw int `json:"draw"`
	Loss int `json:"loss"`
}
