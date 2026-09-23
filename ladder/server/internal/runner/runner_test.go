package runner

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestContractJSONRoundTrip(t *testing.T) {
	literal := `{"games":[{"index":0,"seed":123,"players":["a","b"],"result":{"a":1.0,"b":0.0},"turns":57,"duration_ms":12}],"summary":{"a":{"win":3,"draw":1,"loss":1},"b":{"win":1,"draw":1,"loss":3}}}`
	var output Output
	if err := json.Unmarshal([]byte(literal), &output); err != nil {
		t.Fatal(err)
	}
	if len(output.Games) != 1 || output.Games[0].Index != 0 || output.Games[0].Seed != 123 ||
		!reflect.DeepEqual(output.Games[0].Players, []string{"a", "b"}) ||
		output.Games[0].Result["a"] != 1.0 || output.Games[0].Result["b"] != 0.0 ||
		output.Games[0].Turns != 57 || output.Games[0].DurationMs != 12 {
		t.Fatalf("decoded game = %#v", output.Games[0])
	}
	if output.Summary["a"] != (Record{Win: 3, Draw: 1, Loss: 1}) || output.Summary["b"] != (Record{Win: 1, Draw: 1, Loss: 3}) {
		t.Fatalf("decoded summary = %#v", output.Summary)
	}
	encoded, err := json.Marshal(output)
	if err != nil {
		t.Fatal(err)
	}
	var roundTripped Output
	if err := json.Unmarshal(encoded, &roundTripped); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(roundTripped, output) {
		t.Fatalf("round trip = %#v, want %#v", roundTripped, output)
	}
}

func TestRunFakeArena(t *testing.T) {
	dir := t.TempDir()
	argsPath := filepath.Join(dir, "args.txt")
	scriptPath := filepath.Join(dir, "fake-arena.sh")
	const output = `{"games":[{"index":0,"seed":123,"players":["a","b"],"result":{"a":1,"b":0},"turns":57,"duration_ms":12}],"summary":{"a":{"win":1,"draw":0,"loss":0},"b":{"win":0,"draw":0,"loss":1}}}`
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > '" + argsPath + "'\nprintf '%s\\n' '" + output + "'\n"
	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := New(scriptPath).Run(context.Background(), Spec{
		Agents: []AgentRef{{Name: "a", Path: "/tmp/a.onnx"}, {Name: "b", Path: "builtin"}},
		Games:  3,
		Seed:   42,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Games[0].Seed != 123 || got.Summary["a"].Win != 1 {
		t.Fatalf("decoded output = %#v", got)
	}
	args, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatal(err)
	}
	wantArgs := "--agent\na=/tmp/a.onnx\n--agent\nb=builtin\n--games\n3\n--seed\n42\n--json\n"
	if string(args) != wantArgs {
		t.Fatalf("argv = %q, want %q", string(args), wantArgs)
	}
}

func TestRunMissingBinaryReturnsError(t *testing.T) {
	var runErr error
	panicked := false
	func() {
		defer func() {
			if recover() != nil {
				panicked = true
			}
		}()
		_, runErr = New("definitely-not-an-arena-binary").Run(context.Background(), Spec{
			Agents: []AgentRef{{Name: "a", Path: "a"}, {Name: "b", Path: "b"}},
			Games:  1,
		})
	}()
	if panicked {
		t.Fatal("Run panicked for missing binary")
	}
	if runErr == nil || !strings.Contains(runErr.Error(), "definitely-not-an-arena-binary") {
		t.Fatalf("Run error = %v", runErr)
	}
}
