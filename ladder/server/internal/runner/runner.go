package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
)

// AgentRef identifies an arena agent. Path is an ONNX file path or builtin policy name.
type AgentRef struct {
	Name string
	Path string
}

// Spec describes one arena match invocation.
type Spec struct {
	Agents []AgentRef
	Games  int
	Seed   int64
}

// Runner invokes the arena binary.
type Runner struct {
	bin string
}

// New returns a Runner that invokes bin. A bare name is resolved on PATH.
func New(bin string) *Runner {
	return &Runner{bin: bin}
}

// Run invokes arena and decodes its JSON output.
func (r *Runner) Run(ctx context.Context, spec Spec) (*Output, error) {
	if len(spec.Agents) < 2 {
		return nil, fmt.Errorf("arena binary %q requires at least 2 agents", r.bin)
	}
	if spec.Games < 1 {
		return nil, fmt.Errorf("arena binary %q requires games >= 1", r.bin)
	}

	args := make([]string, 0, len(spec.Agents)*2+4)
	for _, agent := range spec.Agents {
		args = append(args, "--agent", agent.Name+"="+agent.Path)
	}
	args = append(args, "--games", fmt.Sprintf("%d", spec.Games), "--seed", fmt.Sprintf("%d", spec.Seed), "--json")

	cmd := exec.CommandContext(ctx, r.bin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if cmd.Process == nil {
			return nil, fmt.Errorf("arena binary %q could not start: %w", r.bin, err)
		}
		return nil, fmt.Errorf("arena binary %q failed: %w; stderr: %q", r.bin, err, stderrPrefix(stderr.Bytes()))
	}

	var output Output
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		return nil, fmt.Errorf("arena binary %q returned invalid JSON: %w; stderr: %q", r.bin, err, stderrPrefix(stderr.Bytes()))
	}
	return &output, nil
}

func stderrPrefix(stderr []byte) string {
	if len(stderr) > 1024 {
		stderr = stderr[:1024]
	}
	return string(stderr)
}
