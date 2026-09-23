package pairing

import (
	"reflect"
	"testing"
)

func TestPairSortsAndCapsWithoutMutation(t *testing.T) {
	input := []AgentRating{
		{ID: 10, Mu: 500},
		{ID: 2, Mu: 900},
		{ID: 7, Mu: 700},
		{ID: 4, Mu: 800},
		{ID: 1, Mu: 600},
	}
	original := append([]AgentRating(nil), input...)
	want := [][2]AgentID{{2, 4}, {7, 1}}
	if got := Pair(input, 0); !reflect.DeepEqual(got, want) {
		t.Fatalf("Pair() = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(input, original) {
		t.Fatalf("Pair mutated input: got %v, want %v", input, original)
	}
	if got := Pair(input, 1); !reflect.DeepEqual(got, want[:1]) {
		t.Fatalf("Pair(n=1) = %v, want %v", got, want[:1])
	}
	if got := Pair([]AgentRating{{ID: 1, Mu: 600}}, 0); got != nil {
		t.Fatalf("Pair(one agent) = %v, want nil", got)
	}
}

func TestPairBreaksTiesByID(t *testing.T) {
	got := Pair([]AgentRating{{ID: 3, Mu: 600}, {ID: 1, Mu: 600}, {ID: 2, Mu: 600}, {ID: 4, Mu: 600}}, 0)
	want := [][2]AgentID{{1, 2}, {3, 4}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Pair() = %v, want %v", got, want)
	}
}
