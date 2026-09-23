package pairing

import "sort"

type AgentID = int64

type AgentRating struct {
	ID AgentID
	Mu float64
}

// Pair sorts by rating and pairs neighbouring agents. Groups are intentionally ignored.
// レーティング TODO が未実装なので、現在は全 Agent が μ=600 になり、任意の隣接ペアに退化する。これは想定どおりである。
// TODO: Kaggle はおおむね submission ごとに 1 日 8 エピソードを実行し、新しい submission を優先する。
func Pair(agents []AgentRating, n int) [][2]AgentID {
	if len(agents) < 2 {
		return nil
	}
	ordered := append([]AgentRating(nil), agents...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Mu == ordered[j].Mu {
			return ordered[i].ID < ordered[j].ID
		}
		return ordered[i].Mu > ordered[j].Mu
	})
	maxPairs := len(ordered) / 2
	if n > 0 && n < maxPairs {
		maxPairs = n
	}
	pairs := make([][2]AgentID, 0, maxPairs)
	for i := 0; i < maxPairs*2; i += 2 {
		pairs = append(pairs, [2]AgentID{ordered[i].ID, ordered[i+1].ID})
	}
	return pairs
}
