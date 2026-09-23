package rating

import "context"

// GameResult is one game's result in seat order.
type GameResult struct {
	MatchID  int64
	GameIdx  int
	AgentIDs []int64
	Scores   []float64
}

// Rater updates ratings from one game result.
type Rater interface {
	Update(ctx context.Context, r GameResult) error
}

// NoopRater leaves ratings unchanged until the rating policy is decided.
type NoopRater struct{}

func (NoopRater) Update(ctx context.Context, r GameResult) error { return nil }

/*
レーティング実装の TODO

（1）固定（共通）
Kaggle ConnectX 2020 と Lux AI Season 3 2024 の Evaluation の節を 2026-09-22
に読み、次の事実が両方で同じだと確認した。

- skill は正規分布 N(μ, σ²) として扱う。
- μ の初期値は 600 である。
- 対戦相手はレーティングが近い者同士で組まれる。
- 勝てば μ が上がり、負ければ下がり、引き分けは両者を互いの中間へ寄せる。
- 更新の大きさは、期待された結果からのずれに比例し、かつ各自の σ に比例する。
- σ は情報が増えるほど小さくなる。
- リーダーボードに出るのは、その参加者の最良の submission の μ である。
- σ の初期値と更新の係数は公開されていない。
- この挙動は TrueSkill の2人対戦の更新式と一致する。Glicko-2 はこれに volatility
  の項を足したものである。

（2）コンペごとに違う例（Lux AI Season 3 の場合）
以下は Lux AI Season 3 という一つのコンペの例であり、一般的なルールではない。
対象とするコンペでは異なる可能性がある。

- 1チームにつき最新の2 submission だけが対戦に参加する。
- validation の episode では、submission を自分自身の複製と対戦させる。
- レーティングは、その episode に参加した全 submission について更新される（N人対戦）。
- スコアの差（何点差で勝ったか）は更新量に影響しない。
- 新しい submission ほど多くの episode が割り当てられる。

決めること（後で決める項目）
- σ の初期値（σ0）。
- β。
- τ（dynamics）。
- TrueSkill の 1v1 閉形式を使うか、Glicko-2 を使うか、Kaggle の記述から導き直すか。
- 引き分けの扱い。
- 3人以上（N人対戦）の扱い。
- 毎ゲーム逐次更新にするか、バッチで全再計算にするか。
- 順位表に μ を使うか、μ−kσ を使うか。
- フェーズ0で参加するコンペの Evaluation のページを読み、その μ0 と、上の固定の項目との
  違いを docs/competition/rules.md に記録すること。
*/
