# ladder

身内で Agent を対戦させて、順位と対戦の中身をブラウザで見るためのサービスです。コンペに依存しない部分なので、テンプレートのまま次のコンペでも使います。

## 構成

| 部分 | 技術 |
| --- | --- |
| server | Go。chi、huma（OpenAPI を `/api/openapi.yaml` で配信）、modernc.org/sqlite |
| web | TS。React + Vite、bun、shadcn/ui + Tailwind、TanStack Query + TanStack Router、kubb で API の型を生成 |
| 対戦の実行 | server が `crates/arena` をサブプロセスで実行し、stdout の JSON を読む |
| 保存 | SQLite とローカルのファイルシステム |
| 認証 | 範囲外。Tailscale かリバースプロキシで塞ぐ |

## arena との契約

server は `arena match` を次の引数で実行する。

```text
arena match --agent <name>=<path> --agent <name>=<path> --games <N> --seed <S> --json
```

stdout は次の JSON オブジェクトを1つだけ出力する。

```json
{"games":[{"index":0,"seed":123,"players":["a","b"],"result":{"a":1.0,"b":0.0},"turns":57,"duration_ms":12}],
 "summary":{"a":{"win":3,"draw":1,"loss":1},"b":{"win":1,"draw":1,"loss":3}}}
```

`games` は局ごとの結果、`summary` は `win` / `draw` / `loss` の集計結果を持つ。`result` の値はプレイヤーごとに、勝ちは 1.0、引き分けは 0.5、負けは 0.0 とする。N 人対戦では、プレイヤーごとに1つのスコアを持つ。契約の正本は `ladder/server/internal/runner/contract.go` とする。

## 用語

| 用語 | 意味 |
| --- | --- |
| Agent | 登録された1つの方策。モデル名と重み（ONNX）を持つ。動作確認用に組み込みの方策（ランダム、ヒューリスティック）も登録できる |
| Group | Agent に付けるタグ。1つの Agent は複数の Group に所属できる |
| match | A vs B を N 局回す単位 |
| batch | 一括で登録した match のまとまり。履歴は batch ごとに保存する |
| game | match の中の1局。win / draw / lose の結果と棋譜を持つ |

## 画面

左パネルに Ranking、Models、Match の3つのタブを置く。

### Ranking

| 列 | 内容 |
| --- | --- |
| 順位 | スコアの降順 |
| モデル名 | クリックすると Agent の詳細へ移る |
| スコア | レーティングの値 |
| 標準偏差 | レーティングの不確かさ |
| 勝率 | 全対戦の勝率 |

### Models

一覧と、Agent の詳細の2画面を持つ。

一覧でできること。

- 今までの Agent を全部見る
- 新規 Agent を登録する
- Agent の Group を編集する

Agent の詳細でできること。

- ヘッダーに、モデル名、標準偏差、対戦数、勝率、スコアを出す
- 対戦相手別の成績を win-draw-lose で出す
- 対戦履歴を出す。クリックすると対戦の詳細へ移る

### Match

一覧と、対戦の詳細の2画面を持つ。

一覧でできること。

- 今までの対戦を batch ごとに見る
- 新規 match を登録する。どのモデルとどのモデルを何局対戦させるかを決める
- Group を使って一括で登録する。A vs Group α は、A と α の全員の match になる。Group α vs Group β は、α と β の直積の match になる

対戦の詳細でできること。

- A vs B の win-draw-lose と、B に対する A の勝率を出す
- 各局にリプレイを見るボタンを置く

## 決まっていること

| 項目 | 内容 |
| --- | --- |
| DB の実体 | agents、groups、agent_groups（多対多）、batches、matches、games |
| 一括登録の展開 | server 側で Group を Agent に展開して、match を作る |
| 棋譜 | ladder 経由の対戦では、`arena match` に常に棋譜を書かせる。リプレイのボタンはこれを読む |
| リプレイの描画 | ゲームごとに違う。web の replay viewer だけがコンペ固有の差し替え部品になる |
| 結果の3値 | win / draw / lose。引き分けを持つゲームに合わせる |
| 静的ファイルの配信 | server が `LADDER_WEB_DIR` の web の dist を配信する |
| 組み合わせ | グループを無視して μ が最も近い相手と組む |
| Dockerfile | `ladder/Dockerfile` に多段ビルドがある（Rust の arena、Go の server、bun の web） |

## 決めること

| 項目 | 候補 |
| --- | --- |
| レーティングの方式 | TODO。Kaggle と同じ N(μ, σ²) の方式にし、μ0 は設定で変えられる（`LADDER_RATING_MU0` / `LADDER_RATING_SIGMA0`） |
| 棋譜の形式 | `crates/arena` 側で決める |
