// TODO: ゲームの状態、行動、合法手、状態遷移、終局判定を書く。
//   入力:     docs/competition/rules.md、公式 replay、kaggle-environments。
//             env は他 module に依存せず、観測の encode はここに置かない。
//   出力:     rl_core::encode、rl_core::search、rl_core::arena、arena の match/selfplay。
//   完了条件: フェーズ 1 の4条件を全て満たす。
//             「公式の対局記録を再生して一致する」
//             「生成した対局でも一致する」
//             「ルール文の項目ごとに検証がある」
//             「ルール文に無い挙動が一致する」
//   決めること:
//     - 手番制(turn-based)か同時手番(simultaneous moves)か。
//     - 不完全情報があるか. State とプレイヤーごとの Observation を分けるか。
//     - 偶然手(chance events)の rng の持ち方と決定論的な再現の保証方法。
//     - プレイヤー数を2人零和にするか N 人にするか。
//     - outcome を win/draw/lose の3値、スコア、順位のどれで表すか。
