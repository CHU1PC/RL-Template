// TODO: env と encode を使う探索を実装し、葉の評価を Policy に渡す。
//   入力:     env の遷移、encode の tensor、rl_core::agent::Policy。
//   出力:     rl_core::agent と rl_core::arena が使う action 選択。
//   完了条件: フェーズ 4 の順序表の2行目「組み込みの評価で1手読める」。
//   決めること:
//     - MCTS か alpha-beta か。
//     - 不完全情報では determinization か IS-MCTS か。
//     - 葉をまとめて Policy に渡し、batch=1 の推論を回さない実装にするか。
