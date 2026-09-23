// TODO: env の状態をテンソル(固定長の配列)に encode し、合法手 mask も作る。
//   入力:     env の State、rl_template.config.TrainConfig、export/onnx.py。
//             obs_dim と n_actions は TrainConfig と書き出した ONNX の入出力に一致させる。
//   出力:     rl_core::search / rl_core::agent と Python の export が使う。
//             入力 obs f32 [B, obs_dim]、mask bool [B, n_actions]。
//             出力 logits f32 [B, n_actions]、value f32 [B]。
//   完了条件: フェーズ 4 の順序表の1行目「1局面をテンソルにして形が合う」。
//   決めること:
//     - PPO 用と AlphaZero 用で encoder を共通化するか分けるか。
//     - State の特徴量と action index の並びをどう定めるか。
//     - 合法手が無い局面を encode 側で拒否するか env の終局で保証するか。
