// TODO: ONNX または組み込み Policy で自己対戦し、学習データを書き出す。
//   入力:     rl_core::encode の tensor、Policy、指定した ONNX weights。
//   出力:     safetensors を rl_template/data/samples.py が読む。
//
// | Algorithm | Key | dtype | shape |
// | --- | --- | --- | --- |
// | PPO | `obs` | `f32` | `[N, obs_dim]` |
// | PPO | `mask` | `bool` | `[N, n_actions]` |
// | PPO | `actions` | `i64` | `[N]` |
// | PPO | `old_log_probs` | `f32` | `[N]` |
// | PPO | `advantages` | `f32` | `[N]` |
// | PPO | `returns` | `f32` | `[N]` |
// | AlphaZero | `obs` | `f32` | `[N, obs_dim]` |
// | AlphaZero | `mask` | `bool` | `[N, n_actions]` |
// | AlphaZero | `policy_target` | `f32` | `[N, n_actions]` |
// | AlphaZero | `value_target` | `f32` | `[N]` |
//
// `mask` は必須とし、マスクしない場合は全要素を `True` として保存する。
//   完了条件: フェーズ 4 の順序表の4行目「ルール方策同士の対局を
//             samples に書き出せる」。
//   決めること:
//     - PPO 用と AlphaZero 用のどちらを生成するか、別ファイルにするか。
//     - 1局ごとに保存するか、N 行へまとめて保存するか。
