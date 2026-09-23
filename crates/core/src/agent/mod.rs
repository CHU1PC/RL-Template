// TODO: Policy trait と推論時の方策を実装し、random と rule policy を置く。
//   入力:     env / encode / search の型、rl_infer の ONNX 推論 API。
//             ONNX を読む実装は rl_infer に置き、ここへ持ち込まない。
//   出力:     search、arena、crates/arena、crates/submission が使う Policy。
//   完了条件: フェーズ 3 の完了条件「ユーザーが停止と言ったこと」。
//   決めること:
//     - batch の観測から logits と value を返す Policy trait を、同期か非同期かにするか。
//     - エラー型と、探索と提出の両方から使える形をどう定めるか。
//     - ルール方策の版名を stat_v<自然数> に統一するか別形式にするか。
//       採用した版は書き換えず、stat_v1、stat_v2 のように増やす。
