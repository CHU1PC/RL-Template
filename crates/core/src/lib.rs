// TODO: モジュールを宣言し、Policy trait を公開する場所として保つ。
//   入力:     README.md、CLAUDE.md、各 module の型と依存関係。
//   出力:     rl_core を使う rl_infer、crates/arena、crates/submission。
//   完了条件: フェーズ1で env が埋まった時点で pub mod env が機能し、以降は各モジュールの完了条件に従う。
//   決めること:
//     - 依存の向きを env <- encode <- search <- agent <- arena に固定するか。
//     - Policy trait を agent で定義し、lib.rs から再公開するか。

pub mod agent;
pub mod arena;
pub mod encode;
pub mod env;
pub mod search;
