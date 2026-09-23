// TODO: PyO3 で rl_core の型を包み、解析とデバッグ用に Python へ公開する。
//   入力:     rl_core の公開型、PyO3 の #[pyclass] / #[pymethods] API。
//   出力:     局面の確認と単発 rollout に使う Python 拡張。
//             ロジックはここに書かず、学習の本線では使わない。
//   完了条件: CLAUDE.md の「crates/py は解析で必要になったときに埋める。
//             順序には入れない」。
//   決めること:
//     - 解析へ公開する rl_core の型と、Python へコピーするデータを選ぶか。
//     - 単発 rollout の API を同期呼び出しにするか、別の debug API にするか。

use pyo3::prelude::*;

#[pymodule]
fn _core(_m: &Bound<PyModule>) -> PyResult<()> {
    Ok(())
}
