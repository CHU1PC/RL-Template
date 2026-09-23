// TODO: clap で selfplay と match の2つのサブコマンドを構成する。
//   入力:     clap の引数、rl_core の対戦 loop、rl_infer の model。
//   出力:     selfplay の safetensors と ladder/server が読む match の stdout JSON。
//             stdout は結果専用、tracing のログは stderr にし、局は rayon で並列化する。
//   完了条件: フェーズ 1 の「ランダム同士の対戦が並列で回ること」。
//             arena match で1000局を回し、局数と所要時間が出る。
//   決めること:
//     - selfplay と match の引数、組み込み方策の指定方法をどうするか。

mod match_;
mod selfplay;

fn main() {}
