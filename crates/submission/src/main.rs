// TODO: Kaggle 環境で動く提出用バイナリを作り、依存を増やさず届ける。
//   入力:     docs/competition/rules.md、提出形式、rl_core と rl_infer の API。
//             サイズ上限と Kaggle の実行環境、1局の入出力契約も読む。
//   出力:     コンペの提出 artifact と、Kaggle 環境で1局動く実行物。
//   完了条件: フェーズ 4 の順序表の7行目「提出形式でビルドできる」。
//             CLAUDE.md の「提出」の節にあるサイズ上限と、Kaggle の環境で1局動くこと。
//   決めること:
//     - rl_core と rl_infer だけを使う境界を保ち、提出入口をどう切り替えるか。
//     - tar.gz にバイナリを同梱するか、Python の agent 関数から呼ぶか。
fn main() {}
