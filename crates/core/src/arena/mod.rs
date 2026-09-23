// TODO: 複数の Policy で対戦ループを回し、結果と棋譜を返す。
//   入力:     env、Policy、対戦回数、docs/competition/rules.md。
//   出力:     crates/arena の match と selfplay、ladder の結果集計。
//             CLI と学習の自己対戦はこのループを共有する。
//   完了条件: フェーズ 1 の「ランダム同士の対戦が並列で回ること」。
//             arena match で1000局を回し、局数と所要時間が出る。
//   決めること:
//     - 結果を win/draw/lose で返し、評価方法を勝率かレーティングにするか。
//     - 評価方法を trait にして、コンペごとに差し替えられる形にするか。
