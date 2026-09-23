// TODO: match で複数の Policy を N 局対戦させ、stdout に契約 JSON を1つ出す。
//   契約: argv は arena match --agent <name>=<path> --agent <name>=<path> --games <N> --seed <S> --json とする。
//         stdout は次の JSON オブジェクトを1つだけ出力する。
//           {"games":[{"index":0,"seed":123,"players":["a","b"],"result":{"a":1.0,"b":0.0},"turns":57,"duration_ms":12}],
//            "summary":{"a":{"win":3,"draw":1,"loss":1},"b":{"win":1,"draw":1,"loss":3}}}
//         games は局ごとの結果、summary は win/draw/loss の集計結果を持つ。
//         result の値はプレイヤーごとに、勝ちは 1.0、引き分けは 0.5、負けは 0.0 とする。
//         N 人対戦では、プレイヤーごとに1つのスコアを持つ。
//         契約の正本は ladder/server/internal/runner/contract.go とする。
//   入力:     ladder/README.md、match の引数、rl_core::arena の結果と棋譜。
//   出力:     ladder/server が読む stdout JSON と --replay-dir の棋譜。
//             ladder 経由の対戦では arena match に常に棋譜を書かせる。
//   完了条件: フェーズ 1 の「ランダム同士の対戦が並列で回る
//             （1000局の局数と所要時間が出る）」かつ、
//             フェーズ 2 の「提出から順位表まで通る」。
//   決めること:
//     - 棋譜ファイル名をどう付けるか、棋譜の形式を何にするか。
