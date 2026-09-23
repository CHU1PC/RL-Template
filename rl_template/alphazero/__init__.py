# 実装済み: alphazero_update と AlphaZeroConfig がある。探索 (MCTS) と自己対戦は Rust 側にある。
# TODO: 学習ループと CLI 入口を書く。TrainConfig を rl_template.config.load_config で読み、
# load_alphazero_samples を呼び、cfg.epochs 回 alphazero_update を実行してから export_onnx を呼ぶ。
