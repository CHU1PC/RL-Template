# 実装済み: batch.py の PPO/AlphaZero バッチと samples.py の safetensors ローダーを持つ。
# safetensors のキー、dtype、shape は samples.py のモジュール docstring に定義する。
# TODO: Rust selfplay のサンプルを学習ループへ接続する。観測は Rust 側でエンコード済みなので、ここで作り直さない。
