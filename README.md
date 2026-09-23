# RL-Template

強化学習の Kaggle コンペティション（ボット対ボットのシミュレーション形式）を始めるための GitHub テンプレートです。

Rust を主言語にします。環境、探索、推論、自己対戦、対戦評価、提出用バイナリは Rust で書きます。Python は torch で学習を回す部分にだけ使います。身内で対戦させるサーバーは Go、その画面は TS で書きます。

各ファイルの中身は `TODO` だけで、実装は入っていません。技術選定とファイル構成を決めたものです。進め方は `CLAUDE.md` にあります。

## 使い方

GitHub のリポジトリページで `Use this template` を押すか、次のコマンドで新しいリポジトリを作ります。

```bash
gh repo create <name> --template CHU1PC/RL-Template
```

## 構成

```text
RL-Template/
├── CLAUDE.md                # コンペをこのテンプレートから始める手順
├── Cargo.toml               # workspace: core / infer / arena / submission / py
├── pyproject.toml           # maturin。crates/py を Python の拡張モジュールにする
│
├── crates/                  # Rust。学習と提出の本体
│   ├── core/                # env / encode / search / agent / arena ループ / Policy trait
│   ├── infer/               # ort（ONNX Runtime）でバッチ推論
│   ├── arena/               # bin: selfplay / match。clap、rayon、serde_json、tracing
│   ├── submission/          # bin: Kaggle に提出するバイナリ
│   └── py/                  # PyO3。解析とデバッグ用
│
├── rl_template/             # Python。torch の学習だけ。uv で管理
│   ├── ppo/
│   ├── alphazero/
│   ├── data/                # arena selfplay が書いた samples を読む
│   └── export/              # torch → ONNX
│
├── configs/                 # 学習の設定（YAML）。Python は pydantic、Rust は serde_yml
├── docs/
│   ├── competition/         # 参加するコンペの情報
│   │   ├── rules.md         # ルール、評価方式、提出形式、締切
│   │   ├── notebooks.md     # 投票数の上位10件の公開 notebook の要点
│   │   └── discussions.md   # 投票数の上位10件の discussion の要点
│   ├── research/            # 調べた内容。ソース1つにつき1ファイル
│   └── changelog/
│       ├── improvements/    # 実装して良くなったもの
│       └── downgrades/      # 実装して悪くなったもの
│
└── ladder/                  # 身内で対戦させるサービス
    ├── README.md            # 画面と機能の要件
    ├── Dockerfile
    ├── server/              # Go。chi、huma、modernc.org/sqlite。arena をサブプロセスで実行
    └── web/                 # TS。React + Vite、shadcn/ui、TanStack Query + Router、kubb
```

## ツールチェーン


| 領域            | 言語     | ツール        |
| ------------- | ------ | ---------- |
| crates        | Rust   | cargo      |
| rl\_template  | Python | uv、maturin |
| ladder/server | Go     | go         |
| ladder/web    | TS     | bun        |


