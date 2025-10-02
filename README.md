# llm-env-template

llmの学習や評価を行うための環境のテンプレート

## Quick Start

1. [設定](docker-compose.yml)を編集する
- container-name: コンテナ名（他のコンテナと衝突しないように注意）
- ports: コンテナの公開ポート（他のコンテナと衝突しないように注意）
- device_ids: GPUサーバー内で使用するGPUのidを指定する

2. コンテナで開く