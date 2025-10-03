---
description: 'Description of the custom chat mode.'
tools: ['runCommands', 'runTasks', 'edit', 'runNotebooks', 'search', 'new', 'extensions', 'todos', 'runTests', 'usages', 'vscodeAPI', 'problems', 'changes', 'testFailure', 'openSimpleBrowser', 'fetch', 'githubRepo', 'getPythonEnvironmentInfo', 'getPythonExecutableCommand', 'installPythonPackage', 'configurePythonEnvironment']
model: Claude Sonnet 4.5 (Preview) (copilot)
---
# codebase
既存の実装の構造を確認したうえで、どこに追加を加えればいいのかを考えてから実装に進みましょう。
このプロジェクトはHydraを使って設定管理を行っています。
設定はYAMLファイルで行い、コマンドライン引数で上書きが可能です。
また、PyTorch Lightningを使ってモデルの学習ループを管理しています。
README.mdに記載されている主要な機能と設定のカスタマイズ方法を参考にしてください。

実装が完了したら、必ず正しく動くかどうかを確認しましょう。
また、README.mdに新しい実験設定の例を追加してください。
