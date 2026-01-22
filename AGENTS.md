# Agent Interaction Rules

**CRITICAL:**
README.md を読まずに勝手な推論や実行を行うエージェントは、プロジェクトの整合性を損なうため厳禁です。
必ず README.md を読み、プロジェクトの文脈と手順を理解した上で行動してください。

## テストの実行方法
ホスト環境で直接 `pytest` を実行してはいけません。必ず Docker コンテナを使用して実行してください。
`Makefile` が用意されているため、以下のコマンドで実行可能です。

```bash
make test
```

または直接実行する場合：
```bash
docker run --rm  --gpus all -v `pwd`:/workspace cw-decoder python3 -m pytest tests/
```