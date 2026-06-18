# sefia_litellm

[LiteLLM](https://github.com/BerriAI/litellm) を使って各種 LLM プロバイダに接続する
`sefia` 用の `LLMClient` 実装です。

```python
import sefia_litellm

client = sefia_litellm.LiteLLMClient(model="gpt-4o")
```

## ログ出力の抑制

LiteLLM はデフォルトで verbose なログ（`Provider List: ...` バナーや例外時のデバッグ
情報など）を出力します。`LiteLLMClient` はこれらを**既定で抑制**します。

切り替え方法は 2 つあります。

- **コンストラクタ引数** `suppress_logs`（最優先）

  ```python
  # ログを抑制（既定）
  client = sefia_litellm.LiteLLMClient(model="gpt-4o")
  client = sefia_litellm.LiteLLMClient(model="gpt-4o", suppress_logs=True)

  # LiteLLM のログをそのまま出す
  client = sefia_litellm.LiteLLMClient(model="gpt-4o", suppress_logs=False)
  ```

- **環境変数** `SEFIA_LITELLM_SUPPRESS_LOGS`（`suppress_logs=None` のときの既定値）

  ```bash
  # 抑制を無効化（ログを出す）。0/false/no/off が無効化扱い
  export SEFIA_LITELLM_SUPPRESS_LOGS=false
  ```

  `suppress_logs` を明示した場合は環境変数より優先されます。未設定なら抑制 ON です。

抑制 ON のときは `litellm.suppress_debug_info = True` を設定し、標準 `logging` の
`"LiteLLM"` ロガーを `WARNING` レベルに引き上げます（INFO/DEBUG を抑制）。

## import が遅い件について

LiteLLM の `import` は重く、1 秒前後かかることがあります
（参考: [BerriAI/litellm#7605](https://github.com/BerriAI/litellm/issues/7605)）。

本パッケージでは以下の緩和策を取っています。

1. **遅延 import** — LiteLLM は実際にリクエストを送るメソッド内でのみ import します。
   そのため `sefia_litellm` を import しただけではコストはかかりません。初回リクエスト
   後は `sys.modules` にキャッシュされ、以降の import は実質ゼロコストです。これが最大の
   緩和策です。

2. **モデルコストマップのローカル化** — import 前に
   `LITELLM_LOCAL_MODEL_COST_MAP=True` を設定し、LiteLLM がコストマップをネットワークから
   取得するのを回避してバンドル済みの JSON を使わせます。import の高速化に加え、オフライン
   でも安定して動作します。

   最新モデルの価格が必要でローカルマップが古い場合は、環境変数で従来動作に戻せます。

   ```bash
   export LITELLM_LOCAL_MODEL_COST_MAP=False
   ```

3. **（任意）起動時ウォームアップ** — 初回リクエストのレイテンシも隠したい場合は、アプリ
   起動時にバックグラウンドで import を済ませておけます。

   ```python
   import asyncio

   async def _warmup() -> None:
       await asyncio.to_thread(__import__, "litellm")

   # 起動時に投げておく
   asyncio.create_task(_warmup())
   ```
