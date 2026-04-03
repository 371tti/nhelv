# VOICEVOX Issue Draft（feature flag の link-onnxruntime のやつが更新されてないねってやつ）

## 概要
`voicevox_core` を以下の構成で使用すると:

- `link-onnxruntime`
- `buildtime-download-onnxruntime`

`vv_bin` ベースの VVM がすべて `InvalidModelData`（モデルデータ読み込み失敗）で読み込めなくなることがあります。  
ただし、VVM ファイル自体は有効な ZIP コンテナであり、メタデータも正常に読めます。

## 影響
- 起動処理自体は一部完了する場合があるが、モデル読み込みが選択されたすべての VVM で失敗する。
- その結果、音声合成経路が使用不能になる（`failed to load any usable VVM`）。

## 再現手順（概要）
1. `voicevox_core` を `link-onnxruntime` と `buildtime-download-onnxruntime` を有効にしてビルドする。
2. `VVCORE_BUILD_DOWNLOAD_AND_COPY_ORT=1` を設定する。
3. アプリを起動し、VOICEVOX core を preload する。
4. VVM の読み込み失敗が繰り返され、最終的に使用可能なモデルが 1 つもなくなることを確認する。

## 観測されたエラーパターン
- `failed to load VVM '...': ... model data could not be read ...`
- 最終的に: `failed to load any usable VVM under '.../vvms'`

## 根本原因（確認済み）
`voicevox_core` の実行時には、`vv_bin` を扱うために **VOICEVOX パッチ版 ONNX Runtime** が必要です。  
実行時には次のマーカーで判定しています:

- `ort::info().starts_with("VOICEVOX ORT Build Info: ")`

しかし、build-time downloader / linker の経路では通常の `onnxruntime` アーティファクトが解決される場合があります:

- release tag: `onnxruntime-{VERSION}`
- asset prefix: `onnxruntime-*`
- link name: `onnxruntime`

この組み合わせでは `vv_bin` をサポートしていないため、すべてのモデル読み込みが `InvalidModelData` で失敗します。

## これが VVM 破損問題ではない理由
- VVM ファイルは構造的に正しい ZIP である。
- `manifest.json` および `metas.json` は正常に読み出せる。
- 失敗しているのはモデルセッション生成・ロード段階であり、`vv_bin` を期待する runtime と実際の runtime の不一致が原因である。

## 提案する Upstream 修正
1. `voicevox_core_build_features` のダウンロード処理では、このモードにおいてデフォルトで VOICEVOX Runtime 資産を使うようにする:
   - release tag: `voicevox_onnxruntime-{VERSION}`
   - asset prefix: `voicevox_onnxruntime-*`  
     （または既存の命名規則に沿った `voicevox_` プレフィックス付きのアーティファクト名）
2. リンク処理では `voicevox_onnxruntime` ライブラリ名を優先して使用する。
3. `vv_bin` を読み込もうとしているのに非 VOICEVOX ORT が検出された場合、明確な警告またはエラーを出す。


## このリポジトリで適用したローカルパッチ
- vendor した `voicevox_core_build_features` に対して以下を修正:
  - ダウンロード先を `voicevox_onnxruntime-*` の release/tag 経路に変更
  - リンク名を `voicevox_onnxruntime` に変更
- このパッチ適用後、preload は成功し、モデルも正常にロードされた
  - `model_count=25`
  - `style_count=127`
