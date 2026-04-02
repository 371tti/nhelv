<div align="center">
<h1>Nhelv</h1>
</div>

Nhelv(ネルフ) は、Discord 上で会話・ツール実行・VOICEVOX 読み上げを統合して動かす Rust 製の bot です。

Observer-rust の後継プロジェクトとして、スタンドアローン動作を前提に設計されており、OpenAI API を利用した会話機能を中心に、様々なツールや機能を提供します。

## 機能
- 会話: OpenAI API を利用した会話機能 モデル選択可能 レート制限可能 システムプロンプト設定可能
  - tools:
    - get_time.rs: 国コードから現在の時刻を取得
    - latex.rs: TeX 数式を画像化
    - modal_builder.rs: モーダルダイアログを動的に構築
    - discord.rs: Discord 操作系ツール（例: メッセージ送信、チャンネル管理）
    - voicevox.rs: VOICEVOX 操作系ツール（例: 話者変更、スタイル変更、読み上げ）
- VOICEVOX 読み上げ: VC内でのテキスト読み上げ機能 話者・スタイル選択可能 自動読み上げ機能
- TeX 数式の画像化: TeX 数式を画像化して Discord に送信する機能

## セットアップ

1. リリースからバイナリをダウンロード

2. `.env` を作成

```dotenv
# required
DISCORD_TOKEN=xxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# 以下optional
SYSTEM_PROMPT=あなたのシステムプロンプト
VOICEVOX_DEFAULT_SPEAKER=3
VOICEVOX_CORE_ACCELERATION=auto
VOICEVOX_CORE_CPU_THREADS=0
VOICEVOX_CORE_LOAD_ALL_MODELS=false
VOICEVOX_OUTPUT_SAMPLING_RATE=48000
VOICEVOX_PRELOAD_ON_STARTUP=true
VOICEVOX_OPEN_JTALK_DICT_DIR=voicevox_core/dict/open_jtalk_dic_utf_8-1.11
VOICEVOX_VVM_DIR=voicevox_core/models/vvms
VOICEVOX_ONNXRUNTIME_FILENAME=voicevox_core/onnxruntime/lib/voicevox_onnxruntime.dll
```

3. 起動

## 設定

現在の実装では、環境変数（`.env`）から読み込みます。
`DISCORD_TOKEN` と `OPENAI_API_KEY` は未設定だと起動時にエラーになります。

## 主なコマンド

この bot は slash command と prefix command（`!`）の両方を登録します。

基本:

- `/ping`
- `/enable`
- `/disable`
- `/clear`
- `/model`
- `/rate_config`
- `/set_system_prompt`
- `/tex_expr`

VC / TTS:

- `/vc_join [auto_read]`
- `/vc_leave`
- `/vc_say <text>`
- `/vc_system <enabled>`
- `/vc_autoread <enabled>`
- `/vc_dict <source> <target>`
- `/vc_speaker ...`
- `/vc_status`

## 開発用チェック

```powershell
cargo clippy --all-targets --all-features -- -D warnings
```

fmtはやってない。

## トラブルシュート

- `DISCORD_TOKEN must be set`:
	`.env` に `DISCORD_TOKEN` を設定してください。
- `OPENAI_API_KEY must be set`:
	`.env` に `OPENAI_API_KEY` を設定してください。
- VOICEVOX 関連で辞書/モデル/onnxruntime が見つからない:
	`.env` の VOICEVOX 系パス、または `voicevox_core` 配下のディレクトリ構成を確認してください。

## Third-party licenses

- KaTeX font のライセンスは [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) に記載しています。
- `KaTeX_font` ディレクトリを再配布する場合は、上記ライセンス表記を同梱してください。

ダウンロードコンテンツはREADMEを含むのでそちらを参照。

## Thx
名前考えてくれたt3tra-devさんとVCの機能アイデア上げてくれたtabosanさんに感謝
