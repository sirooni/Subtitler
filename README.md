# Subtitler

動画ファイルから音声を抽出し、WhisperXを用いて文字起こしを行い、SRT字幕ファイルとDaVinci Resolve用のFCPXMLファイルを生成するツールです。

## 処理フロー

本ツールは以下の5ステップで処理を実行します。

1. **音声抽出**: `ffmpeg` を使用して動画から音声を抽出します。
2. **文字起こし**: WhisperX を使用して音声をテキスト化します。
3. **アラインメント**: 音声とテキストのタイミングを微調整します。
4. **形態素解析**: Sudachi を使用して日本語を文節や単語単位で適切に分割します。
5. **出力**: 整形されたSRTファイルと、DaVinci Resolve (Text+) 用のFCPXMLファイルを生成します。

## 必要要件

- Python 3.8+
- ffmpeg
- ffprobe
- CUDA (GPUを使用する場合)

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python subtitler.py <動画ファイルパス> [オプション]
```

### オプション

- `--skip-transcribe`: 文字起こしをスキップし、既存の `segments.json` を使用します。
- `--skip-align`: アラインメントをスキップし、既存の `aligned.json` を使用します。
- `--skip-tokenize`: 形態素解析をスキップし、既存の `tokens.json` を使用します。
- `--import-srt`: SRT生成（文字起こし〜SRT書き出し）をスキップし、既存のSRTファイルを読み込みます。
- `--srt-path <path>`: インポートするSRTファイルのパスを指定します（`--import-srt` と併用）。省略時はデフォルトの出力パスを使用します。

### 出力

実行すると `outputs/<動画ファイル名>/` ディレクトリに以下のファイルが生成されます。

- `audio.wav`: 抽出された音声
- `<動画ファイル名>.srt`: SRT形式の字幕ファイル
- `<動画ファイル名>.fcpxml`: DaVinci Resolve (Text+) 用のタイムラインファイル
- 中間JSONファイル (`segments.json`, `aligned.json`, `tokens.json`)

## DaVinci Resolve へのインポート

1. DaVinci Resolve を開き、プロジェクトを作成または開きます。
2. メニューから **File > Import > Timeline...** を選択します。
3. 生成された `.fcpxml` ファイルを選択します。
4. 設定ダイアログが表示されたら、必要に応じて設定を確認し **OK** をクリックします。
5. タイムラインに字幕が編集可能なクリップとして配置されます。
