import os
import sys
import json
import winsound
import subprocess
import whisperx
from sudachipy import tokenizer
from sudachipy import dictionary

# ===== 設定 =====
language = "ja"
whisperx_model = "large-v3"
device = "cpu"  # GPU: "cuda", CPU: "cpu"
compute_type = "float32"  # float16で失敗する場合はfloat32
min_chars = 5  # 1字幕の最低文字数
max_chars = 20  # 1字幕の最大文字数
min_duration = 1.0  # 最低表示時間(秒)
gap_threshold = 0.25  # セリフ切れ目の閾値(秒)
long_token_threshold = 0.1  # 文節の閾値(秒)

# Sudachi設定
sudachi_tokenizer = dictionary.Dictionary().create()
sudachi_mode = tokenizer.Tokenizer.SplitMode.C


# ===== 音声抽出 =====
def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        audio_path,
    ]
    subprocess.run(cmd, check=True)


# ===== WhisperXで文字起こし =====
def transcribe_whisperx(audio_path):
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(whisperx_model, device, compute_type=compute_type)
    result = model.transcribe(audio, language=language)
    return result


# ===== アラインメント =====
def align_segments(audio_path, segments):
    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    aligned_result = whisperx.align(
        segments, align_model, metadata, audio_path, device=device
    )
    return aligned_result


# ===== Sudachiで日本語分割 =====
def tokenize_japanese(aligned_segments, save_json=None):
    tokens = []
    for seg in aligned_segments:
        if "words" not in seg:
            continue

        words = seg["words"]
        if not words:
            continue

        # 1つのsegmentの全textを結合
        text = "".join([w["word"] for w in words if "word" in w])

        # Sudachiで形態素解析
        sudachi_tokens = sudachi_tokenizer.tokenize(text, sudachi_mode)

        # 解析した単語を順番にaligned wordsに対応付ける
        char_index = 0
        for m in sudachi_tokens:
            surface = m.surface()
            length = len(surface)

            # 対応する音声の時間を取得（文字ベースで対応）
            word_start = None
            word_end = None
            for i in range(length):
                if char_index + i < len(words):
                    if word_start is None:
                        word_start = words[char_index + i].get("start")
                    word_end = words[char_index + i].get("end")
            char_index += length

            tokens.append({"surface": surface, "start": word_start, "end": word_end})

    # JSON保存
    if save_json:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)

    return tokens


# ===== SRT整形 =====
def format_srt_with_tokens(
    tokens,
    min_chars=5,
    max_chars=20,
    min_duration=2.0,
    gap_threshold=0.5,
    long_token_threshold=0.2,
):
    srt_lines = []
    idx = 1
    current_text = ""
    current_start = None
    current_end = None

    for token in tokens:
        text = token["surface"]
        start = token["start"]
        end = token["end"]

        # 無効データスキップ
        if start is None or end is None:
            continue

        # 新しいブロックを開始する条件
        should_split = False
        if current_text and (start - current_end > gap_threshold):
            should_split = True
        if len(current_text) + len(text) > max_chars:
            should_split = True

        if should_split:
            # 既存の字幕ブロックを追加
            if current_text:
                duration = max(current_end - current_start, min_duration)
                srt_lines.append(
                    {
                        "index": idx,
                        "start": current_start,
                        "end": current_start + duration,
                        "text": current_text.strip(),
                    }
                )
                idx += 1
            # 新しいブロック開始
            current_text = text
            current_start = start
            current_end = end
        else:
            if not current_text:
                current_start = start
            current_text += text
            current_end = end

        is_phrase_finished = (
            end - start
        ) >= long_token_threshold or text in "。、！？!?"
        over_min_chars = len(current_text) >= min_chars
        if is_phrase_finished and over_min_chars:
            if current_text:
                duration = max(current_end - current_start, min_duration)
                srt_lines.append(
                    {
                        "index": idx,
                        "start": current_start,
                        "end": current_start + duration,
                        "text": current_text.strip(),
                    }
                )
                idx += 1
            # 新しいブロック開始
            current_text = ""
            current_start = start
            current_end = end

    # 最後のブロックを追加
    if current_text:
        duration = max(current_end - current_start, min_duration)
        srt_lines.append(
            {
                "index": idx,
                "start": current_start,
                "end": current_start + duration,
                "text": current_text.strip(),
            }
        )

    return srt_lines


# ===== SRT書き込み =====
def write_srt(srt_lines, srt_path):
    def format_timestamp(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for line in srt_lines:
            f.write(f"{line['index']}\n")
            f.write(
                f"{format_timestamp(line['start'])} --> {format_timestamp(line['end'])}\n"
            )
            f.write(f"{line['text']}\n\n")


# ===== メイン処理 =====
def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python subtitler.py <video_file> [--skip-transcribe] [--skip-align] [--skip-tokenize]"
        )
        sys.exit(1)

    video_path = sys.argv[1]

    base = os.path.splitext(os.path.basename(video_path))[0]
    outputs_dir = os.path.join("outputs", base)
    os.makedirs(outputs_dir, exist_ok=True)
    audio_path = os.path.join(outputs_dir, "audio.wav")
    srt_path = os.path.join(outputs_dir, f"{base}.srt")
    seg_json = os.path.join(outputs_dir, "segments.json")
    aligned_json = os.path.join(outputs_dir, "aligned.json")
    token_json = os.path.join(outputs_dir, "tokens.json")

    skip_transcribe = "--skip-transcribe" in sys.argv
    skip_align = "--skip-align" in sys.argv
    skip_tokenize = "--skip-tokenize" in sys.argv

    print("[1/5] Extracting audio...")
    extract_audio(video_path, audio_path)
    winsound.Beep(1000, 200)

    if skip_transcribe and os.path.exists(seg_json):
        print("[2/5] Skipping transcription, loading segments.json...")
        with open(seg_json, "r", encoding="utf-8") as f:
            segments = json.load(f)["segments"]
    else:
        print("[2/5] Transcribing with WhisperX...")
        result = transcribe_whisperx(audio_path)
        segments = result["segments"]
        with open(seg_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    winsound.Beep(1000, 200)

    if skip_align and os.path.exists(aligned_json):
        print("[3/5] Skipping alignment, loading aligned.json...")
        with open(aligned_json, "r", encoding="utf-8") as f:
            aligned_result = json.load(f)
    else:
        print("[3/5] Aligning segments...")
        aligned_result = align_segments(audio_path, segments)
        with open(aligned_json, "w", encoding="utf-8") as f:
            json.dump(aligned_result, f, ensure_ascii=False, indent=2)
    winsound.Beep(1000, 200)

    if skip_tokenize and os.path.exists(token_json):
        print("[4/5] Skipping tokenization, loading tokens.json...")
        with open(token_json, "r", encoding="utf-8") as f:
            tokens = json.load(f)
    else:
        print("[4/5] Tokenizing with Sudachi B-mode...")
        tokens = tokenize_japanese(aligned_result["segments"], save_json=token_json)
    winsound.Beep(1000, 200)

    print(f"[5/5] Formatting and writing SRT to {srt_path} ...")
    srt_lines = format_srt_with_tokens(
        tokens, min_chars, max_chars, min_duration, gap_threshold, long_token_threshold
    )
    write_srt(srt_lines, srt_path)
    winsound.Beep(1000, 200)

    print("Done.")


if __name__ == "__main__":
    main()
