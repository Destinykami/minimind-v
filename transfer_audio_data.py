from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq

root = Path("dataset/AISHELL-1/data_aishell")
transcript_file = root / "transcript" / "aishell_transcript_v0.8.txt"

trans = {}
with transcript_file.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        utt_id = parts[0]
        # AISHELL 常见格式里词之间有空格；做 LLM/ASR 监督时通常去掉
        text = "".join(parts[1:])
        trans[utt_id] = text

rows = []
for wav_path in (root / "wav" / "train").rglob("*.wav"):
    utt_id = wav_path.stem
    text = trans.get(utt_id)
    if not text:
        continue
    conversations = [
        {"role": "user", "content": "请转写这段中文普通话音频：<audio>"},
        {"role": "assistant", "content": text},
    ]
    rows.append({
        "conversations": json.dumps(conversations, ensure_ascii=False),
        "audio_path": str(wav_path.resolve()),
    })

table = pa.Table.from_pylist(rows)
pq.write_table(table, "dataset/aishell1_train_audio.parquet")
