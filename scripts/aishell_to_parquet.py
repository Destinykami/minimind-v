import argparse
import json
import random
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is required for parquet conversion. Run with `.venv/bin/python` or install pyarrow first."
    ) from exc


# 更弱、更统一的 pretrain 风格模板。
# 这里故意尽量少写任务描述，让模型更像在做模态对齐/继续预训练。
PRETRAIN_PROMPTS = [
    "<audio>",
    "语音转文字：<audio>",
]

# 更自然、多样的 sft 风格模板。
# 这些模板更接近日常问法，适合作为监督微调数据。
SFT_PROMPTS = [
    "请把这段中文语音转写成文字：<audio>",
    "帮我识别这段录音里的内容：<audio>",
    "听一下这段音频，说出里面讲了什么：<audio>",
    "请转录这段普通话语音：<audio>",
    "把这段语音里的内容完整写出来：<audio>",
    "请识别这段音频中的中文文本：<audio>",
    "我给你一段录音，请输出对应文字：<audio>",
    "请帮我做语音转写：<audio>",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AISHELL-1 into AudioLM parquet datasets with pretrain and SFT styles."
    )
    parser.add_argument(
        "--aishell-root",
        default="dataset/AISHELL-1/data_aishell",
        help="AISHELL-1 data_aishell root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Output directory for generated parquet files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "dev", "test"],
        help="Which AISHELL splits to convert.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["pretrain", "sft"],
        choices=["pretrain", "sft"],
        help="Which output styles to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for prompt sampling.",
    )
    parser.add_argument(
        "--abs-path",
        action="store_true",
        help="Store absolute audio_path instead of path relative to repo root.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap per split for quick debugging. 0 means no cap.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    path = Path(path_str)
    return path if path.is_absolute() else (root / path).resolve()


def load_transcripts(transcript_path: Path):
    transcripts = {}
    with transcript_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid transcript line {line_no}: {line}")
            utt_id = parts[0]
            # AISHELL 转写里常带空格分词，这里默认拼回连续文本，更适合中文转写监督。
            text = "".join(parts[1:])
            transcripts[utt_id] = text
    return transcripts


def choose_prompt(style: str, rng: random.Random):
    if style == "pretrain":
        # pretrain 风格保持更统一：大多数情况下用最弱模板，只少量加入轻微提示。
        return PRETRAIN_PROMPTS[0] if rng.random() < 0.85 else PRETRAIN_PROMPTS[1]
    if style == "sft":
        return rng.choice(SFT_PROMPTS)
    raise ValueError(f"Unsupported style: {style}")


def build_conversations(style: str, transcript: str, rng: random.Random):
    prompt = choose_prompt(style, rng)
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": transcript},
    ]


def iter_split_samples(split_dir: Path, transcripts: dict, store_absolute: bool, repo_root: Path, max_samples: int):
    count = 0
    for wav_path in sorted(split_dir.rglob("*.wav")):
        utt_id = wav_path.stem
        text = transcripts.get(utt_id)
        if text is None:
            continue
        speaker = wav_path.parent.name
        audio_path = wav_path.resolve() if store_absolute else wav_path.resolve().relative_to(repo_root)
        yield {
            "utt_id": utt_id,
            "speaker": speaker,
            "split": split_dir.name,
            "text": text,
            "audio_path": str(audio_path),
        }
        count += 1
        if max_samples > 0 and count >= max_samples:
            break


def write_style_parquet(samples, style: str, output_path: Path, seed: int):
    rng = random.Random(seed)
    rows = []
    for sample in samples:
        conversations = build_conversations(style, sample["text"], rng)
        rows.append({
            "conversations": json.dumps(conversations, ensure_ascii=False),
            "audio_path": sample["audio_path"],
            "utt_id": sample["utt_id"],
            "speaker": sample["speaker"],
            "split": sample["split"],
            "text": sample["text"],
            "style": style,
        })

    table = pa.Table.from_pylist(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    print(f"Wrote {len(rows)} rows -> {output_path}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    aishell_root = resolve_path(args.aishell_root)
    output_dir = resolve_path(args.output_dir)
    transcript_path = aishell_root / "transcript" / "aishell_transcript_v0.8.txt"

    if not transcript_path.exists():
        raise SystemExit(f"Transcript file not found: {transcript_path}")

    transcripts = load_transcripts(transcript_path)

    all_samples = []
    for split in args.splits:
        split_dir = aishell_root / "wav" / split
        if not split_dir.exists():
            raise SystemExit(f"Split directory not found: {split_dir}")
        split_samples = list(
            iter_split_samples(
                split_dir,
                transcripts,
                store_absolute=args.abs_path,
                repo_root=repo_root,
                max_samples=args.max_samples,
            )
        )
        print(f"Collected {len(split_samples)} samples from {split_dir}")
        all_samples.extend(split_samples)

    if not all_samples:
        raise SystemExit("No valid AISHELL samples found.")

    split_tag = "-".join(args.splits)
    for idx, style in enumerate(args.styles):
        output_path = output_dir / f"aishell1_{split_tag}_{style}.parquet"
        write_style_parquet(all_samples, style=style, output_path=output_path, seed=args.seed + idx)


if __name__ == "__main__":
    main()
