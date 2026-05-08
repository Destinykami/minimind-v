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


# AudioLM pretrain 阶段尽量保持弱提示，减少过强的任务模板偏置。
AUDIO_PRETRAIN_PROMPTS = [
    "<audio>",
    "语音转文字：<audio>",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build omni_pretrain.parquet by merging local i2t parquet and AISHELL audio data."
    )
    parser.add_argument(
        "--i2t-parquet",
        default="dataset/pretrain_i2t.parquet",
        help="本地 i2t parquet 路径，通常是 dataset/pretrain_i2t.parquet。",
    )
    parser.add_argument(
        "--aishell-root",
        default="dataset/AISHELL-1/data_aishell",
        help="AISHELL-1 的 data_aishell 根目录。",
    )
    parser.add_argument(
        "--output-path",
        default="dataset/omni_pretrain.parquet",
        help="输出的 omni parquet 路径。",
    )
    parser.add_argument(
        "--audio-splits",
        nargs='+',
        default=["train"],
        choices=["train", "dev", "test"],
        help="从 AISHELL 哪些 split 抽取音频样本。通常 pretrain 用 train。",
    )
    parser.add_argument(
        "--audio-style",
        default="pretrain",
        choices=["pretrain", "sft"],
        help="音频样本的 prompt 风格。构建 omni_pretrain 时建议用 pretrain。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于打乱和 prompt 采样。",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="是否在合并后打乱样本顺序。",
    )
    parser.add_argument(
        "--max-image-samples",
        type=int,
        default=0,
        help="最多取多少条 i2t 图像样本，0 表示全部。",
    )
    parser.add_argument(
        "--max-audio-samples",
        type=int,
        default=0,
        help="最多取多少条 AISHELL 音频样本，0 表示全部。",
    )
    parser.add_argument(
        "--image-repeat",
        type=int,
        default=1,
        help="图像样本重复多少次，用于调节图像/音频比例。",
    )
    parser.add_argument(
        "--audio-repeat",
        type=int,
        default=1,
        help="音频样本重复多少次，用于调节图像/音频比例。",
    )
    parser.add_argument(
        "--abs-audio-path",
        action="store_true",
        help="是否把 audio_path 写成绝对路径。默认写仓库相对路径。",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_transcripts(transcript_path: Path):
    transcripts = {}
    with transcript_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f'Invalid transcript line {line_no}: {line}')
            utt_id = parts[0]
            # AISHELL 转写文件里常带空格分词，这里拼回连续中文字符串。
            transcripts[utt_id] = ''.join(parts[1:])
    return transcripts


def choose_audio_prompt(style: str, rng: random.Random):
    if style == 'pretrain':
        return AUDIO_PRETRAIN_PROMPTS[0] if rng.random() < 0.85 else AUDIO_PRETRAIN_PROMPTS[1]
    return '请把这段中文语音转写成文字：<audio>'


def build_audio_conversations(transcript: str, style: str, rng: random.Random):
    return [
        {'role': 'user', 'content': choose_audio_prompt(style, rng)},
        {'role': 'assistant', 'content': transcript},
    ]


def load_i2t_rows(i2t_path: Path, max_samples: int):
    table = pa.Table.from_batches(pq.ParquetFile(i2t_path).iter_batches())
    rows = []
    total = len(table)
    limit = total if max_samples <= 0 else min(total, max_samples)

    for idx in range(limit):
        conversations = table['conversations'][idx].as_py()
        image_bytes = table['image_bytes'][idx].as_py() if 'image_bytes' in table.column_names else None
        image_names = table['image_names'][idx].as_py() if 'image_names' in table.column_names else None
        rows.append({
            'conversations': conversations,
            'image_bytes': image_bytes,
            'image_names': image_names,
            'audio_path': None,
            'utt_id': None,
            'speaker': None,
            'split': None,
            'source': 'i2t',
            'modality': 'image',
        })
    return rows


def load_aishell_rows(aishell_root: Path, splits, style: str, max_samples: int, abs_audio_path: bool, rng: random.Random):
    transcript_path = aishell_root / 'transcript' / 'aishell_transcript_v0.8.txt'
    if not transcript_path.exists():
        raise FileNotFoundError(f'Transcript file not found: {transcript_path}')

    transcripts = load_transcripts(transcript_path)
    rows = []

    for split in splits:
        split_dir = aishell_root / 'wav' / split
        if not split_dir.exists():
            raise FileNotFoundError(f'Split directory not found: {split_dir}')

        for wav_path in sorted(split_dir.rglob('*.wav')):
            utt_id = wav_path.stem
            text = transcripts.get(utt_id)
            if text is None:
                continue

            audio_path = wav_path.resolve() if abs_audio_path else wav_path.resolve().relative_to(PROJECT_ROOT)
            rows.append({
                'conversations': json.dumps(build_audio_conversations(text, style, rng), ensure_ascii=False),
                'image_bytes': None,
                'image_names': None,
                'audio_path': str(audio_path),
                'utt_id': utt_id,
                'speaker': wav_path.parent.name,
                'split': split,
                'source': 'aishell',
                'modality': 'audio',
            })
            if max_samples > 0 and len(rows) >= max_samples:
                return rows
    return rows


def repeat_rows(rows, times: int):
    if times <= 1:
        return rows
    expanded = []
    for _ in range(times):
        expanded.extend(rows)
    return expanded


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    i2t_path = resolve_path(args.i2t_parquet)
    aishell_root = resolve_path(args.aishell_root)
    output_path = resolve_path(args.output_path)

    if not i2t_path.exists():
        raise SystemExit(f'i2t parquet not found: {i2t_path}')

    image_rows = load_i2t_rows(i2t_path, args.max_image_samples)
    audio_rows = load_aishell_rows(
        aishell_root=aishell_root,
        splits=args.audio_splits,
        style=args.audio_style,
        max_samples=args.max_audio_samples,
        abs_audio_path=args.abs_audio_path,
        rng=rng,
    )

    image_rows = repeat_rows(image_rows, args.image_repeat)
    audio_rows = repeat_rows(audio_rows, args.audio_repeat)
    all_rows = image_rows + audio_rows

    if not all_rows:
        raise SystemExit('No rows collected; please check your inputs.')

    if args.shuffle:
        rng.shuffle(all_rows)

    table = pa.Table.from_pylist(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

    print(f'Wrote {len(all_rows)} rows -> {output_path}')
    print(f'  image rows: {len(image_rows)}')
    print(f'  audio rows: {len(audio_rows)}')
    print(f'  columns: {table.column_names}')


if __name__ == '__main__':
    main()
