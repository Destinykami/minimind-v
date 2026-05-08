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


AUDIO_PRETRAIN_PROMPTS = [
    "<audio>",
    "语音转文字：<audio>",
]

AUDIO_SFT_PROMPTS = [
    "请把这段中文语音转写成文字：<audio>",
    "帮我识别这段录音里的内容：<audio>",
    "听一下这段音频，说出里面讲了什么：<audio>",
    "请转录这段普通话语音：<audio>",
    "把这段语音里的内容完整写出来：<audio>",
    "请识别这段音频中的中文文本：<audio>",
    "我给你一段录音，请输出对应文字：<audio>",
    "请帮我做语音转写：<audio>",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build omni_sft.parquet by merging local sft_i2t parquet and AISHELL audio data."
    )
    parser.add_argument(
        "--i2t-parquet",
        default="dataset/sft_i2t.parquet",
        help="本地 i2t parquet 路径，通常是 dataset/sft_i2t.parquet。",
    )
    parser.add_argument(
        "--aishell-root",
        default="dataset/AISHELL-1/data_aishell",
        help="AISHELL-1 的 data_aishell 根目录。当不指定 --aishell-parquet 时，从这里读取原始 wav+transcript。",
    )
    parser.add_argument(
        "--aishell-parquet",
        default="dataset/aishell1_train_sft.parquet",
        help="可选：直接使用已经生成好的 AISHELL SFT parquet；为空时回退到 aishell-root 原始目录。",
    )
    parser.add_argument(
        "--output-path",
        default="dataset/omni_sft.parquet",
        help="输出的 omni parquet 路径。",
    )
    parser.add_argument(
        "--audio-splits",
        nargs='+',
        default=["train"],
        choices=["train", "dev", "test"],
        help="从 AISHELL 哪些 split 抽取音频样本。只有在从原始目录构建时生效。",
    )
    parser.add_argument(
        "--audio-style",
        default="sft",
        choices=["pretrain", "sft"],
        help="音频样本的 prompt 风格。构建 omni_sft 时建议用 sft。",
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
        default=40000,
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
            transcripts[utt_id] = ''.join(parts[1:])
    return transcripts


def choose_audio_prompt(style: str, rng: random.Random):
    if style == 'pretrain':
        return AUDIO_PRETRAIN_PROMPTS[0] if rng.random() < 0.85 else AUDIO_PRETRAIN_PROMPTS[1]
    if style == 'sft':
        return rng.choice(AUDIO_SFT_PROMPTS)
    raise ValueError(f'Unsupported audio style: {style}')


def build_audio_conversations(transcript: str, style: str, rng: random.Random):
    return [
        {'role': 'user', 'content': choose_audio_prompt(style, rng)},
        {'role': 'assistant', 'content': transcript},
    ]


def normalize_audio_path(audio_path: str, abs_audio_path: bool):
    path = resolve_path(audio_path)
    return str(path if abs_audio_path else path.relative_to(PROJECT_ROOT))


def load_i2t_rows(i2t_path: Path, max_samples: int):
    table = pa.Table.from_batches(pq.ParquetFile(i2t_path).iter_batches())
    rows = []
    total = len(table)
    limit = total if max_samples <= 0 else min(total, max_samples)

    for idx in range(limit):
        rows.append({
            'conversations': table['conversations'][idx].as_py(),
            'image_bytes': table['image_bytes'][idx].as_py() if 'image_bytes' in table.column_names else None,
            'image_names': table['image_names'][idx].as_py() if 'image_names' in table.column_names else None,
            'audio_path': None,
            'utt_id': None,
            'speaker': None,
            'split': None,
            'source': 'i2t',
            'modality': 'image',
        })
    return rows


def load_aishell_rows_from_parquet(aishell_parquet: Path, max_samples: int, abs_audio_path: bool):
    table = pa.Table.from_batches(pq.ParquetFile(aishell_parquet).iter_batches())
    rows = []
    total = len(table)
    limit = total if max_samples <= 0 else min(total, max_samples)

    for idx in range(limit):
        audio_path = table['audio_path'][idx].as_py() if 'audio_path' in table.column_names else None
        if audio_path is None:
            continue
        rows.append({
            'conversations': table['conversations'][idx].as_py(),
            'image_bytes': None,
            'image_names': None,
            'audio_path': normalize_audio_path(audio_path, abs_audio_path),
            'utt_id': table['utt_id'][idx].as_py() if 'utt_id' in table.column_names else None,
            'speaker': table['speaker'][idx].as_py() if 'speaker' in table.column_names else None,
            'split': table['split'][idx].as_py() if 'split' in table.column_names else None,
            'source': 'aishell_parquet',
            'modality': 'audio',
        })
    return rows


def load_aishell_rows_from_root(aishell_root: Path, splits, style: str, max_samples: int, abs_audio_path: bool, rng: random.Random):
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
                'source': 'aishell_root',
                'modality': 'audio',
            })
            if max_samples > 0 and len(rows) >= max_samples:
                return rows
    return rows


def load_aishell_rows(args, rng: random.Random):
    aishell_parquet = resolve_path(args.aishell_parquet) if args.aishell_parquet else None
    if aishell_parquet and aishell_parquet.exists():
        return load_aishell_rows_from_parquet(
            aishell_parquet=aishell_parquet,
            max_samples=args.max_audio_samples,
            abs_audio_path=args.abs_audio_path,
        )

    aishell_root = resolve_path(args.aishell_root)
    return load_aishell_rows_from_root(
        aishell_root=aishell_root,
        splits=args.audio_splits,
        style=args.audio_style,
        max_samples=args.max_audio_samples,
        abs_audio_path=args.abs_audio_path,
        rng=rng,
    )


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
    output_path = resolve_path(args.output_path)

    if not i2t_path.exists():
        raise SystemExit(f'i2t parquet not found: {i2t_path}')

    image_rows = load_i2t_rows(i2t_path, args.max_image_samples)
    audio_rows = load_aishell_rows(args, rng)

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
