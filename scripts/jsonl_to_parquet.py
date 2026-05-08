import argparse
import json
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is required for parquet conversion. Install it with `pip install pyarrow` or use `.venv/bin/python`."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OCR-style JSONL data into MiniMind-V parquet format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path. Each line should contain image path(s) and conversations.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Base directory used to resolve relative image paths. Defaults to the JSONL parent directory.",
    )
    return parser.parse_args()


def resolve_image_paths(sample, image_root: Path):
    image_field = sample.get("image", sample.get("images"))
    if image_field is None:
        raise KeyError("Missing `image` or `images` field.")

    image_paths = image_field if isinstance(image_field, list) else [image_field]
    if not image_paths or not all(isinstance(path, str) and path.strip() for path in image_paths):
        raise ValueError("`image`/`images` must be a non-empty string or list of strings.")

    resolved = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_absolute():
            path = (image_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        resolved.append(path)
    return resolved


def normalize_conversations(sample):
    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("`conversations` must be a non-empty list.")

    for idx, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            raise ValueError(f"`conversations[{idx}]` must be a dict.")
        if "content" not in turn:
            raise ValueError(f"`conversations[{idx}]` is missing `content`.")

    return json.dumps(conversations, ensure_ascii=False)


def convert_jsonl_to_parquet(input_path: Path, output_path: Path, image_root: Path):
    conversation_values = []
    image_values = []
    has_multi_image_sample = False

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                conversations = normalize_conversations(sample)
                image_paths = resolve_image_paths(sample, image_root)
                image_bytes = [path.read_bytes() for path in image_paths]
            except Exception as exc:
                raise ValueError(f"Failed to parse line {line_no} in {input_path}: {exc}") from exc

            conversation_values.append(conversations)
            if len(image_bytes) == 1:
                image_values.append(image_bytes[0])
            else:
                has_multi_image_sample = True
                image_values.append(image_bytes)

    if not conversation_values:
        raise ValueError(f"No valid samples found in {input_path}.")

    if has_multi_image_sample:
        normalized_images = [value if isinstance(value, list) else [value] for value in image_values]
        image_array = pa.array(normalized_images, type=pa.list_(pa.binary()))
    else:
        image_array = pa.array(image_values, type=pa.binary())

    table = pa.Table.from_arrays(
        [
            pa.array(conversation_values, type=pa.string()),
            image_array,
        ],
        names=["conversations", "image_bytes"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

    sample_kind = "multi-image" if has_multi_image_sample else "single-image"
    print(f"Converted {len(conversation_values)} samples -> {output_path} ({sample_kind})")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else input_path.parent.resolve()
    convert_jsonl_to_parquet(input_path, output_path, image_root)


if __name__ == "__main__":
    main()
