import argparse
import json
from pathlib import Path


try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is required to inspect parquet files. Run with `.venv/bin/python` or install pyarrow first."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a parquet dataset used by MiniMind training pipelines."
    )
    parser.add_argument("parquet_path", help="Path to the parquet file.")
    parser.add_argument("--rows", type=int, default=3, help="How many samples to print. Default: 3")
    parser.add_argument(
        "--truncate",
        type=int,
        default=160,
        help="Max characters to print for long string fields. Default: 160",
    )
    return parser.parse_args()


def shorten_text(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_scalar(value, truncate):
    if value is None:
        return "None"
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, str):
        return shorten_text(value, truncate)
    return repr(value)


def preview_conversations(raw_value, truncate):
    try:
        conversations = json.loads(raw_value)
    except Exception as exc:
        print(f"    [conversations] invalid JSON: {exc}")
        print(f"    raw: {shorten_text(raw_value, truncate)}")
        return

    if not isinstance(conversations, list):
        print(f"    [conversations] parsed type={type(conversations).__name__}, expected list")
        return

    print(f"    turns: {len(conversations)}")
    for idx, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            print(f"    turn[{idx}]: invalid type={type(turn).__name__}")
            continue
        role = turn.get("role", "<missing>")
        content = turn.get("content", "<missing>")
        print(f"    turn[{idx}] role={role}: {shorten_text(content, truncate)}")


def preview_binary_list(name, value):
    if isinstance(value, list):
        lengths = [len(item) if isinstance(item, (bytes, bytearray)) else None for item in value]
        print(f"    [{name}] list len={len(value)}, item_byte_lengths={lengths}")
    elif isinstance(value, (bytes, bytearray)):
        print(f"    [{name}] bytes len={len(value)}")
    else:
        print(f"    [{name}] unexpected type={type(value).__name__}")


def preview_path_list(name, value):
    if isinstance(value, list):
        print(f"    [{name}] list len={len(value)}")
        for idx, item in enumerate(value[:3]):
            print(f"      - {idx}: {item}")
    else:
        print(f"    [{name}] {value}")


def preview_generic_row(row, truncate):
    for key, value in row.items():
        if key == "conversations":
            print(f"  - {key}:")
            preview_conversations(value, truncate)
        elif key in {"image_bytes", "audio_bytes"}:
            print(f"  - {key}:")
            preview_binary_list(key, value)
        elif key in {"audio_path", "image_path", "wav_path"}:
            print(f"  - {key}:")
            preview_path_list(key, value)
        else:
            print(f"  - {key}: {describe_scalar(value, truncate)}")


def inspect_parquet(path: Path, rows: int, truncate: int):
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read()

    print(f"[File] {path}")
    print(f"[Rows] {table.num_rows}")
    print(f"[Columns] {table.num_columns}")
    print("[Schema]")
    for name, field in zip(table.column_names, table.schema):
        print(f"  - {name}: {field.type}")

    print("[Format Checks]")
    has_conversations = "conversations" in table.column_names
    print(f"  - conversations column: {'OK' if has_conversations else 'MISSING'}")

    modality_cols = [
        name for name in table.column_names
        if name in {"image_bytes", "audio_bytes", "audio_path", "image_path", "wav_path"}
    ]
    if modality_cols:
        print(f"  - modality columns: {', '.join(modality_cols)}")
    else:
        print("  - modality columns: none detected")

    if table.num_rows == 0:
        print("[Samples] parquet is empty")
        return

    print("[Samples]")
    sample_count = min(rows, table.num_rows)
    columns = table.column_names
    for idx in range(sample_count):
        row = {name: table[name][idx].as_py() for name in columns}
        print(f"\n=== Sample {idx} ===")
        preview_generic_row(row, truncate)


def main():
    args = parse_args()
    path = Path(args.parquet_path).resolve()
    if not path.exists():
        raise SystemExit(f"Parquet file not found: {path}")
    inspect_parquet(path, rows=args.rows, truncate=args.truncate)


if __name__ == "__main__":
    main()
