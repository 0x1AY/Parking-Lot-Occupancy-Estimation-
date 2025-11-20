#!/usr/bin/env python3
"""Convert YOLO segmentation labels into bounding boxes for detection training.

Usage:
    python tools/convert_to_bboxes.py \
        --source "../Dataset-V1" \
        --dest   "../Dataset-V1-detect"

The script keeps a copy of the folder structure (train/valid/test) at the
destination, symlinks the `images` directories to avoid duplicating data, and
rewrites every `.txt` file inside each `labels` directory into YOLO box format.
Existing bounding boxes (four numbers after the class id) pass through
unchanged, while polygons (>=6 coordinates) are squashed into their enclosing
rectangle. The resulting dataset is ready for YOLO detection tasks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Original dataset root that contains train/valid/test splits.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Dataset-V1-detect",
        help="Destination root for the detection-ready dataset.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Optional path to data.yaml (defaults to <source>/data.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting destination label files if they already exist.",
    )
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polygon_to_box(coords: Sequence[float]) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    x_min = clamp01(min(xs))
    y_min = clamp01(min(ys))
    x_max = clamp01(max(xs))
    y_max = clamp01(max(ys))
    width = clamp01(x_max - x_min)
    height = clamp01(y_max - y_min)
    if width == 0 or height == 0:
        # Small epsilon to avoid degenerate boxes during training.
        width = max(width, 1e-6)
        height = max(height, 1e-6)
    x_center = clamp01(x_min + width / 2.0)
    y_center = clamp01(y_min + height / 2.0)
    return x_center, y_center, width, height


def parse_row(row: str) -> Tuple[int, float, float, float, float]:
    parts = row.strip().split()
    if not parts:
        raise ValueError("Empty row")
    class_id = int(float(parts[0]))
    coords = [float(value) for value in parts[1:]]
    if len(coords) == 4:
        x_center, y_center, width, height = coords
        return x_center, y_center, width, height
    if len(coords) >= 6 and len(coords) % 2 == 0:
        return polygon_to_box(coords)
    raise ValueError(f"Unsupported coordinate count ({len(coords)}) in row: {row}")


def format_row(class_id: int, bbox: Tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = bbox
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_file(src: Path, dst: Path, overwrite: bool = False) -> Tuple[int, int]:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")

    converted = []
    skipped = 0
    for row in src.read_text().splitlines():
        row = row.strip()
        if not row:
            continue
        try:
            class_id = int(float(row.split()[0]))
            bbox = parse_row(row)
            converted.append(format_row(class_id, bbox))
        except ValueError as exc:
            skipped += 1
            print(f"[WARN] {src.name}: {exc}")

    dst.write_text("\n".join(converted) + ("\n" if converted else ""))
    return len(converted), skipped


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def load_names(data_yaml: Path) -> List[str]:
    text = data_yaml.read_text()
    start = text.find("names:")
    if start == -1:
        return []
    names_part = text[start + len("names:") :].strip()
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        names = parsed.get("names", []) if isinstance(parsed, dict) else []
        if isinstance(names, dict):
            return [str(name) for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
        if isinstance(names, list):
            return [str(name) for name in names]
    except Exception:
        pass

    import ast

    try:
        literal = ast.literal_eval(names_part.splitlines()[0])
        return [str(name) for name in literal]
    except Exception:
        return []


def write_data_yaml(dest_root: Path, names: Sequence[str]) -> None:
    data_yaml = dest_root / "data.yaml"
    names_repr = "[" + ", ".join(f"'{name}'" for name in names) + "]"
    content = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(names)}",
            f"names: {names_repr}",
        ]
    )
    data_yaml.write_text(content + "\n")


def main() -> None:
    args = parse_args()
    src_root = args.source.resolve()
    dest_root = args.dest.resolve()
    data_yaml_path = args.data_file or (src_root / "data.yaml")

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    dest_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_rows = 0
    total_skipped = 0
    for split in ("train", "valid", "test"):
        src_images = src_root / split / "images"
        src_labels = src_root / split / "labels"
        if not src_images.exists() or not src_labels.exists():
            print(f"[WARN] Missing split '{split}' in {src_root}")
            continue

        dest_images = dest_root / split / "images"
        dest_labels = dest_root / split / "labels"
        ensure_symlink(src_images, dest_images)
        dest_labels.mkdir(parents=True, exist_ok=True)

        for label_file in sorted(src_labels.glob("*.txt")):
            dst_file = dest_labels / label_file.name
            converted_rows, skipped = convert_file(label_file, dst_file, overwrite=args.overwrite)
            total_files += 1
            total_rows += converted_rows
            total_skipped += skipped

    names = load_names(data_yaml_path) if data_yaml_path.exists() else []
    if names:
        write_data_yaml(dest_root, names)

    print(
        f"Finished: {total_files} label files converted ({total_rows} rows, {total_skipped} skipped)."
    )
    print(f"Detection-ready dataset: {dest_root}")


if __name__ == "__main__":
    main()
