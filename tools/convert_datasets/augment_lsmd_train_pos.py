#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import rasterio


MODALITY_DIRS = (
    "images/t2",
    "labels/t1",
    "labels/t2",
    "labels/change",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Augment only positive(change class 1/2) train tiles and create a new train split."
        )
    )
    parser.add_argument("--tiles-root", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Directory that contains train.txt (default: <tiles-root>/splits)",
    )
    parser.add_argument(
        "--train-split-name",
        type=str,
        default="train.txt",
    )
    parser.add_argument(
        "--out-train-split-name",
        type=str,
        default="train_aug_pos.txt",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="hflip,vflip,rot90",
        help="Comma separated transforms: hflip,vflip,rot90,rot180,rot270",
    )
    parser.add_argument(
        "--pos-classes",
        type=str,
        default="1,2",
        help="Comma separated class ids to treat as positive in change label",
    )
    parser.add_argument("--compress", type=str, default="LZW")
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Do not create augmented tif files; only oversample positive ids in split.",
    )
    parser.add_argument(
        "--repeat-pos",
        type=int,
        default=None,
        help="Used with --split-only. Repeat each positive tile this many extra times. "
        "Default: len(--transforms).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_split_line(line: str):
    s = line.strip()
    if not s:
        return None, None
    parts = s.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid split line: {line}")
    return parts[0], parts[1]


def apply_transform(arr: np.ndarray, name: str):
    # arr shape: (C, H, W)
    if name == "hflip":
        return arr[:, :, ::-1]
    if name == "vflip":
        return arr[:, ::-1, :]
    if name == "rot90":
        return np.rot90(arr, 1, axes=(1, 2))
    if name == "rot180":
        return np.rot90(arr, 2, axes=(1, 2))
    if name == "rot270":
        return np.rot90(arr, 3, axes=(1, 2))
    raise ValueError(f"Unsupported transform: {name}")


def is_positive(change_path: Path, pos_classes):
    with rasterio.open(change_path) as ds:
        a = ds.read(1)
    return np.isin(a, pos_classes).any()


def save_tif(src_path: Path, dst_path: Path, arr: np.ndarray, compress: str):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        compress=compress,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(arr)


def main():
    args = parse_args()
    split_dir = args.split_dir or (args.tiles_root / "splits")
    train_path = split_dir / args.train_split_name
    out_train_path = split_dir / args.out_train_split_name

    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    pos_classes = [int(x.strip()) for x in args.pos_classes.split(",") if x.strip()]

    if not train_path.exists():
        raise FileNotFoundError(f"train split not found: {train_path}")
    if not transforms:
        raise ValueError("No transforms provided")

    train_lines = [ln.strip() for ln in train_path.read_text().splitlines() if ln.strip()]
    positives = []
    for line in train_lines:
        region, tile_id = parse_split_line(line)
        change_path = args.tiles_root / "labels/change" / region / f"{tile_id}.tif"
        if not change_path.exists():
            raise FileNotFoundError(f"Missing change label: {change_path}")
        if is_positive(change_path, pos_classes):
            positives.append((region, tile_id))

    print(f"[info] train tiles: {len(train_lines)}")
    print(f"[info] positive tiles (classes {pos_classes}): {len(positives)}")
    print(f"[info] transforms: {transforms}")
    if args.split_only:
        repeat_pos = args.repeat_pos if args.repeat_pos is not None else len(transforms)
        if repeat_pos <= 0:
            raise ValueError("--repeat-pos must be >= 1")
        print(f"[info] split-only mode: repeat positive tiles x{repeat_pos}")
    else:
        repeat_pos = len(transforms)

    aug_lines = []
    write_jobs = 0
    for region, tile_id in positives:
        if args.split_only:
            for _ in range(repeat_pos):
                aug_lines.append(f"{region}/{tile_id}")
            continue

        for tf in transforms:
            aug_id = f"{tile_id}_aug_{tf}"
            aug_lines.append(f"{region}/{aug_id}")
            for rel in MODALITY_DIRS:
                src = args.tiles_root / rel / region / f"{tile_id}.tif"
                dst = args.tiles_root / rel / region / f"{aug_id}.tif"
                if dst.exists() and not args.overwrite:
                    continue
                write_jobs += 1
                if args.dry_run:
                    continue
                with rasterio.open(src) as ds:
                    arr = ds.read()
                arr_tf = apply_transform(arr, tf)
                save_tif(src, dst, arr_tf, args.compress)

    out_lines = train_lines + aug_lines
    if not args.dry_run:
        out_train_path.write_text("\n".join(out_lines) + "\n")

    print(f"[info] new augmented lines: {len(aug_lines)}")
    print(f"[info] total train lines (new split): {len(out_lines)}")
    print(f"[info] write jobs: {write_jobs}")
    if args.dry_run:
        print("[done] dry-run only (no files written)")
    else:
        print(f"[done] wrote split: {out_train_path}")


if __name__ == "__main__":
    main()
