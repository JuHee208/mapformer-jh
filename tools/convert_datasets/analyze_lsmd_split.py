#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


SPLIT_NAMES = ("train", "val", "test")
TILE_ID_RE = re.compile(r"^(?P<region>[^/]+)/r(?P<row>\d+)_c(?P<col>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze LSMD split quality using change labels (class 0/1/2/3)."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--change-name", type=str, default="road_change_label.tif")
    return parser.parse_args()


def read_split_ids(split_file: Path):
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    ids = []
    for line in split_file.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        m = TILE_ID_RE.match(s)
        if m is None:
            raise ValueError(f"Invalid tile id: {s}")
        ids.append((m.group("region"), int(m.group("row")), int(m.group("col"))))
    return ids


def main():
    args = parse_args()

    # Open only regions used by split files.
    regions = set()
    split_ids = {}
    for split in SPLIT_NAMES:
        ids = read_split_ids(args.split_dir / f"{split}.txt")
        split_ids[split] = ids
        for region, _, _ in ids:
            regions.add(region)

    ds_map = {}
    for region in sorted(regions):
        p = args.data_root / region / args.change_name
        if not p.exists():
            raise FileNotFoundError(f"Missing change raster: {p}")
        ds_map[region] = rasterio.open(p)

    try:
        for split in SPLIT_NAMES:
            ids = split_ids[split]
            cls = np.zeros(4, dtype=np.int64)
            pos12_tiles = 0
            for region, row, col in ids:
                ds = ds_map[region]
                win = Window(col * args.tile_size, row * args.tile_size, args.tile_size, args.tile_size)
                arr = ds.read(1, window=win)
                b = np.bincount(arr.ravel(), minlength=4)[:4]
                cls += b
                if b[1] + b[2] > 0:
                    pos12_tiles += 1

            total_px = int(cls.sum())
            print(f"[{split}]")
            print(f"  tiles: {len(ids)}")
            print(f"  pos12_tiles: {pos12_tiles} ({(pos12_tiles / len(ids) * 100.0) if ids else 0.0:.3f}%)")
            print(
                "  class_pct(0/1/2/3): "
                f"{(cls[0] / total_px * 100.0) if total_px else 0.0:.4f}, "
                f"{(cls[1] / total_px * 100.0) if total_px else 0.0:.4f}, "
                f"{(cls[2] / total_px * 100.0) if total_px else 0.0:.4f}, "
                f"{(cls[3] / total_px * 100.0) if total_px else 0.0:.4f}"
            )
            print(f"  class12_pct: {((cls[1] + cls[2]) / total_px * 100.0) if total_px else 0.0:.4f}%")
    finally:
        for ds in ds_map.values():
            ds.close()


if __name__ == "__main__":
    main()
