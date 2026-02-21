#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create spatial block + buffer train/val/test splits on tile grid."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--trainval-regions", nargs="+", required=True,
        help="Regions used to build train/val splits."
    )
    parser.add_argument(
        "--test-regions", nargs="+", required=True,
        help="Regions used only for test split."
    )
    parser.add_argument("--change-name", default="road_change_label.tif")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--block-h", type=int, default=20, help="Block height (tiles).")
    parser.add_argument("--block-w", type=int, default=20, help="Block width (tiles).")
    parser.add_argument("--buffer-tiles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            out |= padded[
                radius + dr: radius + dr + h,
                radius + dc: radius + dc + w,
            ]
    return out


def sample_val_blocks(nh, nw, target_tiles, block_h, block_w, buffer_tiles, rng):
    step_h = max(1, block_h // 2)
    step_w = max(1, block_w // 2)
    candidates = []
    for r in range(0, max(1, nh - block_h + 1), step_h):
        for c in range(0, max(1, nw - block_w + 1), step_w):
            if r + block_h <= nh and c + block_w <= nw:
                candidates.append((r, c))
    rng.shuffle(candidates)

    val = np.zeros((nh, nw), dtype=bool)
    occupied = np.zeros((nh, nw), dtype=bool)
    blocks = []
    val_count = 0

    for r, c in candidates:
        if val_count >= target_tiles:
            break
        if occupied[r:r + block_h, c:c + block_w].any():
            continue

        val[r:r + block_h, c:c + block_w] = True
        blocks.append((r, r + block_h, c, c + block_w))
        val_count += block_h * block_w

        rr0 = max(0, r - buffer_tiles)
        rr1 = min(nh, r + block_h + buffer_tiles)
        cc0 = max(0, c - buffer_tiles)
        cc1 = min(nw, c + block_w + buffer_tiles)
        occupied[rr0:rr1, cc0:cc1] = True

    return val, blocks


def tile_id(region, r, c):
    return f"{region}/r{r:03d}_c{c:03d}"


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids, buffer_ids = [], [], [], []
    summary = {
        "tile_size": args.tile_size,
        "val_ratio": args.val_ratio,
        "block_h": args.block_h,
        "block_w": args.block_w,
        "buffer_tiles": args.buffer_tiles,
        "regions": {},
    }

    for region in args.trainval_regions:
        change_path = args.data_root / region / args.change_name
        with rasterio.open(change_path) as ds:
            nh = ds.height // args.tile_size
            nw = ds.width // args.tile_size

        total = nh * nw
        target = int(round(total * args.val_ratio))
        val_mask, blocks = sample_val_blocks(
            nh, nw, target, args.block_h, args.block_w, args.buffer_tiles, rng
        )
        buf_mask = dilate(val_mask, args.buffer_tiles) & (~val_mask)
        train_mask = ~(val_mask | buf_mask)

        for r in range(nh):
            for c in range(nw):
                tid = tile_id(region, r, c)
                if val_mask[r, c]:
                    val_ids.append(tid)
                elif buf_mask[r, c]:
                    buffer_ids.append(tid)
                elif train_mask[r, c]:
                    train_ids.append(tid)

        summary["regions"][region] = {
            "grid_h": nh,
            "grid_w": nw,
            "tiles_total": total,
            "tiles_train": int(train_mask.sum()),
            "tiles_val": int(val_mask.sum()),
            "tiles_buffer": int(buf_mask.sum()),
            "val_blocks": blocks,
        }

    for region in args.test_regions:
        change_path = args.data_root / region / args.change_name
        with rasterio.open(change_path) as ds:
            nh = ds.height // args.tile_size
            nw = ds.width // args.tile_size
        for r in range(nh):
            for c in range(nw):
                test_ids.append(tile_id(region, r, c))
        summary["regions"][region] = {
            "grid_h": nh,
            "grid_w": nw,
            "tiles_total": nh * nw,
            "tiles_test": nh * nw,
        }

    (args.out_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (args.out_dir / "val.txt").write_text("\n".join(val_ids) + "\n")
    (args.out_dir / "test.txt").write_text("\n".join(test_ids) + "\n")
    (args.out_dir / "buffer_excluded.txt").write_text("\n".join(buffer_ids) + "\n")

    summary["counts"] = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "buffer_excluded": len(buffer_ids),
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print("Saved to:", args.out_dir)
    print("train:", len(train_ids))
    print("val:", len(val_ids))
    print("test:", len(test_ids))
    print("buffer_excluded:", len(buffer_ids))


if __name__ == "__main__":
    main()
