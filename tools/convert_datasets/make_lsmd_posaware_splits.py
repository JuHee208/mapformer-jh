#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create spatial block+buffer splits, selecting validation blocks with high class1/2 presence."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trainval-regions", nargs="+", required=True)
    parser.add_argument("--test-regions", nargs="+", required=True)
    parser.add_argument("--change-name", type=str, default="road_change_label.tif")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--target-val-tiles", type=int, default=3200)
    parser.add_argument(
        "--min-val-tiles",
        type=int,
        default=2800,
        help="Minimum val tile budget (across trainval regions).",
    )
    parser.add_argument(
        "--target-val-pos12",
        type=int,
        default=120,
        help="Target number of class(1|2)-positive tiles in val (across trainval regions).",
    )
    parser.add_argument("--block-h", type=int, default=12)
    parser.add_argument("--block-w", type=int, default=12)
    parser.add_argument("--buffer-tiles", type=int, default=4)
    parser.add_argument("--stride-h", type=int, default=0, help="0 => block_h//2")
    parser.add_argument("--stride-w", type=int, default=0, help="0 => block_w//2")
    return parser.parse_args()


def tid(region: str, r: int, c: int) -> str:
    return f"{region}/r{r:03d}_c{c:03d}"


def build_pos_map(change_path: Path, tile_size: int):
    with rasterio.open(change_path) as ds:
        nh = ds.height // tile_size
        nw = ds.width // tile_size
        pos = np.zeros((nh, nw), dtype=np.uint8)
        for r in range(nh):
            for c in range(nw):
                win = Window(c * tile_size, r * tile_size, tile_size, tile_size)
                arr = ds.read(1, window=win)
                if np.any((arr == 1) | (arr == 2)):
                    pos[r, c] = 1
    return pos


def integral_image(a: np.ndarray):
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant")


def rect_sum(ii: np.ndarray, r0: int, r1: int, c0: int, c1: int):
    return int(ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0])


def choose_blocks_for_region(
    pos_map,
    min_tiles,
    target_tiles,
    target_pos,
    block_h,
    block_w,
    buffer_tiles,
    stride_h,
    stride_w,
):
    nh, nw = pos_map.shape
    step_h = stride_h if stride_h > 0 else max(1, block_h // 2)
    step_w = stride_w if stride_w > 0 else max(1, block_w // 2)

    ii = integral_image(pos_map.astype(np.int32))
    cand = []
    for r0 in range(0, nh - block_h + 1, step_h):
        for c0 in range(0, nw - block_w + 1, step_w):
            r1 = r0 + block_h
            c1 = c0 + block_w
            p = rect_sum(ii, r0, r1, c0, c1)
            # prioritize positive count first, then top-left deterministic tie break
            cand.append((int(p), r0, r1, c0, c1))
    cand_hi = sorted(cand, key=lambda x: (x[0], -x[1], -x[3]), reverse=True)
    cand_lo = sorted(cand, key=lambda x: (x[0], x[1], x[3]))

    occupied = np.zeros((nh, nw), dtype=bool)
    val = np.zeros((nh, nw), dtype=bool)
    blocks = []
    picked_tiles = 0
    picked_pos = 0

    def try_pick(candidates, need_pos_first=False, need_min_tiles=False):
        nonlocal picked_tiles, picked_pos
        for p, r0, r1, c0, c1 in candidates:
            if picked_tiles >= target_tiles:
                break
            if need_pos_first and picked_pos >= target_pos:
                break
            if need_min_tiles and picked_tiles >= min_tiles:
                break
            if occupied[r0:r1, c0:c1].any():
                continue

            val[r0:r1, c0:c1] = True
            picked_tiles += (r1 - r0) * (c1 - c0)
            picked_pos += p
            blocks.append([r0, r1, c0, c1, int(p)])

            rr0 = max(0, r0 - buffer_tiles)
            rr1 = min(nh, r1 + buffer_tiles)
            cc0 = max(0, c0 - buffer_tiles)
            cc1 = min(nw, c1 + buffer_tiles)
            occupied[rr0:rr1, cc0:cc1] = True

    # Phase 1: secure positive tiles with high-positive blocks.
    if target_pos > 0:
        try_pick(cand_hi, need_pos_first=True, need_min_tiles=False)

    # Phase 2: reach minimum val size with low-positive blocks to avoid over-draining positives from train.
    if picked_tiles < min_tiles:
        try_pick(cand_lo, need_pos_first=False, need_min_tiles=True)

    # Phase 3: if min tiles met but pos target still missed, top up positives (bounded by target_tiles).
    if picked_pos < target_pos and picked_tiles < target_tiles:
        try_pick(cand_hi, need_pos_first=True, need_min_tiles=False)

    # Phase 4: optional fill to target tiles using low-positive blocks.
    if picked_tiles < target_tiles:
        try_pick(cand_lo, need_pos_first=False, need_min_tiles=False)

    # keep compatibility with old stopping logic
    if picked_tiles > target_tiles:
        picked_tiles = target_tiles

    # buffer = dilated(val) - val
    if buffer_tiles > 0:
        padded = np.pad(val, buffer_tiles, mode="constant")
        dil = np.zeros_like(val, dtype=bool)
        for dr in range(-buffer_tiles, buffer_tiles + 1):
            for dc in range(-buffer_tiles, buffer_tiles + 1):
                dil |= padded[
                    buffer_tiles + dr: buffer_tiles + dr + nh,
                    buffer_tiles + dc: buffer_tiles + dc + nw,
                ]
        buf = dil & (~val)
    else:
        buf = np.zeros_like(val, dtype=bool)

    train = ~(val | buf)
    return train, val, buf, blocks, picked_tiles, picked_pos


def count_pos_tiles_in_mask(pos_map, mask):
    return int((pos_map.astype(bool) & mask).sum())


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = []
    val_ids = []
    test_ids = []
    buffer_ids = []

    region_pos = {}
    region_shapes = {}
    total_trainval_tiles = 0
    for region in args.trainval_regions:
        p = args.data_root / region / args.change_name
        if not p.exists():
            raise FileNotFoundError(f"Missing change file: {p}")
        pos = build_pos_map(p, args.tile_size)
        region_pos[region] = pos
        region_shapes[region] = pos.shape
        total_trainval_tiles += int(pos.size)

    summary = {
        "tile_size": args.tile_size,
        "min_val_tiles_total": args.min_val_tiles,
        "target_val_tiles_total": args.target_val_tiles,
        "target_val_pos12_total": args.target_val_pos12,
        "block_h": args.block_h,
        "block_w": args.block_w,
        "buffer_tiles": args.buffer_tiles,
        "regions": {},
    }

    # train/val from pos-aware blocks
    total_pos12 = int(sum(region_pos[r].sum() for r in args.trainval_regions))

    for region in args.trainval_regions:
        pos = region_pos[region]
        nh, nw = pos.shape
        region_total = int(pos.size)
        region_pos12_total = int(pos.sum())
        target_region = int(round(args.target_val_tiles * (region_total / total_trainval_tiles)))
        target_region = max(args.block_h * args.block_w, target_region)
        min_region = int(round(args.min_val_tiles * (region_total / total_trainval_tiles)))
        min_region = max(args.block_h * args.block_w, min_region)
        min_region = min(min_region, target_region)
        if total_pos12 > 0:
            target_region_pos = int(round(args.target_val_pos12 * (region_pos12_total / total_pos12)))
        else:
            target_region_pos = 0
        target_region_pos = max(1, target_region_pos) if region_pos12_total > 0 else 0

        train_m, val_m, buf_m, blocks, val_tiles, val_pos = choose_blocks_for_region(
            pos,
            min_tiles=min_region,
            target_tiles=target_region,
            target_pos=target_region_pos,
            block_h=args.block_h,
            block_w=args.block_w,
            buffer_tiles=args.buffer_tiles,
            stride_h=args.stride_h,
            stride_w=args.stride_w,
        )

        for r in range(nh):
            for c in range(nw):
                t = tid(region, r, c)
                if val_m[r, c]:
                    val_ids.append(t)
                elif buf_m[r, c]:
                    buffer_ids.append(t)
                else:
                    train_ids.append(t)

        summary["regions"][region] = {
            "grid_h": nh,
            "grid_w": nw,
            "tiles_total": region_total,
            "target_val_tiles": target_region,
            "min_val_tiles": int(min_region),
            "tiles_train": int(train_m.sum()),
            "tiles_val": int(val_m.sum()),
            "tiles_buffer": int(buf_m.sum()),
            "pos12_tiles_total": int(pos.sum()),
            "target_val_pos12": int(target_region_pos),
            "pos12_tiles_train": count_pos_tiles_in_mask(pos, train_m),
            "pos12_tiles_val": count_pos_tiles_in_mask(pos, val_m),
            "val_blocks": blocks,
            "val_tiles_selected": int(val_tiles),
            "val_pos_selected": int(val_pos),
        }

    # test from full test regions
    for region in args.test_regions:
        p = args.data_root / region / args.change_name
        if not p.exists():
            raise FileNotFoundError(f"Missing change file: {p}")
        with rasterio.open(p) as ds:
            nh = ds.height // args.tile_size
            nw = ds.width // args.tile_size
        for r in range(nh):
            for c in range(nw):
                test_ids.append(tid(region, r, c))
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

    print("Saved:", args.out_dir)
    print("train:", len(train_ids))
    print("val:", len(val_ids))
    print("test:", len(test_ids))
    print("buffer_excluded:", len(buffer_ids))


if __name__ == "__main__":
    main()
