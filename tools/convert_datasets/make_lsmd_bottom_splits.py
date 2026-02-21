#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import rasterio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create LSMD splits by holding out bottom rows for validation."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--trainval-regions", nargs="+", required=True,
        help="Regions used for train/val (e.g., anyang jungnang)",
    )
    parser.add_argument(
        "--test-regions", nargs="+", required=True,
        help="Regions used for test only (e.g., gangnam)",
    )
    parser.add_argument("--change-name", type=str, default="road_change_label.tif")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Bottom-row ratio per trainval region (0~1)."
    )
    parser.add_argument(
        "--buffer-rows", type=int, default=4,
        help="Rows excluded between train and val per trainval region."
    )
    return parser.parse_args()


def tid(region: str, r: int, c: int) -> str:
    return f"{region}/r{r:03d}_c{c:03d}"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = []
    val_ids = []
    test_ids = []
    buffer_ids = []
    summary = {
        "tile_size": args.tile_size,
        "val_ratio": args.val_ratio,
        "buffer_rows": args.buffer_rows,
        "regions": {},
    }

    # Train/Val regions: bottom rows as val
    for region in args.trainval_regions:
        ch_path = args.data_root / region / args.change_name
        if not ch_path.exists():
            raise FileNotFoundError(f"Missing change raster: {ch_path}")

        with rasterio.open(ch_path) as ds:
            nh = ds.height // args.tile_size
            nw = ds.width // args.tile_size

        val_rows = max(1, int(round(nh * args.val_ratio)))
        if val_rows + args.buffer_rows >= nh:
            raise ValueError(
                f"[{region}] val_rows({val_rows}) + buffer_rows({args.buffer_rows}) >= total_rows({nh})"
            )

        val_r0 = nh - val_rows
        val_r1 = nh
        buf_r0 = val_r0 - args.buffer_rows
        buf_r1 = val_r0

        for r in range(nh):
            for c in range(nw):
                t = tid(region, r, c)
                if val_r0 <= r < val_r1:
                    val_ids.append(t)
                elif buf_r0 <= r < buf_r1:
                    buffer_ids.append(t)
                else:
                    train_ids.append(t)

        summary["regions"][region] = {
            "grid_h": nh,
            "grid_w": nw,
            "tiles_total": nh * nw,
            "val_rows": val_rows,
            "val_row_range": [val_r0, val_r1 - 1],
            "buffer_row_range": [buf_r0, buf_r1 - 1],
            "tiles_train": (buf_r0 * nw),
            "tiles_val": (val_rows * nw),
            "tiles_buffer": (args.buffer_rows * nw),
        }

    # Test regions: all rows/cols
    for region in args.test_regions:
        ch_path = args.data_root / region / args.change_name
        if not ch_path.exists():
            raise FileNotFoundError(f"Missing change raster: {ch_path}")
        with rasterio.open(ch_path) as ds:
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
