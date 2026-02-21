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
        description="Create LSMD 512 tiles from train/val/test split txt files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root path containing region folders (anyang/jungnang/gangnam).",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Directory containing train.txt, val.txt, test.txt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for tiled dataset",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--image-bands",
        type=str,
        default="1,2,3",
        help="Comma-separated 1-based band indices for T2 image output (default: 1,2,3).",
    )
    parser.add_argument(
        "--compress",
        type=str,
        default="LZW",
        help="GTiff compression (default: LZW)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Optional limit for quick smoke test (0 = all).",
    )
    return parser.parse_args()


def find_years(name: str):
    years = []
    for m in re.finditer(r"(19|20)\d{2}", name):
        years.append(int(m.group(0)))
    return years


def choose_t1_t2_road_files(road_files):
    ranked = []
    for path in road_files:
        years = find_years(path.name)
        if years:
            ranked.append((min(years), path))
        else:
            ranked.append((9999, path))
    ranked.sort(key=lambda x: (x[0], x[1].name))
    t1 = ranked[0][1]
    t2 = ranked[-1][1]
    return t1, t2


def discover_region_paths(region_dir: Path):
    tif_files = sorted(region_dir.glob("*.tif"))
    road_files = [p for p in tif_files if "road_gt" in p.name.lower()]
    change_files = [p for p in tif_files if "change" in p.name.lower()]
    image_files = [
        p for p in tif_files
        if "road_gt" not in p.name.lower() and "change" not in p.name.lower()
    ]

    if len(road_files) != 2:
        raise ValueError(f"[{region_dir.name}] expected 2 road_gt files, found {len(road_files)}")
    if len(change_files) != 1:
        raise ValueError(f"[{region_dir.name}] expected 1 change file, found {len(change_files)}")
    if len(image_files) != 1:
        raise ValueError(f"[{region_dir.name}] expected 1 T2 image file, found {len(image_files)}")

    t1_road, t2_road = choose_t1_t2_road_files(road_files)
    return {
        "image_t2": image_files[0],
        "road_t1": t1_road,
        "road_t2": t2_road,
        "change": change_files[0],
    }


def check_alignment(paths):
    with rasterio.open(paths["image_t2"]) as d_img, rasterio.open(paths["road_t1"]) as d_t1, rasterio.open(paths["road_t2"]) as d_t2, rasterio.open(paths["change"]) as d_ch:
        ok = (
            d_img.height == d_t1.height == d_t2.height == d_ch.height
            and d_img.width == d_t1.width == d_t2.width == d_ch.width
            and d_img.transform == d_t1.transform == d_t2.transform == d_ch.transform
            and d_img.crs == d_t1.crs == d_t2.crs == d_ch.crs
        )
        if not ok:
            raise ValueError(
                "Grid mismatch (height/width/transform/crs). "
                f"image={d_img.height}x{d_img.width}, "
                f"t1={d_t1.height}x{d_t1.width}, "
                f"t2={d_t2.height}x{d_t2.width}, "
                f"change={d_ch.height}x{d_ch.width}"
            )
        return d_img.height, d_img.width


def read_split_ids(split_file: Path):
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    ids = []
    for raw in split_file.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if TILE_ID_RE.match(s) is None:
            raise ValueError(f"Invalid tile id format in {split_file}: {s}")
        ids.append(s)
    return ids


def build_profile(src, count, tile_size, transform, compress):
    profile = {
        "driver": "GTiff",
        "height": tile_size,
        "width": tile_size,
        "count": count,
        "crs": src.crs,
        "dtype": src.dtypes[0],
        "transform": transform,
        "tiled": False,
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
    }
    if src.nodata is not None:
        profile["nodata"] = src.nodata
    return profile


def write_tile(dst_path: Path, data, profile):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def main():
    args = parse_args()
    image_bands = [int(x.strip()) for x in args.image_bands.split(",") if x.strip()]
    if not image_bands:
        raise ValueError("--image-bands is empty")

    split_ids = {}
    for split in SPLIT_NAMES:
        split_ids[split] = read_split_ids(args.split_dir / f"{split}.txt")

    all_ids = []
    seen = set()
    for split in SPLIT_NAMES:
        for tid in split_ids[split]:
            if tid not in seen:
                seen.add(tid)
                all_ids.append(tid)
    if args.max_tiles > 0:
        all_ids = all_ids[:args.max_tiles]

    # Discover required regions from split ids only.
    regions = sorted({tid.split("/")[0] for tid in all_ids})
    region_info = {}
    for region in regions:
        region_dir = args.data_root / region
        if not region_dir.exists():
            raise FileNotFoundError(f"Region directory not found: {region_dir}")
        paths = discover_region_paths(region_dir)
        h, w = check_alignment(paths)
        region_info[region] = {"paths": paths, "height": h, "width": w}

    # Open all datasets once.
    readers = {}
    try:
        for region, info in region_info.items():
            paths = info["paths"]
            readers[region] = {
                "image_t2": rasterio.open(paths["image_t2"]),
                "road_t1": rasterio.open(paths["road_t1"]),
                "road_t2": rasterio.open(paths["road_t2"]),
                "change": rasterio.open(paths["change"]),
            }

        # Output root structure.
        # out_dir/
        #   images/t2/<region>/rXXX_cYYY.tif
        #   labels/t1/<region>/rXXX_cYYY.tif
        #   labels/t2/<region>/rXXX_cYYY.tif
        #   labels/change/<region>/rXXX_cYYY.tif
        #   splits/{train,val,test}.txt
        (args.out_dir / "images" / "t2").mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / "t1").mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / "t2").mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / "change").mkdir(parents=True, exist_ok=True)
        (args.out_dir / "splits").mkdir(parents=True, exist_ok=True)

        for split in SPLIT_NAMES:
            (args.out_dir / "splits" / f"{split}.txt").write_text(
                "\n".join(split_ids[split]) + "\n"
            )
        buffer_file = args.split_dir / "buffer_excluded.txt"
        if buffer_file.exists():
            (args.out_dir / "splits" / "buffer_excluded.txt").write_text(
                buffer_file.read_text()
            )

        n_done = 0
        for tid in all_ids:
            m = TILE_ID_RE.match(tid)
            region = m.group("region")
            row = int(m.group("row"))
            col = int(m.group("col"))

            ds_img = readers[region]["image_t2"]
            ds_t1 = readers[region]["road_t1"]
            ds_t2 = readers[region]["road_t2"]
            ds_ch = readers[region]["change"]

            y = row * args.tile_size
            x = col * args.tile_size
            if y + args.tile_size > ds_img.height or x + args.tile_size > ds_img.width:
                raise ValueError(f"Tile {tid} out of bounds for {region} ({ds_img.height}x{ds_img.width})")

            win = Window(x, y, args.tile_size, args.tile_size)
            win_transform = ds_img.window_transform(win)

            if max(image_bands) > ds_img.count:
                raise ValueError(
                    f"Tile {tid}: requested image band {max(image_bands)} "
                    f"but {region} T2 image has only {ds_img.count} bands"
                )
            img = ds_img.read(indexes=image_bands, window=win)
            t1 = ds_t1.read(1, window=win)
            t2 = ds_t2.read(1, window=win)
            ch = ds_ch.read(1, window=win)

            p_img = build_profile(ds_img, img.shape[0], args.tile_size, win_transform, args.compress)
            p_lbl = build_profile(ds_t1, 1, args.tile_size, win_transform, args.compress)

            tile_name = f"r{row:03d}_c{col:03d}.tif"
            write_tile(args.out_dir / "images" / "t2" / region / tile_name, img, p_img)
            write_tile(args.out_dir / "labels" / "t1" / region / tile_name, t1, p_lbl)
            write_tile(args.out_dir / "labels" / "t2" / region / tile_name, t2, p_lbl)
            write_tile(args.out_dir / "labels" / "change" / region / tile_name, ch, p_lbl)

            n_done += 1
            if n_done == 1 or n_done % 500 == 0 or n_done == len(all_ids):
                print(f"[tiles] {n_done}/{len(all_ids)}")
    finally:
        for reg in readers.values():
            for ds in reg.values():
                ds.close()

    print("Done.")
    print("Output:", args.out_dir)
    print("Splits copied to:", args.out_dir / "splits")


if __name__ == "__main__":
    main()
