#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproject LSMD region rasters to EPSG:5186 with per-region aligned grid."
    )
    parser.add_argument("--src-root", type=Path, required=True, help="Source root (e.g., data/lsmd)")
    parser.add_argument("--dst-root", type=Path, required=True, help="Destination root (e.g., data/lsmd_5186)")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["anyang", "jungnang", "gangnam"],
        help="Region folders to process",
    )
    parser.add_argument(
        "--target-epsg", type=int, default=5186, help="Target EPSG code (default: 5186)"
    )
    parser.add_argument(
        "--compress", type=str, default="LZW", help="GTiff compression for output (default: LZW)"
    )
    parser.add_argument(
        "--copy-splits",
        action="store_true",
        help="Copy split directories under src-root into dst-root.",
    )
    return parser.parse_args()


def pick_region_files(region_dir: Path):
    tif_files = sorted(region_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No tif files in {region_dir}")
    image = [p for p in tif_files if "road_gt" not in p.name.lower() and "change" not in p.name.lower()]
    roads = [p for p in tif_files if "road_gt" in p.name.lower()]
    changes = [p for p in tif_files if "change" in p.name.lower()]
    if len(image) != 1:
        raise ValueError(f"[{region_dir.name}] expected 1 image file, found {len(image)}")
    if len(roads) != 2:
        raise ValueError(f"[{region_dir.name}] expected 2 road_gt files, found {len(roads)}")
    if len(changes) != 1:
        raise ValueError(f"[{region_dir.name}] expected 1 change file, found {len(changes)}")
    return image[0], roads[0], roads[1], changes[0]


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def reproject_to_grid(
    src_path: Path,
    dst_path: Path,
    dst_crs: CRS,
    dst_transform,
    dst_width: int,
    dst_height: int,
    compress: str,
    resampling: Resampling,
):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            compress=compress,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    num_threads=2,
                )


def main():
    args = parse_args()
    dst_crs = CRS.from_epsg(args.target_epsg)
    args.dst_root.mkdir(parents=True, exist_ok=True)

    for region in args.regions:
        src_region = args.src_root / region
        dst_region = args.dst_root / region
        if not src_region.exists():
            raise FileNotFoundError(f"Missing region dir: {src_region}")

        image, road1, road2, change = pick_region_files(src_region)
        files = [image, road1, road2, change]

        with rasterio.open(image) as src_img:
            src_res_x = abs(src_img.transform.a)
            src_res_y = abs(src_img.transform.e)
            same_crs = src_img.crs == dst_crs

            if same_crs:
                # Preserve original grid for this region.
                dst_transform = src_img.transform
                dst_width = src_img.width
                dst_height = src_img.height
            else:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_img.crs,
                    dst_crs,
                    src_img.width,
                    src_img.height,
                    *src_img.bounds,
                    resolution=(src_res_x, src_res_y),
                )

        print(
            f"[{region}] src_crs={src_img.crs} -> dst_crs={dst_crs}, "
            f"dst_size={dst_height}x{dst_width}"
        )

        for f in files:
            src_path = src_region / f.name
            dst_path = dst_region / f.name
            is_label = ("road_gt" in f.name.lower()) or ("change" in f.name.lower())
            if same_crs:
                copy_file(src_path, dst_path)
                print(f"  copied: {f.name}")
            else:
                resampling = Resampling.nearest if is_label else Resampling.bilinear
                reproject_to_grid(
                    src_path=src_path,
                    dst_path=dst_path,
                    dst_crs=dst_crs,
                    dst_transform=dst_transform,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    compress=args.compress,
                    resampling=resampling,
                )
                print(f"  reprojected: {f.name}")

    if args.copy_splits:
        for p in sorted(args.src_root.glob("splits*")):
            dst = args.dst_root / p.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(p, dst)
            print(f"copied split dir: {p.name}")

    print("Done.")
    print("Output root:", args.dst_root)


if __name__ == "__main__":
    main()
