#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw


SPLITS = ("train", "val", "test", "buffer_excluded")
TILE_ID_RE = re.compile(r"^(?P<region>[^/]+)/r(?P<row>\d+)_c(?P<col>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize spatial split coverage on LSMD regions."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--legend-height", type=int, default=90)
    return parser.parse_args()


def read_ids(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        m = TILE_ID_RE.match(s)
        if not m:
            continue
        out.append((m.group("region"), int(m.group("row")), int(m.group("col"))))
    return out


def find_image_path(region_dir: Path):
    tifs = sorted(region_dir.glob("*.tif"))
    imgs = [p for p in tifs if "road_gt" not in p.name.lower() and "change" not in p.name.lower()]
    if len(imgs) != 1:
        raise ValueError(f"[{region_dir.name}] expected 1 image tif, found {len(imgs)}")
    return imgs[0]


def stretch_rgb(img):
    # img: H,W,3 uint8/uint16
    out = np.zeros_like(img, dtype=np.float32)
    for b in range(3):
        band = img[..., b].astype(np.float32)
        lo, hi = np.percentile(band, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        band = np.clip((band - lo) / (hi - lo), 0, 1)
        out[..., b] = band
    return out


def build_tile_code_map(region, nh, nw, split_ids):
    # codes: 0=none, 1=train, 2=val, 3=test, 4=buffer
    code = np.zeros((nh, nw), dtype=np.uint8)
    order = [("train", 1), ("val", 2), ("test", 3), ("buffer_excluded", 4)]
    for split_name, split_code in order:
        for rgn, r, c in split_ids.get(split_name, []):
            if rgn != region:
                continue
            if 0 <= r < nh and 0 <= c < nw:
                code[r, c] = split_code
    return code


def make_quicklook(ds, max_side):
    h, w = ds.height, ds.width
    scale = min(1.0, max_side / max(h, w))
    oh = max(1, int(round(h * scale)))
    ow = max(1, int(round(w * scale)))

    # Always read 3 bands for display; if single-band image exists, replicate.
    if ds.count >= 3:
        arr = ds.read(indexes=[1, 2, 3], out_shape=(3, oh, ow), resampling=rasterio.enums.Resampling.bilinear)
        rgb = np.transpose(arr, (1, 2, 0))
    else:
        arr = ds.read(1, out_shape=(oh, ow), resampling=rasterio.enums.Resampling.bilinear)
        rgb = np.stack([arr, arr, arr], axis=-1)

    rgb = stretch_rgb(rgb)
    return rgb, oh, ow


def downsample_tile_codes(tile_codes, src_h, src_w, out_h, out_w, tile_size):
    nh, nw = tile_codes.shape
    y = ((np.arange(out_h) + 0.5) * (src_h / out_h)).astype(np.float32)
    x = ((np.arange(out_w) + 0.5) * (src_w / out_w)).astype(np.float32)
    rr = np.floor(y / tile_size).astype(np.int32)[:, None]
    cc = np.floor(x / tile_size).astype(np.int32)[None, :]

    valid = (rr >= 0) & (rr < nh) & (cc >= 0) & (cc < nw)
    rr_clip = np.clip(rr, 0, max(0, nh - 1))
    cc_clip = np.clip(cc, 0, max(0, nw - 1))
    out = tile_codes[rr_clip, cc_clip]
    out = np.where(valid, out, 0).astype(np.uint8)
    return out


def add_legend(img_rgb: np.ndarray, code_to_rgb: np.ndarray, legend_height: int):
    h, w, _ = img_rgb.shape
    canvas = np.full((h + legend_height, w, 3), 255, dtype=np.uint8)
    canvas[legend_height:, :, :] = img_rgb
    labels = [
        (1, "train"),
        (2, "val"),
        (3, "test"),
        (4, "buffer_excluded"),
    ]
    x = 20
    y = 18
    box_w = 28
    box_h = 20
    gap = 14
    text_gap = 8
    legend_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(legend_img)
    for code, label in labels:
        color = tuple(code_to_rgb[code].tolist())
        draw.rectangle([x, y, x + box_w, y + box_h], fill=color, outline=(0, 0, 0))
        draw.text((x + box_w + text_gap, y + 2), label, fill=(0, 0, 0))
        x += box_w + text_gap + len(label) * 9 + gap

    title = "Split overlay"
    draw.text((20, y + box_h + 12), title, fill=(0, 0, 0))
    return np.array(legend_img)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_ids = {name: read_ids(args.split_dir / f"{name}.txt") for name in SPLITS}
    regions = sorted({r for ids in split_ids.values() for r, _, _ in ids})

    cmap = np.array(
        [
            [0, 0, 0, 0],          # none / outside floor area
            [30, 180, 30, 255],    # train
            [255, 180, 30, 255],   # val
            [60, 120, 255, 255],   # test
            [160, 160, 160, 255],  # buffer
        ],
        dtype=np.uint8,
    )

    for region in regions:
        img_path = find_image_path(args.data_root / region)
        with rasterio.open(img_path) as ds:
            h, w = ds.height, ds.width
            nh, nw = h // args.tile_size, w // args.tile_size
            tile_codes = build_tile_code_map(region, nh, nw, split_ids)
            rgb, oh, ow = make_quicklook(ds, args.max_side)
            code_quick = downsample_tile_codes(tile_codes, h, w, oh, ow, args.tile_size)

        overlay = cmap[code_quick]
        overlay_rgb = overlay[..., :3].astype(np.float32) / 255.0
        overlay_a = (overlay[..., 3].astype(np.float32) / 255.0) * args.alpha

        base = rgb.copy()
        base = base * (1 - overlay_a[..., None]) + overlay_rgb * overlay_a[..., None]
        base_u8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)
        base_u8 = add_legend(base_u8, cmap[:, :3], args.legend_height)

        out_path = args.out_dir / f"{region}_split_overlay.png"
        Image.fromarray(base_u8).save(out_path)

        # Also save pure grid map
        grid_out = args.out_dir / f"{region}_split_grid.png"
        grid_rgb = cmap[code_quick][:, :, :3]
        grid_rgb = add_legend(grid_rgb, cmap[:, :3], args.legend_height)
        Image.fromarray(grid_rgb).save(grid_out)

        print(f"[saved] {out_path}")
        print(f"[saved] {grid_out}")


if __name__ == "__main__":
    main()
