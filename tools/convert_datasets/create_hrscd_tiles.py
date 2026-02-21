import os
import os.path as osp
from PIL import Image
import numpy as np
import argparse

Image.MAX_IMAGE_PIXELS = 10_000_000_000

parser = argparse.ArgumentParser(description='Create compressed tiles.')
parser.add_argument('--data_dir', type=str, help='Data dir')
parser.add_argument('--out_dir', type=str, help='Out dir')
parser.add_argument('--tile_size', type=int, help='Size of the final tiles')
parser.add_argument('--img_compression', type=str, default='jpeg')
parser.add_argument('--label_compression', type=str, default='tiff_lzw')
parser.add_argument('--skip_images', action='store_true', help='Skip image tiling')
parser.add_argument('--skip_labels', action='store_true', help='Skip semantic label tiling')
parser.add_argument('--only_change', action='store_true', help='Process only change labels')
parser.add_argument('--years', type=str, default='2006,2012',
                    help='Comma-separated years to process (e.g., "2012")')
args = parser.parse_args()
years = [y.strip() for y in args.years.split(',') if y.strip()]
if args.only_change:
    args.skip_images = True
    args.skip_labels = True

if not args.skip_images:
    print('Images...')
    img_total = 0
    for year in years:
        for d in ['D14', 'D35']:
            img_total += len(os.listdir(osp.join(args.data_dir, 'images', year, d)))
    img_done = 0
    for year in years:
        for d in ['D14', 'D35']:
            for file in os.listdir(osp.join(args.data_dir, 'images', year, d)):
                if not file.lower().endswith(('.tif', '.tiff')):
                    continue
                img_done += 1
                if img_done == 1 or img_done % 50 == 0 or img_done == img_total:
                    print(f'  [images] {img_done}/{img_total} files')
                name, suffix = file.rsplit('.', 1)
                os.makedirs(osp.join(args.out_dir, 'images', year, d, name), exist_ok=True)
                img = np.array(Image.open(osp.join(args.data_dir, 'images', year, d, file)))
                for i in range(img.shape[0] // args.tile_size):
                    for j in range(img.shape[1] // args.tile_size):
                        img_tile = img[args.tile_size * i:args.tile_size * (i+1),
                            args.tile_size * j:args.tile_size * (j+1)]
                        
                        img_tile_path = osp.join(args.out_dir, 'images', year, d, name, f'{i}_{j}.'+suffix)
                        Image.fromarray(img_tile).save(img_tile_path, compression=args.img_compression)

if not args.skip_labels:
    print('Semantic labels...')
    sem_total = 0
    for year in years:
        for d in ['D14', 'D35']:
            sem_total += len(os.listdir(osp.join(args.data_dir, 'labels', year, d)))
    sem_done = 0
    for year in years:
        for d in ['D14', 'D35']:
            for file in os.listdir(osp.join(args.data_dir, 'labels', year, d)):
                if not file.lower().endswith(('.tif', '.tiff')):
                    continue
                sem_done += 1
                if sem_done == 1 or sem_done % 50 == 0 or sem_done == sem_total:
                    print(f'  [labels] {sem_done}/{sem_total} files')
                name, suffix = file.rsplit('.', 1)
                os.makedirs(osp.join(args.out_dir, 'labels', year, d, name), exist_ok=True)
                sem = np.array(Image.open(osp.join(args.data_dir, 'labels', year, d, file)))
                for i in range(sem.shape[0] // args.tile_size):
                    for j in range(sem.shape[1] // args.tile_size):
                        sem_tile = sem[args.tile_size * i:args.tile_size * (i+1),
                            args.tile_size * j:args.tile_size * (j+1)]
                        
                        sem_tile_path = osp.join(args.out_dir, 'labels', year, d, name, f'{i}_{j}.'+suffix)
                        Image.fromarray(sem_tile).save(sem_tile_path, compression=args.label_compression)

print('Binary change labels...')
change_root = osp.join(args.data_dir, 'labels', 'change')
if not osp.exists(change_root):
    change_root = osp.join(args.data_dir, 'change')
bc_total = 0
for d in ['D14', 'D35']:
    bc_total += len(os.listdir(osp.join(change_root, d)))
bc_done = 0
for d in ['D14', 'D35']:
    for file in os.listdir(osp.join(change_root, d)):
        if not file.lower().endswith(('.tif', '.tiff')):
            continue
        bc_done += 1
        if bc_done == 1 or bc_done % 50 == 0 or bc_done == bc_total:
            print(f'  [change] {bc_done}/{bc_total} files')
        name, suffix = file.rsplit('.', 1)
        os.makedirs(osp.join(args.out_dir, 'labels', 'change', d, name), exist_ok=True)
        bc = np.array(Image.open(osp.join(change_root, d, file)))
        for i in range(bc.shape[0] // args.tile_size):
            for j in range(bc.shape[1] // args.tile_size):
                bc_tile = bc[args.tile_size * i:args.tile_size * (i+1),
                    args.tile_size * j:args.tile_size * (j+1)]
                
                bc_tile_path = osp.join(args.out_dir, 'labels', 'change', d, name, f'{i}_{j}.'+suffix)
                Image.fromarray(bc_tile).save(bc_tile_path, compression=args.label_compression)
