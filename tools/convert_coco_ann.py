import argparse
import os

import mmcv

parser = argparse.ArgumentParser()
parser.add_argument('img_dir')
args = parser.parse_args()

ann = {'images': []}
imgs = sorted(os.listdir(args.img_dir))
for i, img in enumerate(imgs):
    print(f'processing [{i + 1}/{len(imgs)}]')
    img_info = dict()
    img = mmcv.imread(os.path.join(args.img_dir, img))
    img_info['filename'] = imgs[i]
    img_info['height'] = img.shape[0]
    img_info['width'] = img.shape[1]
    ann['images'].append(img_info)
    print(img_info)
mmcv.dump(ann, 'data/vehicle_val.json')
