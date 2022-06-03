'''
Removing broken images in the BFM dataset (synface)
'''
from PIL import Image
import os
import os.path as path

TRAIN_IMAGE_PATH = '/root/unsup3d-rep/data/synface/train/image'
TRAIN_DEPTH_PATH = '/root/unsup3d-rep/data/synface/train/depth'
TEST_IMAGE_PATH = '/root/unsup3d-rep/data/synface/test/image'
TEST_DEPTH_PATH = '/root/unsup3d-rep/data/synface/test/depth'
VAL_IMAGE_PATH = '/root/unsup3d-rep/data/synface/val/image'
VAL_DEPTH_PATH = '/root/unsup3d-rep/data/synface/val/depth'

for image_path, depth_path in zip([VAL_IMAGE_PATH], [VAL_DEPTH_PATH]):
    img_list = [name for name in os.listdir(image_path) if path.isfile(path.join(image_path, name))]
    depth_list = [name for name in os.listdir(depth_path) if path.isfile(path.join(depth_path, name))]
    img_list.sort()
    depth_list.sort()

    for depth in depth_list:
        try:
            d = Image.open(path.join(depth_path, depth))
        except:
            print(f'remove: {path.join(depth_path, depth)}')
            os.remove(path.join(depth_path, depth))

        if depth.replace('depth', 'image') not in img_list:
            print('not in image!')
            os.remove(path.join(depth_path, depth))
            continue

        try:
            d.verify()
        except:
            # print(f'broken file: {depth}')
            os.remove(path.join(depth_path, depth))

    for img in img_list:
        try:
            im = Image.open(path.join(image_path, img))
        except:
            print(f'remove: {path.join(image_path, img)}')
            os.remove(path.join(image_path, img))

        if img.replace('image', 'depth') not in depth_list:
            print('not in depth!')
            os.remove(path.join(image_path, img))
            continue

        try:
            im.verify()
        except:
            print(f'broken file: {img}')
            os.remove(path.join(image_path, img))