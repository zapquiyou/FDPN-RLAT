from glob import glob
import os.path as osp
import random
import os
import shutil

root='/data/StereoDatasets/sceneflow/'
target1_root = '/data/StereoDatasets/sceneflow_rltrain/'
target2_root = '/data/StereoDatasets/sceneflow_rltest/'
dstype='frames_finalpass'

left_images = sorted( glob(osp.join(root, dstype, 'flying', '*/*/*/left/*.png')) + glob(osp.join(root, dstype, 'monkaa', '*/left/*.png')) + glob(osp.join(root, dstype, 'driving', '*/*/*/left/*.png')) )
print(f"Total left images: {len(left_images)}")

random_left_images = random.sample(left_images, 15000)
random_right_images = [img.replace('left', 'right') for img in random_left_images]
random_disparities_images = [ im.replace(dstype, 'disparity').replace('.png', '.pfm') for im in random_left_images ]
for img in random_left_images:
    relative_path = osp.relpath(img, root)
    target_path = osp.join(target1_root, relative_path)
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    shutil.move(img, target_path)

for img in random_right_images:
    relative_path = osp.relpath(img, root)
    target_path = osp.join(target1_root, relative_path)
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    shutil.move(img, target_path)

for img in random_disparities_images:
    relative_path = osp.relpath(img, root)
    target_path = osp.join(target1_root, relative_path)
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    shutil.move(img, target_path)

left_images = sorted( glob(osp.join(root, dstype, 'flying', '*/*/*/left/*.png')) + glob(osp.join(root, dstype, 'monkaa', '*/left/*.png')) + glob(osp.join(root, dstype, 'driving', '*/*/*/left/*.png')) )
random_left_images = random.sample(left_images, 5000)
random_right_images = [img.replace('left', 'right') for img in random_left_images]
random_disparities_images = [ im.replace(dstype, 'disparity').replace('.png', '.pfm') for im in random_left_images ]
for img in random_left_images:
    relative_path = osp.relpath(img, root)
    target_path = osp.join(target2_root, relative_path)
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    shutil.move(img, target_path)

for img in random_right_images:
    relative_path = osp.relpath(img, root)
    target_path = osp.join(target2_root, relative_path)
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    shutil.move(img, target_path)

for img in random_disparities_images:
    relative_path = osp.relpath(img, root)
    target_path = osp.join(target2_root, relative_path)
    os.makedirs(osp.dirname(target_path), exist_ok=True)
    shutil.move(img, target_path)