import glob
import os
import sys
import numpy as np

if __name__ == '__main__':
    dirname_pattern = 'run'
    base_dir = '/home/vobecant/datasets/pix2pixhd/crops'
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if
            dirname_pattern in d and os.path.isdir(os.path.join(base_dir, d))]
    print('Found directories: {}'.format(dirs))

    all_files = []

    for d in dirs:
        files_dir = []
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    if len(files_dir) == 0:
                        print(file_path)
                    files_dir.append(file_path)
        all_files.extend(files_dir)
        print('Found {} images in directory {}'.format(len(files_dir), d))

    np.save('/home/vobecant/datasets/YBB/generated/pix2pixhd_paths.npy', all_files)
