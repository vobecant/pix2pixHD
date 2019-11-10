import glob
import os
import sys
import numpy as np
import pickle
import json

if __name__ == '__main__':
    dirname_pattern = 'run'
    base_dir = '/home/vobecant/datasets/pix2pixhd/crops'
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if
            dirname_pattern in d and os.path.isdir(os.path.join(base_dir, d))]
    print('Found directories: {}'.format(dirs))

    all_files = []
    all_files_by_vid = {}

    for d in dirs:
        files_dir = []
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    vid_name = root.split(os.sep)[-1]
                    if vid_name not in all_files_by_vid.keys():
                        all_files_by_vid[vid_name] = []
                    if len(files_dir) == 0:
                        print('sample found file:', file_path)
                    files_dir.append(file_path)
                    all_files_by_vid[vid_name].append(file_path)
        all_files.extend(files_dir)
        print('Found {} images in directory {}'.format(len(files_dir), d))

    np.save('/home/vobecant/datasets/YBB/generated/pix2pixhd_paths.npy', all_files)
    with open('/home/vobecant/datasets/YBB/generated/pix2pixhd_paths_by_vid.npy', 'wb') as f:
        pickle.dump(all_files_by_vid, f)

    all_files = np.load('/home/vobecant/datasets/YBB/generated/pix2pixhd_paths.npy')
    with open('/home/vobecant/datasets/YBB/generated/pix2pixhd_paths_by_vid.npy', 'rb') as f:
        all_files_by_vid = pickle.load(f)
    print('Length of all files: {}'.format(len(all_files)))
    registered_videos = all_files_by_vid.keys()
    print('Registered videos: {}'.format(registered_videos))

    # cluster the data
    path = '/home/vobecant/datasets/YBB/GAN_data/splits/2019-11-05_unique_vid/clust_to_pers_map_class.json'
    clustered = {}

    with open(path, 'r') as infile:
        clust_to_pers_map = json.load(infile)

    # clust_to_pers_map_class['clusters'] = [] - here are lists with video names, frame numbers and person IDs, num_of_samples for every cluster
    # clust_to_pers_map_class['target_distribution'] = Base64Encode(target_distribution) - can be decoded and used if needed
    # clust_to_pers_map_class['trn_distribution'] = Base64Encode(trn_distribution) - can be decoded and used if needed
    # clust_to_pers_map_class['tst_distribution'] = Base64Encode(tst_distribution) - can be decoded and used if needed
    # clust_to_pers_map['clusters'] consists of following lists {'video_name': [], 'frame_num': [], 'person_id': [], 'num_samples': 0}

    for i in range(len(clust_to_pers_map['clusters'])):
        print('----------------------------------------------------------------------------------------------------')
        print('Cluster {}/{}, size: {}'.format(i, len(clust_to_pers_map['clusters']),
                                               len(clust_to_pers_map['clusters'][i]['person_id'])))

        clustered[i] = []

        for j in range(len(clust_to_pers_map['clusters'][i]['person_id'])):
            vid_name = clust_to_pers_map['clusters'][i]['video_name'][j]
            # print(vid_name)
            splitted = vid_name.split(os.sep)
            # print('splitted: {}'.format(splitted))
            vid_name = vid_name.split(os.sep)[-2]
            # print(vid_name)
            # print('Video name: ' + vid_name)
            frame_num = int(clust_to_pers_map['clusters'][i]['frame_num'][j])
            frame_num5 = format(frame_num, '05')
            # print('Frame number: ' + str(frame_num))
            person_id = clust_to_pers_map['clusters'][i]['person_id'][j]
            # print('Person id: ' + str(person_id))

            pattern = '{}/{}_{}'.format(vid_name, frame_num5, person_id)

            if vid_name in registered_videos:
                imgs = [img for img in all_files_by_vid[vid_name] if pattern in img]
                clustered[i].extend(imgs)

        print('Found {} images belonging to cluster {}'.format(len(clustered[i]), i))

    with open('/home/vobecant/datasets/YBB/generated/pix2pixhd_clusters_train_class.npy', 'wb') as f:
        pickle.dump(clustered, f)
