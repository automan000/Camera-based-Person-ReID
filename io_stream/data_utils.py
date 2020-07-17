import random
from collections import defaultdict
import numpy as np


def reorganize_images_by_camera(data, sample_per_camera):
    cams = np.unique([x[2] for x in data])
    images_per_cam = defaultdict(list)
    images_per_cam_sampled = defaultdict(list)
    for cam_id in cams:
        all_file_info = [x for x in data if x[2] == cam_id]
        all_file_info = sorted(all_file_info, key=lambda x: x[0])
        random.shuffle(all_file_info)
        images_per_cam[cam_id] = all_file_info
        images_per_cam_sampled[cam_id] = all_file_info[:sample_per_camera]

    return images_per_cam, images_per_cam_sampled
