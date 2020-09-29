from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import random
import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import opt
from io_stream import data_manager

from frameworks.models import ResNetBuilder
from frameworks.evaluating import evaluator_manager

from utils.serialization import Logger, load_previous_model
from utils.transforms import TestTransform


def test(**kwargs):
    opt._parse(kwargs)
    sys.stdout = Logger(
        os.path.join("./pytorch-ckpt/current", opt.save_dir, 'log_test_{}.txt'.format(opt.testset_name)))

    torch.backends.cudnn.deterministic = True  # I think this line may slow down the testing process
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    print('initializing dataset {}'.format(opt.testset_name))
    dataset = data_manager.init_dataset(name=opt.testset_name,
                                        num_bn_sample=opt.batch_num_bn_estimatation * opt.test_batch)

    pin_memory = True if use_gpu else False

    print('loading model from {} ...'.format(opt.save_dir))
    model = ResNetBuilder()
    model_path = os.path.join("./pytorch-ckpt/current", opt.save_dir,
                              'model_best.pth.tar')
    model = load_previous_model(model, model_path, load_fc_layers=False)
    model.eval()

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    reid_evaluator = evaluator_manager.init_evaluator(opt.testset_name, model, flip=True)

    def _calculate_bn_and_features(all_data, sampled_data):
        time.sleep(1)
        all_features, all_ids, all_cams = [], [], []
        available_cams = list(sampled_data)

        for current_cam in tqdm.tqdm(available_cams):
            camera_samples = sampled_data[current_cam]
            data_for_camera_loader = DataLoader(
                data_manager.init_datafolder(opt.testset_name, camera_samples, TestTransform(opt.height, opt.width)),
                batch_size=opt.test_batch, num_workers=opt.workers,
                pin_memory=False, drop_last=True,
            )
            reid_evaluator.collect_sim_bn_info(data_for_camera_loader)

            camera_data = all_data[current_cam]
            data_loader = DataLoader(
                data_manager.init_datafolder(opt.testset_name, camera_data, TestTransform(opt.height, opt.width)),
                batch_size=opt.test_batch, num_workers=opt.workers,
                pin_memory=pin_memory, shuffle=False,
            )
            fs, pids, camids = reid_evaluator.produce_features(data_loader, normalize=True)
            all_features.append(fs)
            all_ids.append(pids)
            all_cams.append(camids)

        all_features = torch.cat(all_features, 0)
        all_ids = np.concatenate(all_ids, axis=0)
        all_cams = np.concatenate(all_cams, axis=0)
        time.sleep(1)
        return all_features, all_ids, all_cams

    print('Processing query features...')
    qf, q_pids, q_camids = _calculate_bn_and_features(dataset.query_per_cam, dataset.query_per_cam_sampled)
    print('Processing gallery features...')
    gf, g_pids, g_camids = _calculate_bn_and_features(dataset.gallery_per_cam,
                                                      dataset.gallery_per_cam_sampled)
    print('Computing CMC and mAP...')
    reid_evaluator.get_final_results_with_features(qf, q_pids, q_camids, gf, g_pids, g_camids)


if __name__ == '__main__':
    import fire

    fire.Fire()
