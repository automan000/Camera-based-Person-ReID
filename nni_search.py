from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import numpy as np
import tqdm
import time
import argparse
import logging

import nni
from nni.utils import merge_parameter

from config import opt

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from io_stream import data_manager, NormalCollateFn, IdentitySampler

from frameworks.models import ResNetBuilder
from frameworks.training import CameraClsTrainer, get_model_optimizer_strategy, get_center_optimizer_strategy, CamDataParallel
from frameworks.evaluating import evaluator_manager

from utils.centerloss import CenterLoss
from utils.serialization import save_checkpoint, load_previous_model
from utils.transforms import TrainTransform, TestTransform

logger = logging.getLogger('centerloss_AutoML')


def train(kwargs):
    opt._parse(kwargs)
    # set random seed and cudnn benchmark
    torch.backends.cudnn.deterministic = True  # I think this line may slow down the training process
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.trainset_name))
    train_dataset = data_manager.init_dataset(name=opt.trainset_name,
                                              num_bn_sample=opt.batch_num_bn_estimatation * opt.test_batch)
    pin_memory = True if use_gpu else False
    summary_writer = SummaryWriter(os.path.join('./pytorch-ckpt/current', opt.save_dir, 'tensorboard_log'))
    trainloader = DataLoader(
        data_manager.init_datafolder(opt.trainset_name, train_dataset.train, TrainTransform(opt.height, opt.width)),
        sampler=IdentitySampler(train_dataset.train, opt.train_batch, opt.num_instances),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True, collate_fn=NormalCollateFn(),
        worker_init_fn=np.random.seed(opt.seed)
    )
    print('initializing model ...')
    model = ResNetBuilder(train_dataset.num_train_pids)
    optim_policy = model.get_optim_policy()
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))
    if use_gpu:
        model = CamDataParallel(model).cuda()
    cent = CenterLoss(num_classes=train_dataset.num_train_pids, feat_dim=2048, use_gpu=True)

    def standard_cls_criterion(preditions,
                               logits,
                               targets,
                               global_step,
                               summary_writer):
        identity_loss = cent(preditions, targets) * opt.center_loss_alpha
        summary_writer.add_scalar('cls_loss', identity_loss.item(), global_step)
        return identity_loss

    # get trainer and evaluator
    optimizer, adjust_lr = get_model_optimizer_strategy(opt, optim_policy)
    optimizer_cent, adjust_center_lr = get_center_optimizer_strategy(opt, cent.parameters())

    reid_trainer = CameraClsTrainer(opt, model, optimizer, optimizer_cent, standard_cls_criterion, summary_writer)
    print('Start training')
    for epoch in range(opt.max_epoch):
        adjust_lr(optimizer, epoch)
        reid_trainer.train(epoch, trainloader)
    if use_gpu:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    save_checkpoint({
        'state_dict': state_dict,
        'epoch': epoch + 1,
    }, save_dir=os.path.join('./pytorch-ckpt/current', opt.save_dir))


def test(kwargs):
    opt._parse(kwargs)
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
    rank1 = reid_evaluator.get_final_results_with_features(qf, q_pids, q_camids, gf, g_pids, g_camids)
    return rank1


def main_search(tuner_params):
    train(tuner_params)
    test_rank1 = test(tuner_params)
    nni.report_final_result(test_rank1)
    logger.debug('Final result is %g', test_rank1)
    logger.debug('Send final result done.')


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        print(tuner_params)
        logger.debug(tuner_params)
        main_search(tuner_params)
    except Exception as exception:
        logger.exception(exception)
        raise
