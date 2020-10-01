from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import random
import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import opt

from io_stream import data_manager, NormalCollateFn, IdentitySampler

from frameworks.models import ResNetBuilder
from frameworks.training import CameraClsTrainer, get_model_optimizer_strategy, CamDataParallel

from utils.centerloss import CenterLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TrainTransform


from torch.nn.parallel import DistributedDataParallel

def train(**kwargs):
    opt._parse(kwargs)
    # set random seed and cudnn benchmark
    torch.backends.cudnn.deterministic = True  # I think this line may slow down the training process
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(os.path.join('./pytorch-ckpt/current', opt.save_dir, 'log_train.txt'))

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

    # center loss
    cent = CenterLoss(num_classes=train_dataset.num_train_pids, feat_dim=2048, use_gpu=True).cuda()
    # cent_param_group = filter(lambda p: p.requires_grad, xent.parameters())
    # optim_policy.append({'params': cent_param_group, "weight_decay": 0.0005})
    # xent = nn.CrossEntropyLoss()

    def standard_cls_criterion(preditions,
                               logits,
                               targets,
                               global_step,
                               summary_writer):
        identity_loss = cent(preditions, targets) * opt.center_loss_alpha
        # identity_accuracy = torch.mean((torch.argmax(preditions, dim=1) == targets).float())
        summary_writer.add_scalar('cls_loss', identity_loss.item(), global_step)
        # summary_writer.add_scalar('cls_accuracy', identity_accuracy.item(), global_step)
        return identity_loss

    # get trainer and evaluator
    optimizer, adjust_lr = get_model_optimizer_strategy(opt, optim_policy)
    optimizer_cent = torch.optim.SGD(cent.parameters(), lr=opt.center_base_lr)

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


if __name__ == '__main__':
    import fire
    fire.Fire()
