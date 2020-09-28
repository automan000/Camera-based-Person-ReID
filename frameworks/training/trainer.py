from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import torch
from frameworks.training.base import BaseTrainer
from utils.meters import AverageMeter
import time


class CameraClsTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer):
        super().__init__(opt, model, optimizer, criterion, summary_writer)

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        self.data = imgs.cuda()
        self.pids = pids.cuda()
        self.camids = camids.cuda()

    def _ogranize_data(self):
        unique_camids = torch.unique(self.camids).cpu().numpy()
        reorg_data = []
        reorg_pids = []
        for current_camid in unique_camids:
            current_camid = (self.camids == current_camid).nonzero().view(-1)
            if current_camid.size(0) > 1:
                data = torch.index_select(self.data, index=current_camid, dim=0)
                pids = torch.index_select(self.pids, index=current_camid, dim=0)
                reorg_data.append(data)
                reorg_pids.append(pids)

        # Sort the list for our modified data-parallel
        # This process helps to increase efficiency when utilizing multiple GPUs
        # However, our experiments show that this process slightly decreases the final performance
        # You can enable the following process if you prefer
        # sort_index = [x.size(0) for x in reorg_pids]
        # sort_index = [i[0] for i in sorted(enumerate(sort_index), key=lambda x: x[1], reverse=True)]
        # reorg_data = [reorg_data[i] for i in sort_index]
        # reorg_pids = [reorg_pids[i] for i in sort_index]
        # ===== The end of the sort process ==== #
        self.data = reorg_data
        self.pids = reorg_pids

    def _forward(self, data):
        feat, id_scores = self.model(data)
        return feat, id_scores

    def _backward(self):
        self.loss.backward()

    def train(self, epoch, data_loader):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        for i, inputs in enumerate(data_loader):
            self._parse_data(inputs)
            self._ogranize_data()

            torch.cuda.synchronize()
            tic = time.time()

            feat, id_scores = self._forward(self.data)
            pids = torch.cat(self.pids, dim=0)
            self.loss = self.criterion(feat, id_scores, pids, self.global_step,
                                       self.summary_writer)
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            torch.cuda.synchronize()
            batch_time.update(time.time() - tic)
            losses.update(self.loss.item())
            # tensorboard
            self.global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), self.global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.mean, batch_time.val,
                              losses.mean, losses.val))

        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
