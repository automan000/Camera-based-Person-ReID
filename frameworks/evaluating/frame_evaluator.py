from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from frameworks.evaluating.base import BaseEvaluator


class FrameEvaluator(BaseEvaluator):
    def __init__(self, model, flip=True):
        super().__init__(model)
        self.loop = 2 if flip else 1
        self.evaluator_prints = []

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def flip_tensor_lr(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        if isinstance(feature, tuple) or isinstance(feature, list):
            output = []
            for x in feature:
                if isinstance(x, tuple) or isinstance(x, list):
                    output.append([item.cpu() for item in x])
                else:
                    output.append(x.cpu())
            return output
        else:
            return feature.cpu()

    def produce_features(self, dataloader, normalize=True):
        self.model.eval()
        all_feature_norm = []
        qf, q_pids, q_camids = [], [], []
        for batch_idx, inputs in enumerate(dataloader):
            inputs, pids, camids = self._parse_data(inputs)
            feature = None
            for i in range(self.loop):
                if i == 1:
                    inputs = self.flip_tensor_lr(inputs)
                global_f = self._forward(inputs)

                if feature is None:
                    feature = global_f
                else:
                    feature += global_f
            if normalize:
                fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
                all_feature_norm.extend(list(fnorm.cpu().numpy()[:, 0]))
                feature = feature.div(fnorm.expand_as(feature))
            else:
                feature = feature / 2
            qf.append(feature)
            q_pids.extend(pids)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        return qf, q_pids, q_camids

    def get_final_results_with_features(self, qf, q_pids, q_camids, gf, g_pids, g_camids, target_ranks=[1, 5, 10, 20]):
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
        cmc, mAP, ranks = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        self.evaluator_prints.append("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        self.evaluator_prints.append("CMC curve")
        for r in target_ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
            self.evaluator_prints.append("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        return cmc[0]

    def collect_sim_bn_info(self, dataloader):
        network_bns = [x for x in list(self.model.modules()) if
                       isinstance(x, torch.nn.BatchNorm2d) or isinstance(x, torch.nn.BatchNorm1d)]
        for bn in network_bns:
            bn.running_mean = torch.zeros(bn.running_mean.size()).float().cuda()
            bn.running_var = torch.ones(bn.running_var.size()).float().cuda()
            bn.num_batches_tracked = torch.tensor(0).cuda().long()

        self.model.train()
        for batch_idx, inputs in enumerate(dataloader):
            # each camera should has at least 2 images for estimating BN statistics
            assert len(inputs[0].size()) == 4 and inputs[0].size(
                0) > 1, 'Cannot estimate BN statistics. Each camera should have at least 2 images'
            inputs, pids, camids = self._parse_data(inputs)
            for i in range(self.loop):
                if i == 1:
                    inputs = self.flip_tensor_lr(inputs)
                self._forward(inputs)
        self.model.eval()
