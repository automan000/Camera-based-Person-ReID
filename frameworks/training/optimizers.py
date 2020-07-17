from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def get_optimizer_strategy(opt, optim_policy=None):
    optimizer = torch.optim.SGD(
        optim_policy, lr=1e-2, weight_decay=5e-4, momentum=0.9
    )

    def adjust_lr(optimizer, ep):
        if ep < opt.decay_epoch:
            lr = 1e-2
        else:
            lr = 1e-3
        for i, p in enumerate(optimizer.param_groups):
            p['lr'] = lr
        return lr

    return optimizer, adjust_lr
