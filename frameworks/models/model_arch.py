from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from getpass import getuser

from torch import nn
import torch.nn.functional as F

from frameworks.models.backbone import ResNet_Backbone
from frameworks.models.weight_utils import weights_init_kaiming


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_pids=None, last_stride=1):
        super().__init__()
        self.num_pids = num_pids
        self.base = ResNet_Backbone(last_stride)
        model_path = '/home/' + getuser() + '/.torch/models/resnet50-19c8e357.pth'
        self.base.load_param(model_path)
        bn_neck = nn.BatchNorm1d(2048, momentum=None)
        bn_neck.bias.requires_grad_(False)
        self.bottleneck = nn.Sequential(bn_neck)
        self.bottleneck.apply(weights_init_kaiming)
        if self.num_pids is not None:
            self.classifier = nn.Linear(2048, self.num_pids, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat_before_bn = self.base(x)
        feat_before_bn = F.avg_pool2d(feat_before_bn, feat_before_bn.shape[2:])
        feat_before_bn = feat_before_bn.view(feat_before_bn.shape[0], -1)
        feat_after_bn = self.bottleneck(feat_before_bn)
        if self.num_pids is not None:
            classification_results = self.classifier(feat_after_bn)
            return feat_after_bn, classification_results
        else:
            return feat_after_bn

    def get_optim_policy(self):
        base_param_group = filter(lambda p: p.requires_grad, self.base.parameters())
        add_param_group = filter(lambda p: p.requires_grad, self.bottleneck.parameters())
        cls_param_group = filter(lambda p: p.requires_grad, self.classifier.parameters())

        all_param_groups = []
        all_param_groups.append({'params': base_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': add_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': cls_param_group, "weight_decay": 0.0005})
        return all_param_groups
