from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import warnings


class DefaultConfig(object):
    seed = 0
    # dataset options
    trainset_name = 'market'
    testset_name = 'duke'
    height = 256
    width = 128
    # sampler
    workers = 8
    num_instances = 4
    # default optimization params
    train_batch = 64
    test_batch = 64
    max_epoch = 60
    decay_epoch = 40
    # estimate bn statistics
    batch_num_bn_estimatation = 50
    # io
    print_freq = 50
    save_dir = './pytorch-ckpt/market'

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
