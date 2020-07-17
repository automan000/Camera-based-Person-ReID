from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import os
import sys

import os.path as osp
import torch


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, save_dir):
    mkdir_if_missing(save_dir)
    fpath = osp.join(save_dir, 'model_best.pth.tar')
    torch.save(state, fpath)


def load_previous_model(model, file_path=None, load_fc_layers=True):
    assert file_path is not None, 'Must define the path of the saved model'
    ckpt = torch.load(file_path)
    if load_fc_layers:
        state_dict = ckpt['state_dict']
    else:
        state_dict = dict()
        for k, v in ckpt['state_dict'].items():
            if 'classifer' not in k:
                state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    return model
