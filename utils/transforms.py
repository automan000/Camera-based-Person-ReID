from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torchvision import transforms as T


class TrainTransform(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x):
        x = T.Resize((self.h, self.w))(x)
        x = T.RandomHorizontalFlip()(x)
        x = T.Pad(10)(x)
        x = T.RandomCrop(size=(self.h, self.w))(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(x)
        return x


class TestTransform(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x=None):
        x = T.Resize((self.h, self.w))(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(x)
        return x
