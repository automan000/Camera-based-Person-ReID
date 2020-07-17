from __future__ import print_function, absolute_import

from PIL import Image
from torch.utils.data import Dataset

from io_stream.datasets.market import Market1501
from io_stream.datasets.msmt import MSMT17
from io_stream.datasets.duke import Duke


class ReID_Data(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img_path, pid, camid = self.dataset[item]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)


"""Create datasets"""

__data_factory = {
    'market': Market1501,
    'duke': Duke,
    'msmt': MSMT17,
}

__folder_factory = {
    'market': ReID_Data,
    'duke': ReID_Data,
    'msmt': ReID_Data,
}


def  init_dataset(name, *args, **kwargs):
    if name not in __data_factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __data_factory[name](*args, **kwargs)


def init_datafolder(name, data_list, transforms):
    if name not in __folder_factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __folder_factory[name](data_list, transforms)
