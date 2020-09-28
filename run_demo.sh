#!/usr/bin/env bash
# train
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python train_model.py train --trainset_name market --save_dir='market_cent_4'
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python test_model.py test --testset_name market --save_dir='market_cent_4'
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python test_model.py test --testset_name duke --save_dir='market_demo'
