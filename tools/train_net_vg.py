# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import logging
import os
import torch
import random   
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


# try to solve too many opened file Error
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_printoptions(precision=12)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def random_init(seed=0):
    """ Set the seed for random sampling of pytorch related random packages
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

random_init(0)

cfg.merge_from_file("../configs/lcmcg_flickr_bottom_up_10_100_1x.yaml")

def main():
    #! test the dataloader
    data_loader = make_data_loader(
    cfg,
    is_train=True,
    is_distributed=False,
    start_iter=0,
    )  
    
    for i, (_, target, img_id, phrase_ids, sent_id, sentence, precompute_bbox, precompute_score, feature_map, vocab_label_elmo, sent_sg, topN_box) in enumerate(data_loader):
        pass


if __name__ == "__main__":
    main()
