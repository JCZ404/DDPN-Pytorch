# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import imp
import logging
import os
import torch
import random   
import numpy as np
import sys
import time


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
from maskrcnn_benchmark.engine.inference import eval_while_train
from torch.utils.tensorboard import SummaryWriter


current_path = os.path.dirname(__file__)
sys.path.append(current_path)

# try to solve too many opened file Error
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_printoptions(precision=12)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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


def main():
    cfg.merge_from_file("../configs/lcmcg_flickr_bottom_up_10_100_1x.yaml")
    cfg.freeze()    

    """" train model """
    model = train(cfg, eval_while_training=True)

    

def train(cfg,  eval_while_training):

    """ create model, data and learning configs """
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    data_loader = make_data_loader(
    cfg,
    is_train=True,
    is_distributed=False,
    start_iter=0,
    ) 

    """ set evaluate when training """
    if eval_while_training:
        val_data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
    else:
        print("test_while_training off ")
        val_data_loader = None

    """ create the saving config """
    # set the output dir
    output_dir = cfg.OUTPUT_DIR
    if output_dir != '':
        mkdir(output_dir)
    else:
        output_dir = None

    checkpoint_output_dir = os.path.join(output_dir, 'checkpoints')
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, checkpoint_output_dir, save_to_disk
    )


    """ do training """
    max_iter = len(data_loader)

    model.train()
    # make the bn layer of backbone fixed
    if cfg.MODEL.VG_ON and cfg.MODEL.VG.FIXED_RESNET:
        model.backbone.eval()

    # create the summary writer
    writer = SummaryWriter(log_dir='../logs')
    # set the evaluation interval
    eval_interval = 500
    best_result = 0
    for iteration, (images, targets, img_ids, phrase_ids, sent_id, sentence, precompute_bbox, precompute_score, feature_map, vocab_label_elmo, sent_sg, topN_box) in enumerate(data_loader):
        # put the feature into the cuda devices
        features_list = [feat.to(device) for feat in feature_map]
        vocab_label_elmo = [vocab.to(device) for vocab in vocab_label_elmo]


        """ forward pass """
        loss_dict = model(images, features_list, targets, phrase_ids, sentence, precompute_bbox,
                          precompute_score, img_ids, vocab_label_elmo, sent_sg)

        """ backward pass """
        losses = sum(loss for loss in loss_dict.values())

        if losses > 0:
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        scheduler.step()

        """ print the training information """
        if iteration % 100 == 0 or iteration == max_iter:
            lr=optimizer.param_groups[0]["lr"]
            if iteration % 100 == 0:
                record = "{}/{} lr:{} ".format(iteration,max_iter, lr)
                for key,value in loss_dict.items():
                    record = record + key + ":" + str(value.data.cpu().numpy()[0]) + " | "
                record = record + "total_loss:" + str(losses.data.cpu().numpy()[0])
                print(record)

        """ evaluate and save model """
        if eval_while_training:
            if iteration % eval_interval == 0:
                # test while training
                if val_data_loader is not None:

                    for idx, val_data_loader_i in enumerate(val_data_loader):
                        dataset_name = cfg.DATASETS.TEST[idx]

                        output_folder = os.path.join(cfg.OUTPUT_DIR, 'eval_res', dataset_name)
                        if not os.path.exists(output_folder):
                            mkdir(output_folder)

                        curr_acc = eval_while_train(cfg=cfg,
                                        model=model,
                                        curr_iter=iteration,
                                        data_loader=val_data_loader_i,
                                        output_folder=output_folder)

                        # save the better model
                        if curr_acc > best_result:
                            checkpointer.save("model_{:07d}".format(iteration))

                    model.train()
                    if cfg.MODEL.VG_ON and cfg.MODEL.VG.FIXED_RESNET:
                        model.module.backbone.eval()

        

        """ visualization """
        for key,value in loss_dict.items(): 
            writer.add_scalar(key, value, iteration)

    
    return model


if __name__ == "__main__":
    main()
