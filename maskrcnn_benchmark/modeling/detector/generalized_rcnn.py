# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, image_name=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        else: 
            #! for test, we save the feature and proposal we want
            import numpy as np
            res = dict()
            res["features"] = features[0].cpu().numpy()
            scores = result[0].get_field('scores')[:,None].cpu().numpy()
            # print("Shape of Proposal score:",scores.shape)
            labels = result[0].get_field('labels')[:,None].cpu().numpy()
            # print("Shape of Proposal labels:",labels.shape)
            boxes = result[0].bbox.cpu().numpy()
            t = np.hstack([boxes,scores,labels])
            # print("Shape of the final box information:",t.shape)
            res["boxes"] = t
            res["img_scale"] = result[0].size
            # print("The input image size of the network:", res["img_scale"])
            import pickle as pkl
            with open("/data2/zjc/flickr30k_feat_nms/{}.pkl".format(image_name.split(".")[0]),'wb') as f:
                pkl.dump(res,f)
            
        return result
