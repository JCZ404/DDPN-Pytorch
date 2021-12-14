"""
Implements the Visual Grounding based on Generalized R-CNN framework
"""

import torch
import logging
from torch import nn
from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads
from ..vg.vg_head import build_vg_head

class GeneralizedRCNNVG(nn.Module):
    """
    Main class for Visual Grounding based on Generalized R-CNN
    """

    def __init__(self,cfg) -> None:
        super(GeneralizedRCNNVG, self).__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)             
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        roi_heads = build_roi_heads(cfg, out_channels)

        det_roi_head_feature_extractor = roi_heads.box.feature_extractor  ## to extract the feature from C4 of ResNet
        self.vg_head = build_vg_head(cfg, det_roi_head_feature_extractor)
        self.is_resnet_fix = cfg.MODEL.VG.FIXED_RESNET
        self.is_fpn_fixed = cfg.MODEL.VG.FIXED_FPN
        self.is_det_head_fixed = cfg.MODEL.VG.FIXED_ROI_HEAD


        self.is_vg_on = cfg.MODEL.VG_ON
        if self.is_vg_on:
            self.detection_backbone = None
            if self.is_resnet_fix:
                print('fix resent on')
                self.backbone[0].eval()
                for each in self.backbone[0].parameters():
                    each.requires_grad = False


                for key, value in det_roi_head_feature_extractor.named_parameters():
                    if 'head' in key:
                        value.requires_grad = False

                # self.vg_head.RCNN_top.eval()
                # for each in self.vg_head.RCNN_top.parameters():
                #     each.requires_grad = False
            if self.is_fpn_fixed:
                print('fix fpn on')
                self.backbone[1].eval()
                for each in self.backbone[1].parameters():
                    each.requires_grad = False

        self.logger = logging.getLogger(__name__)

    def forward(self, images, features=None, targets=None, phrase_ids=None, sentence=None, precomp_props=None,
                precomp_props_score=None, img_ids=None, object_vocab_elmo=None, sent_sg=None):
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
        det_target = targets

        assert phrase_ids is not None and sentence is not None
        all_loss, results = \
            self.vg_head(features, det_target, phrase_ids, sentence, precomp_props, precomp_props_score, img_ids, object_vocab_elmo, sent_sg)

        losses = {}
        losses.update(all_loss)
        if self.training:
            return losses

        return losses, results