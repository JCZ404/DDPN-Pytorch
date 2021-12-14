import torch
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import  boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
import numpy as np


class VGLoss:
    def __init__(self, cfg):
        self._proposals = None

        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)
        self.proposal_matcher = Matcher(
            cfg.MODEL.VG.FG_IOU_THRESHOLD,
            cfg.MODEL.VG.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )

    def __call__(self, batch_phrase_ids, batch_all_phrase_ids, batch_det_target, batch_pred_similarity,
                 batch_reg_offset, batch_precomp_boxes, device_id):

        cls_loss = torch.zeros(1).to(device_id)
        reg_loss = torch.zeros(1).to(device_id)
    

        batch_gt_boxes = []
        for bid, (phrase_ids, all_phrase_ids, det_target, pred_similarity, reg_offset, precomp_boxes) \
                in enumerate(zip(batch_phrase_ids, batch_all_phrase_ids, batch_det_target, batch_pred_similarity,
                       batch_reg_offset, batch_precomp_boxes)):

            order = []
            for id in phrase_ids:
                order.append(all_phrase_ids.index(id))
            gt_boxes = det_target[np.array(order)].to(device_id)                    # gt_boxes: [valid_phrase_num, 4]
            batch_gt_boxes.append(gt_boxes)                                         # batch_gt_boxes: [valid_phrase_num, 4]
            

            ious = boxlist_iou(gt_boxes, precomp_boxes)  ## M*100                   # precomp_boxes: [precomp_box_num, 4], ious: [valid_phrase_num, precomp_box_num]
            mask = ious.ge(cfg.MODEL.VG.FG_IOU_THRESHOLD)
            gt_scores = F.normalize(ious * mask.float(), p=1, dim=1)                # gt_scores is the soft label of the similarity prediction, gt_scores: [valid_phrase_num, precomp_box_num]

           

            # print(ious)
            # print(ious_topN)
            # input("please check the iou!")
            """ classification loss """
            if cfg.MODEL.VG.CLS_LOSS_TYPE == 'Softmax':
                cls_loss += -(gt_scores * pred_similarity.log()).mean()
            else:
                raise NotImplementedError("Only use Softmax loss")

            """ reg loss """
            pos_inds = torch.nonzero(ious >= (cfg.MODEL.VG.FG_REG_IOU_THRESHOLD))
            if len(pos_inds) > 0:
                phr_ind, obj_ind = pos_inds.transpose(0, 1)         # pos_inds: [num_positive_precomp_box, 2], here 2 means the 2-d index,
                regression_targets = self.box_coder.encode(
                    gt_boxes[phr_ind].bbox, precomp_boxes[obj_ind].bbox
                )
                obj_ind += phr_ind * precomp_boxes.bbox.shape[0]
                regression_pred = reg_offset[obj_ind]
                reg_loss += cfg.SOLVER.REGLOSS_FACTOR * smooth_l1_loss(
                    regression_pred,
                    regression_targets,
                    size_average=True,
                    beta=1,
                )

        return cls_loss, reg_loss, batch_gt_boxes