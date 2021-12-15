from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
import json
from maskrcnn_benchmark.config import cfg
import numpy as np
import os.path as osp
import os

def eval_recall(dataset, predictions, image_ids, output_folder=None):
    total_num = 0
    recall_num = 0
    
    for img_sent_id in image_ids:

        result = predictions[img_sent_id]

        
        gt_boxes, pred_boxes, pred_sim = result

    
        pred_boxes = BoxList(pred_boxes, gt_boxes.size, mode="xyxy")
        pred_boxes.clip_to_image()
        ious = boxlist_iou(gt_boxes, pred_boxes)
        iou = ious.cpu().numpy().diagonal()
        total_num += iou.shape[0]
        recall_num += int((iou>=cfg.MODEL.VG.EVAL_THRESH).sum()) # 0.5

    acc = recall_num/total_num
 
    return acc