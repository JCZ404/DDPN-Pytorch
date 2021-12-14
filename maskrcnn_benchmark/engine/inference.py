# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
from numpy import record

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


""" add the val during training """
def eval_while_train(cfg, model, curr_iter, data_loader, output_folder):
    torch.cuda.empty_cache()
    print("start testing while training...")

    # only the first one for test 
    model.eval()
    results_dict = {}
    device = torch.device('cuda')
    cpu_device = torch.device("cpu")

    for bid, (
    images, targets, image_ids, phrase_ids, sent_ids, sentence, precompute_bbox, precompute_score, feature_map,
    vocab_label_elmo, sent_sg, topN_box) in enumerate(tqdm(data_loader)):

        # put the data into the cuda
        vocab_label_elmo = [vocab.to(device) for vocab in vocab_label_elmo]
        features_list = [feat.to(device) for feat in feature_map]

        # forward pass without gradient
        with torch.no_grad():

            loss_dict, results = model(images, features_list, targets, phrase_ids, sentence, precompute_bbox,
                                       precompute_score, image_ids, vocab_label_elmo, sent_sg, topN_box)

            # collect and move result to cpu memory
            moved_res = []
               
            batch_gt_boxes, batch_pred_box, batch_pred_similarity = results
            for idx, each_gt_boxes in enumerate(batch_gt_boxes):
                moved_res.append((each_gt_boxes.to(cpu_device),
                                    batch_pred_box[idx].to(cpu_device),
                                    batch_pred_similarity[idx].to(cpu_device)))

            results_dict.update(
                {img_id + '_' + sent_id: result
                 for img_id, sent_id, result in zip(image_ids, sent_ids, moved_res)}
            )

    predictions = results_dict
    image_sent_id = predictions.keys()

    torch.cuda.empty_cache()
    if not is_main_process():
        return

    # do evaluation
    acc = evaluate(dataset=data_loader.dataset,
                            predictions=predictions,
                            image_ids=image_sent_id)

    # print the evaluation information
    print("Evaluation ......")
    record = ""
    for key, value in loss_dict.items():
        record = record + key + ":" + str(value.data.cpu().numpy()) + "|"
    record = record + "Current Accuracy:{}".format(acc)
    print(record)
    print("Evulation Done!")
    return acc
