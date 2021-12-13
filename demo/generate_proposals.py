import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import os

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12
os.environ['CUDA_VISIBLE_DEVICES']='2'

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


"""
RPN
    POST_NMS_TOP_N_TRAIN    2000
    POST_NMS_TOP_N_TEST     1000
    NMS_THRESH              0.7

ROI_HEAD
    ROI_HEADS.SCORE_THRESH  0.01
    ROI_HEADS.NMS           0.3
    DETECTIONS_PER_IMG      100

"""
config_file = "../configs/e2e_faster_rcnn_R_50_C4_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

def load(file_name):
    """
    Gvien the file name, open the image and return PIL format image
    """
    image_dir = "../flickr_datasets/flickr30k_images"
    pil_image = Image.open(os.path.join(image_dir,file_name))
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()
#! create Faster RCNN with the pre-trained weight
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    weight_loading="../pretrained_models/e2e_faster_rcnn_R_50_C4_1x.pth"
)

#! extract the training image's feature and proposal
with open('/home/zhangjiacheng/Code/maskrcnn-benchmark/flickr_datasets/split/train_backpack.txt','r') as f:
    image_name_list = f.readlines()
    for image_name in image_name_list:
        image_file = image_name.strip()+'.jpg'

        image = load(image_file)     
        predictions = coco_demo.run_on_opencv_image(image, image_file)