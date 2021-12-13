import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import os

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def load(file_name):
    """
    Gvien the file name, open the image and return PIL format image
    """
    image_dir = "../data/flickr30k_images"
    pil_image = Image.open(os.path.join(image_dir,file_name))
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

config_file = "../configs/e2e_faster_rcnn_R_50_C4_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
    weight_loading="../pretrained_models/e2e_faster_rcnn_R_50_C4_1x.pth"
)

# compute predictions
image = load("390369.jpg")
predictions = coco_demo.run_on_opencv_image(image)
imshow(predictions)
# 