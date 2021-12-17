## Phrase Grounding Project

This is the Pytorch implementation of the Paper `Rethinking Diversified and Discriminative Proposal Generation for Visual Grounding`. The official implementation is based on Caffe, so I convert that project to the Pytorch based on maskrcnn-benchmark.

### Requirements
This propject is based on maskrcnn-benchmark, so you need to follow the install instruction of the maskrcnn-benchmark. However, because maskrcnn-benchmark is no longer maintained, so the installation maybe annoyed, here I provided my installation.
* Pytorch 1.2.0
* torchvision 0.4.0
* RTX 2080Ti Cuda9.2

After finishing these installation, you can follow the original installation of maskrcnn-benchmark repo to install.

### Usage
DDPN is a two stage method of Phrase Grounding, so there are steps to do the Phrase Grounding.
1. Step1: Generation Proposals
First you need to generate some proposals on the image, the most popular way is use the  `bottom-up-attention` model which is pretrained on the Visual GNome dataset to generate some proposals on the image, e.g., 10-100 proposals per image. Recommend to use `https://github.com/MILVLG/bottom-up-attention.pytorch` repo with ROI_NMS=0.3, ROI_SCORE_THRESHOLD=0.1 to generate 10-100 proposals per image and save these precomputed proposals 
2. Step2: Check Generated Proposals
I have alread write the demo in folder demo/check_proposal.ipynb, you can follow this file to write your own checking file.
3. Step3: Training your model
Run the command:
```shell
python train_net_vg.py
```