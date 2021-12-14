import numpy as np
import torch
from torch.nn import functional as F
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch.nn as nn
from .vg_loss import VGLoss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.vg.phrase_embedding import PhraseEmbeddingSent
from maskrcnn_benchmark.layers.spatial_coordinate import *


class VGHead(torch.nn.Module):
    """
    Basic Visual Grounding Head based on Similarity Network
    """
    def __init__(self, cfg, det_roi_head_feature_extractor: torch.nn.Module) -> None:
        super(VGHead,self).__init__()
        self.cfg = cfg
        # extract proposal feature on the C4 feature of ResNet 
        self.det_roi_head_feature_extractor =  det_roi_head_feature_extractor
        self.obj_embed_dim = self.det_roi_head_feature_extractor.out_channels  # 2048
        self.phrase_embed_dim = 1024

        # encoding the phrase text
        self.phrase_embed = PhraseEmbeddingSent(cfg, phrase_embed_dim=self.phrase_embed_dim, bidirectional=True)
        self.recognition_dim = 1024

        if cfg.MODEL.VG.SPATIAL_FEAT:
            self.obj_embed_dim = self.obj_embed_dim + 256

        # encodeing the extracted proposal feature
        self.visual_embedding = nn.Sequential(
            nn.Linear(self.obj_embed_dim, self.recognition_dim),
            nn.LeakyReLU(),
            nn.Linear(self.recognition_dim, self.recognition_dim)
        )

       
        # input network dim of similarity network
        self.similarity_input_dim = self.recognition_dim + self.phrase_embed_dim * 3

        # similarity network
        self.similarity = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        # regression network for refine proposal
        self.box_reg = nn.Sequential(
        nn.Linear(self.similarity_input_dim, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 4)
        )

        # box_coder used to apply the predicted offset to refine the proposal
        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)
 
        # loss of similarity network
        self.VGLoss = VGLoss(cfg)


    def init_weights(self):
    
        nn.init.xavier_normal_(self.visual_embedding.weight.data)
        self.visual_embedding.bias.data.zero_()
        nn.init.xavier_normal_(self.similarity.weight.data)
        self.similarity.bias.data.zero_()


    def forward(self, features, batch_det_target, batch_all_phrase_ids, all_sentences, precomp_boxes,
                precomp_boxes_score, img_ids, object_vocab_elmo, all_sent_sgs):
        """
        feature: [B, 1024, H, W], tensor, C4 feature of ResNet101
        batch_det_target: [B, target_box_num, 4], list, target phrase ground truth box, x means changeable
        batch_all_phrase_ids: [B, phrase_num, ], list, all_phrase_ids for each image caption, phrase_ids is a string
        all_sentences: [B, sentence_num,], list, all_sentence in a batch data, sentence is a dict stores the information about the sentence
        precomp_boxes: [B, proposal_num, 4], list, generated boxes for each image in the batch

        """
        device_id = features[0].get_device()                            # features: [b, 1024, H, W]
        precomp_boxes_size = [len(props) for props in precomp_boxes]    # precomp_boxes: [b, precomp_box_num, 4]

        # encodeing the phrase text
        batch_phrase_ids, batch_phrase_types, batch_word_embed, batch_phrase_embed, \
        batch_rel_phrase_embed, batch_relation_conn, batch_word_to_graph_conn = \
            self.phrase_embed(all_sentences, batch_all_phrase_ids, all_sent_sgs, device_id=device_id)


        batch_final_similarity = []
        batch_final_box = []
     

        batch_similarity_pred = []
        batch_offset_pred = []
        batch_precomp_boxes = []

     

        for bid, each_img_prop_size in enumerate(precomp_boxes_size):
            precomp_boxes_i = precomp_boxes[bid].to(device_id)                                                              # precmp_boxes_i: [precomp_box_num, 4]

            feat = features[bid]

            """ add the spatial feature channel """
            if cfg.MODEL.VG.SPATIAL_FEAT:
                spa_feat = meshgrid_generation(feat=feat)          # spa_feat: [1, 2, H, W]
                feat = torch.cat((features[bid], spa_feat), 1)     # feat: [1,1024+2,H,W]
          
            features_bid = self.det_roi_head_feature_extractor(tuple([feat]), [precomp_boxes_i])                            # features_bid: [precomp_box_num, 2048+256]
            phrase_embed_i = batch_phrase_embed[bid]
            batch_precomp_boxes.append(precomp_boxes_i)                                                                     # batch_precom_boxes: [b, precomp_box_num, 4]

            num_box = precomp_boxes_i.bbox.size(0)
            num_phrase = phrase_embed_i.size(0)

            all_phr_ind, all_obj_ind = make_pair(num_phrase, num_box) 

            features_bid = self.visual_embedding(features_bid)     # 1024

            relation_conn_i = batch_relation_conn[bid]


            ## prediction the similarity between each phrase and all proposals
            pred_similarity, pred_offset = self.prediction(features_bid[all_obj_ind],phrase_embed_i[all_phr_ind])           # pred_similarity: [precomp_box_num*phrase_num,1], reg_offset: [precomp_box_num*phrase_num,4]

            pred_similarity = torch.softmax(pred_similarity.reshape(num_phrase, num_box), dim=1)                            # pred_similarity: [num_phrase,num_precomp_box]
            batch_similarity_pred.append(pred_similarity)
            batch_offset_pred.append(pred_offset)

            if not self.training:
                """ select the best matching proposal """
                pred_similarity_all = pred_similarity.detach().cpu().numpy()                                                # pred_similarity: [num_prhase, num_precomp_box]
                select_ind_all = pred_similarity_all.argmax(1)
       

                select_box_all = precomp_boxes_i[select_ind_all]                                                            # precomp_boxes_i: [num_precomp_box, 4], select_box_all: [num_phrase, 4]
                select_reg_ind_all = select_ind_all + precomp_boxes_i.bbox.shape[0] * np.arange(num_phrase)
                select_offset_all = pred_offset[select_reg_ind_all]                                                         # select_offset_all: [num_phrase, 4]
                pred_box_all = self.VGLoss.box_coder.decode(select_offset_all, select_box_all.bbox)                         # apply the offset to the region proposal

                batch_final_box.append(pred_box_all)


        # calculate the classification loss and the regression loss
        cls_loss, reg_loss, batch_gt_boxes = self.VGLoss(batch_phrase_ids, batch_all_phrase_ids,                            # batch_phrase_ids: [b, num_valid_phrase], batch_all_phrase_ids: [b, num_phrase]
                                                         batch_det_target, batch_similarity_pred, batch_offset_pred,        # batch_det_target: [b, phrase_num, 4]                            
                                                         batch_precomp_boxes,
                                                         device_id)

        all_loss = dict(cls_loss=cls_loss,
                        reg_loss=reg_loss,
                    )


        if self.training:
            return all_loss,None

       
        return all_loss, (batch_gt_boxes, batch_final_box, batch_similarity_pred)


    def prediction(self, features, phrase_embed):
        fusion_embed = torch.cat((phrase_embed, features), 1)       # phrase_embed [215,1024] features [215,1024]
        cosine_feature = fusion_embed[:, :1024] * fusion_embed[:, 1024:2048]
        delta_feature = fusion_embed[:, :1024] - fusion_embed[:, 1024:2048]
        fusion_embed = torch.cat((cosine_feature, delta_feature, fusion_embed), 1)
        
        pred_similarity = self.similarity(fusion_embed)
        reg_offset = self.box_reg(fusion_embed)
        return pred_similarity, reg_offset



def make_pair(phr_num: int, box_num: int):
    ind_phr, ind_box = np.meshgrid(range(phr_num), range(box_num), indexing='ij')
    ind_phr = ind_phr.reshape(-1)
    ind_box = ind_box.reshape(-1)
    return ind_phr, ind_box



def build_vg_head(cfg, det_roi_heads):
    if cfg.MODEL.VG_ON:
        return VGHead(cfg, det_roi_heads)
    else:
        return None