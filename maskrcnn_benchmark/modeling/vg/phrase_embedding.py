#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-06-16 14:32
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn


import torch
import torch.nn as nn
from maskrcnn_benchmark.config import cfg

import json
import numpy as np


class PhraseEmbeddingSent(torch.nn.Module):
    def __init__(self, cfg, phrase_embed_dim=1024, bidirectional=True):
        super(PhraseEmbeddingSent, self).__init__()

        self.phrase_select_type = 'Mean'

        vocab_file = open(cfg.MODEL.VG.VOCAB_FILE)
        self.vocab = json.load(vocab_file)
        vocab_file.close()
  
        self.vocab_to_id = {v:i+1 for i,v in enumerate(self.vocab)}

        self.embed_dim = phrase_embed_dim
        self.hidden_dim = self.embed_dim//2

        self.embedding = nn.Embedding(num_embeddings=len(self.vocab_to_id) + 1,
                                      embedding_dim=self.embed_dim,
                                      padding_idx=0,  # -> first_dim = zeros
                                      sparse=False)

        self.sent_rnn = nn.GRU(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)
        if cfg.MODEL.RELATION.INTRA_LAN:
            self.rel_rnn = nn.GRU(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)


    def forward(self, all_sentences, all_phrase_ids, all_sent_sgs, device_id):

        batch_phrase_ids = []
        batch_phrase_types = []
        batch_phrase_embed = []
        batch_rel_phrase_embed = []
        batch_relation_conn = []
        batch_word_embed = []
        batch_word_to_graph_conn = []

        for idx, sent in enumerate(all_sentences):                          # all_sentence: [b, dict] for one caption ; all_phrase_ids: [b, max_phrase_num] which is the phrase have the bbox annotation; all_sent_sgs: [b, relation_num, 3] 
            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            input_phr = []
            lengths = []
            phrase_embeds_list = []

            valid_phrases = filter_phrase(phrases, all_phrase_ids[idx])     # valid_phrases are the phrase which have the bbox annotation
            tokenized_seq = seq.split(' ')

            input_seq_idx = []
            for w in tokenized_seq:
                input_seq_idx.append(self.vocab_to_id[w])                   # self.vocab_to_ids is a dict record all word in caption

            input_seq_idx = torch.LongTensor(input_seq_idx).to(device_id)   # input_seq_idx: [word_num]
            seq_embeds = self.embedding(input_seq_idx)                      # seq_embeds: [word_num, embedding_dim]
            seq_embeds, _ = self.sent_rnn(seq_embeds.unsqueeze(0))          # seq_embeds: [1, word_num, 2*hidden_dim]

            word_to_graph_conn = np.zeros((len(valid_phrases), seq_embeds.shape[1]))

            for pid, phr in enumerate(valid_phrases):                       # word_to_graph_conn: [valid_phrase_num, word_num] means the position of each phrase in the caption
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')

                phr_len = len(tokenized_phr)
                start_ind = phr['first_word_index']
                if self.phrase_select_type == 'Mean':
                    phrase_embeds_list.append(torch.mean(seq_embeds[:, start_ind:start_ind+phr_len, :], 1))
                elif self.phrase_select_type == 'Sum':
                    phrase_embeds_list.append(torch.sum(seq_embeds[:, start_ind:start_ind+phr_len, :], 1))
                else:
                    raise NotImplementedError

                lengths.append(phr_len)
                input_phr.append(tokenized_phr)                             # input_phr: [valid_phrase_num,] in which is the list store the tokenized phrase
                word_to_graph_conn[pid, start_ind:start_ind+phr_len] = 1

            phrase_embeds = torch.cat(tuple(phrase_embeds_list), 0)         # phrase_embeds: [valid_phrase_num, 2*hidden_dim]

            batch_word_embed.append(seq_embeds[0])                          # batch_word_embed: [b, word_num, 2*hidden_dim]
            batch_phrase_ids.append(phrase_ids)                             # batch_phrase_ids: [b, valid_phrase_num]      string
            batch_phrase_types.append(phrase_types)                         # batch_phrase_types: [b, valied_phrase_num]   string
            batch_phrase_embed.append(phrase_embeds)                        # batch_phrase_embed: [b, valied_phrase_num, 2*hidden_dim]
            batch_word_to_graph_conn.append(word_to_graph_conn)             # batch_word_to_graph_conn: [b, valid_phrase_num, word_num]

            if cfg.MODEL.RELATION.INTRA_LAN:
                """
                rel phrase embedding
                """
                # get sg
                sent_sg = all_sent_sgs[idx]
                relation_conn = []                                          # relation_conn: [valid_relation_num, 3]
                rel_lengths = []
                input_rel_phr = []
                input_rel_phr_idx = []

                for rel_id, rel in enumerate(sent_sg):
                    sbj_id, obj_id, rel_phrase = rel
                    if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                        continue
                    relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_id]) 

                    uni_rel_phr_idx = torch.zeros(len(tokenized_seq)+5).long()
                    tokenized_phr_rel = rel_phrase.lower().split(' ')
                    if cfg.MODEL.RELATION.INCOR_ENTITIES_IN_RELATION:
                        tokenized_phr_rel = input_phr[phrase_ids.index(sbj_id)] + tokenized_phr_rel + input_phr[
                            phrase_ids.index(obj_id)]

                    rel_phr_idx = []                                    # tokenized_phr_rel: list, the tokens of relation phrase(maybe can incoperate the subject and object phrase)
                    for w in tokenized_phr_rel:
                        rel_phr_idx.append(self.vocab_to_id[w])

                    rel_phr_len = len(tokenized_phr_rel)
                    rel_lengths.append(rel_phr_len)
                    input_rel_phr.append(tokenized_phr_rel)
                    uni_rel_phr_idx[:rel_phr_len] = torch.Tensor(rel_phr_idx).long()
                    input_rel_phr_idx.append(uni_rel_phr_idx)           # input_rel_phr_idx: list, the relation phrase index in caption vocab

                if len(relation_conn) > 0:
                    input_rel_phr_idx = torch.stack(input_rel_phr_idx)  # input_rel_phr_idx: [valid_relation_num, max_relation_len]
                    rel_phrase_embeds = self.embedding(input_rel_phr_idx.to(device_id))
                    rel_phrase_embeds, _ = self.rel_rnn(rel_phrase_embeds)
                    rel_phrase_embeds = select_embed(rel_phrase_embeds, lengths=rel_lengths, select_type=self.phrase_select_type)  # rel_phrase_embeds: [valid_relation_num, 1]
                    batch_rel_phrase_embed.append(rel_phrase_embeds)
                else:
                    batch_rel_phrase_embed.append(None)                 # batch_rel_phrase_embed: [b, valid_relation_num, 1]

                batch_relation_conn.append(relation_conn)               # batch_relation_conn: [b, valid_relation_num,3]

        return batch_phrase_ids, batch_phrase_types, batch_word_embed, batch_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_word_to_graph_conn


def filter_phrase(phrases, all_phrase):
    phrase_valid = []
    for phr in phrases:
        if phr['phrase_id'] in all_phrase:
            phrase_valid.append(phr)
    return phrase_valid


def select_embed(x, lengths, select_type=None):
    batch_size = x.size(0)          
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        # if select_type == 'last':
        #     mask[i][lengths[i] - 1].fill_(1)
        if select_type == 'Mean':
            mask[i][:lengths[i]].fill_(1/lengths[i])
        elif select_type == 'Sum':
            mask[i][:lengths[i]].fill_(1)
        else:
            raise NotImplementedError

    x = x.mul(mask)
    x = x.sum(1).view(batch_size, -1) 
    return x


def specific_word_replacement(word_list):
    """
    :param word_list: ["xxx", "xxx", "xxx"]
    :return: new word_list: ["xxx", 'xxx', 'xxx']
    """
    new_word_list = []
    for word in word_list:
        if word in left_word_dict:
            word = word.replace('left', 'right')
        elif word in right_word_dict:
            word = word.replace('right', 'left')
        new_word_list.append(word)
    return new_word_list


