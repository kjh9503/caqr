#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
from structural_table import *


def Identity(x):
    return x


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding


class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

class Fusion_all(nn.Module): #projection 후에 cascading error 줄이기
    def __init__(self, entity_dim, structure_dim, proj_reg=None):
        super(Fusion_all, self).__init__()
        self.entity_dim = entity_dim
        self.structure_dim = structure_dim
        self.proj_reg = proj_reg

        emb_expand_dim = entity_dim * 2
        structure_expand_dim = structure_dim * 2

        post_dim = emb_expand_dim + structure_expand_dim #(800*2 + 108*2)

        self.layer1 = nn.Linear(self.structure_dim, self.entity_dim)
        self.layer2 = nn.Linear(self.entity_dim, self.entity_dim)

        self.mats1_emb = nn.Parameter(torch.FloatTensor(entity_dim, emb_expand_dim))
        nn.init.xavier_uniform_(self.mats1_emb)
        self.register_parameter("mats1_emb", self.mats1_emb)

        self.post_mats_emb = nn.Parameter(torch.FloatTensor(post_dim, entity_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_emb)
        self.register_parameter("post_mats_emb", self.post_mats_emb)

        self.mats1_structure = nn.Parameter(torch.FloatTensor(structure_dim, structure_expand_dim))
        nn.init.xavier_uniform_(self.mats1_structure)
        self.register_parameter("mats1_structure", self.mats1_structure)

    def forward(self, query_emb, meta_embeddings):
        #query_emb : (bs, 800) meta_embeddings : [pos:(1, 800), rol:(1, 800), strc:(1,800)]

        bs = query_emb.size(0)

        structure_embeddings = torch.cat(meta_embeddings[:-1], dim=0) #pos, rol, strc -> (3, 108)
        structure_embeddings = structure_embeddings.unsqueeze(0) # (1, 3, 108)
        structure_embeddings_q = F.relu(self.layer1(structure_embeddings)) # (1, 3, 800)
        structure_embeddings_q = self.layer2(structure_embeddings_q) #(1, 3, 800)
        structure_embeddings_q = structure_embeddings_q.repeat(bs, 1, 1) #(bs, 3, 800)

        attention = F.softmax(torch.matmul(structure_embeddings_q, torch.transpose(query_emb.unsqueeze(1), 1, 2)), dim=1) #(bs, 3, 1)

        structure_embedding = torch.sum(attention * structure_embeddings, dim=1) #(bs, 108)

        post_concat = []

        post_query_emb = F.relu(torch.matmul(query_emb, self.mats1_emb)) #(bs, 2d)
        post_concat.append(post_query_emb)

        post_structure_emb = F.relu(torch.matmul(structure_embedding, self.mats1_structure)) #(bs, 216)
        post_concat.append(post_structure_emb) #[(bs, 1600), (bs, 216)

        post_concat = torch.cat(post_concat, dim=-1) #(bs, 1, 2*(800 + 108))
        #####################################################################################
        emb = torch.matmul(post_concat, self.post_mats_emb)
        emb = self.proj_reg(emb)

        return emb

class Fusion(nn.Module): #projection 후에 cascading error 줄이기
    def __init__(self, entity_dim, structure_dim, proj_reg=None):
        super(Fusion, self).__init__()
        self.entity_dim = entity_dim
        self.structure_dim = structure_dim
        self.proj_reg = proj_reg

        emb_expand_dim = entity_dim * 2
        structure_expand_dim = structure_dim * 2
        hyper_expand_dim = entity_dim * 2

        post_dim = emb_expand_dim + structure_expand_dim + hyper_expand_dim #(800*2 + strc_dim*2 + 800*2)

        self.layer1 = nn.Linear(self.structure_dim, self.entity_dim)
        self.layer2 = nn.Linear(self.entity_dim, self.entity_dim)

        self.mats1_emb = nn.Parameter(torch.FloatTensor(entity_dim, emb_expand_dim))
        nn.init.xavier_uniform_(self.mats1_emb)
        self.register_parameter("mats1_emb", self.mats1_emb)

        self.post_mats_emb = nn.Parameter(torch.FloatTensor(post_dim, entity_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_emb)
        self.register_parameter("post_mats_emb", self.post_mats_emb)

        self.mats1_structure = nn.Parameter(torch.FloatTensor(structure_dim, structure_expand_dim))
        nn.init.xavier_uniform_(self.mats1_structure)
        self.register_parameter("mats1_structure", self.mats1_structure)

        self.mats1_hyper = nn.Parameter(torch.FloatTensor(entity_dim, hyper_expand_dim))
        nn.init.xavier_uniform_(self.mats1_hyper)
        self.register_parameter("mats1_hyper", self.mats1_hyper)

    def forward(self, query_emb, structure_meta_embeddings, hyper_embedding):
        #query_emb : (bs, 800) meta_embeddings : [pos:(1, 800), rol:(1, 800), strc:(1,800)], hpyer_embedding : (bs, 800)

        bs = query_emb.size(0)

        structure_embeddings = torch.cat(structure_meta_embeddings, dim=0) #pos, rol, strc -> (3, 108)
        structure_embeddings = structure_embeddings.unsqueeze(0) # (1, 3, 108)
        structure_embeddings_q = F.relu(self.layer1(structure_embeddings)) # (1, 3, 800)
        structure_embeddings_q = self.layer2(structure_embeddings_q) #(1, 3, 800)
        structure_embeddings_q = structure_embeddings_q.repeat(bs, 1, 1) #(bs, 3, 800)

        attention = F.softmax(torch.matmul(structure_embeddings_q, torch.transpose(query_emb.unsqueeze(1), 1, 2)), dim=1) #(bs, 3, 1)

        structure_embedding = torch.sum(attention * structure_embeddings, dim=1) #(bs, 108)

        post_concat = []

        post_query_emb = F.relu(torch.matmul(query_emb, self.mats1_emb)) #(bs, 2d)
        post_concat.append(post_query_emb)

        post_structure_emb = F.relu(torch.matmul(structure_embedding, self.mats1_structure)) #(bs, 216)
        post_concat.append(post_structure_emb)

        post_hyper_emb = F.relu(torch.matmul(hyper_embedding, self.mats1_hyper))
        post_concat.append(post_hyper_emb) #[(bs, 1600), (bs, 1600), (bs, 1600)]

        post_concat = torch.cat(post_concat, dim=-1) #(bs, 1, 2*(800 + 800 + 800))
        #####################################################################################
        emb = torch.matmul(post_concat, self.post_mats_emb)
        emb = self.proj_reg(emb)

        return emb


class Regularizer():
    def __init__(self, base_add, min_val, max_val): # 1, 0.05, 1ㄷ9
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,

                 rel_hmap=None,
                 rel_tmap=None,
                 rel_h_degree=None,
                 rel_t_degree=None,

                 max_qry_len=4, num_role=3, structure_dim=108,
                 PE=False, RE=False, SE=False, table_normalization=False,
                 fusion_all=True, var_lambda=10e-5,
                 ckpt_path='data/NELL-betae',


                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size,
                                                                               1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        ) #(60 + 2)/ 800 = 0.0775

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))  # centor for entities
            activation, cen = box_mode
            self.cen = cen  # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))  # center for entities
        elif self.geo == 'beta':

            self.entity_embedding = nn.Parameter(torch.zeros(nentity+1, self.entity_dim*2),
                                                     requires_grad=True)  # axis for entities
            self.entity_regularizer = Regularizer(1, 0.05,
                                                  1e9)  # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05,
                                                      1e9)  # make sure the parameters of beta embeddings after relation projection are positive
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        ##########################################################################################
        self.PE = PE
        self.RE = RE
        self.SE = SE
        self.var_lambda = var_lambda
        ############################################
        ############################################
        if var_lambda :
            ckpt = torch.load('{}/checkpoint'.format(ckpt_path))['model_state_dict']
            self.ref_ent_emb = nn.Parameter(ckpt['entity_embedding'], requires_grad=False)
            self.ref_rel_emb = nn.Parameter(ckpt['relation_embedding'], requires_grad=False)
        ############################################
        ############################################

        self.max_qry_len = max_qry_len
        self.num_role = num_role
        self.structure_dim = structure_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.table_normalization = table_normalization

        self.last_rel_idx = {'ip': 4, 'inp': 5}
        self.ent_rel_2_idx = {'pi': [3, 4], 'pin': [3, 4], 'pni': [4, 5]}

        self.Hyper = False
        if rel_hmap is not None :
            self.Hyper = True
            self.hyper_dim = hidden_dim
            assert rel_tmap is not None and rel_h_degree is not None and rel_t_degree is not None

            self.rel_hmap = nn.Parameter(torch.tensor(rel_hmap), requires_grad=False).to('cuda')
            self.rel_tmap = nn.Parameter(torch.tensor(rel_tmap), requires_grad=False).to('cuda')
            self.max_freq = self.rel_hmap.size(1)
            self.rel_h_degree = nn.Parameter(torch.tensor(rel_h_degree), requires_grad=False).unsqueeze(1).to('cuda')
            self.rel_t_degree = nn.Parameter(torch.tensor(rel_t_degree), requires_grad=False).unsqueeze(1).to('cuda')

            self.ent_pad_id = nentity

            logging.info('head of relation map : {}'.format(self.rel_hmap.size()))
            logging.info('tail of relation map : {}'.format(self.rel_tmap.size()))
            logging.info('head degree of relation : {}'.format(self.rel_h_degree.size()))
            logging.info('tail degree of relation : {}'.format(self.rel_t_degree.size()))
        else:
            self.hyper_dim = None

        self.transform = (self.PE or SE or RE or self.Hyper)

        """
        rel_hmap : kg_triple에서 r의 head가 되는 애들
        rel_tmap : kg_triple에서 r의 tail이 되는 애들
        """

        if not (self.PE and self.RE):
            assert not self.SE

        if self.table_normalization:
            assert self.SE

        self.init_role_embedding()
        self.init_position_embedding()
        self.init_structure_embedding()

        self.fusion_all = fusion_all

        if self.fusion_all :
            self.alpha_embedding_trans = Fusion_all(entity_dim=self.entity_dim, structure_dim=self.structure_dim,
                                                proj_reg=self.projection_regularizer)
            self.beta_embedding_trans = Fusion_all(entity_dim=self.entity_dim, structure_dim=self.structure_dim,
                                               proj_reg=self.projection_regularizer)
        else :
            self.alpha_embedding_trans = Fusion(entity_dim=self.entity_dim, structure_dim=self.structure_dim,
                                                proj_reg=self.projection_regularizer)
            self.beta_embedding_trans = Fusion(entity_dim=self.entity_dim, structure_dim=self.structure_dim,
                                               proj_reg=self.projection_regularizer)

        logging.info('Use position embedding : {}'.format(self.PE))
        logging.info('Use role embedding : {}'.format(self.RE))
        logging.info('Use structure embedding : {}'.format(self.SE))

        ##########################################################################################

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2,
                                                 self.relation_dim,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)

    def init_position_embedding(self):
        self.pos_alpha_embedding = nn.Parameter(torch.zeros(self.max_qry_len, self.structure_dim))
        self.pos_beta_embedding = nn.Parameter(torch.zeros(self.max_qry_len, self.structure_dim))

        nn.init.uniform_(tensor=self.pos_alpha_embedding, a=-self.embedding_range.item(),
                         b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.pos_beta_embedding, a=-self.embedding_range.item(),
                         b=self.embedding_range.item())

    def init_role_embedding(self):
        self.role_alpha_embedding = nn.Parameter(torch.zeros(self.num_role, self.structure_dim))
        self.role_beta_embedding = nn.Parameter(torch.zeros(self.num_role, self.structure_dim))

        nn.init.uniform_(tensor=self.role_alpha_embedding, a=-self.embedding_range.item(),
                         b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.role_beta_embedding, a=-self.embedding_range.item(),
                         b=self.embedding_range.item())

    def init_structure_embedding(self):
        self.query2onehot = get_query2onehot(self.device, self.table_normalization)
        self.structural_alpha_mapping = nn.Linear(self.max_qry_len * self.num_role, self.structure_dim)
        self.structural_beta_mapping = nn.Linear(self.max_qry_len * self.num_role, self.structure_dim)

    def node_hyper_embedding(self, var_neigh_rel_inds, ans_neigh_rel_inds, qtype):
        """
        var_neigh_rel_inds,     ans_neigh_rel_inds
        1c    : (None, None),        (bs, 1))
        2c    : ((bs, 2), None),     (bs, 1))
        3c    : ((bs, 2), (bs, 2)),  (bs, 1))
        2i,2u : (None,               (bs, 2))
        3i    : (None,               (bs, 3))
        ci    : ((bs, 2),            (bs, 2))
        ic,uc : ((bs, 3),            (bs, 1))

        """
        if qtype in ['pi', 'pin', 'pni'] :
            #var_neigh_rel_inds : (bs, 2), ans_neigh_rel_inds : (bs, 2)

            var_hyper_emb = self.in1out1_hyper_agg(var_neigh_rel_inds)
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds)

        elif qtype in ['ip', 'inp', 'up-DNF', 'up-DM'] :
            #var_neigh_rel_inds : (bs, 3), ans_neigh_rel_inds : (bs, 1)

            var_hyper_emb = self.in2out1_hyper_agg(var_neigh_rel_inds)
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds)

        elif qtype in ['2i', '2in', '3i', '3in', '2u'] :
            #var_neigh_rel_inds : None, ans_neigh_rel_inds : (bs, N)

            var_hyper_emb = None
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds) #(bs, d)

        else :
            # var_neigh_rel_inds := 1c : (None, None), 2c : ((bs, 2 ), None), 3c : ((bs, 2), (bs, 2))
            # ans_neigh_rel_inds : (bs, 1)

            rel_len = int(qtype[0])
            var1_neigh_rel_inds, var2_neigh_rel_inds = var_neigh_rel_inds
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds)
            if rel_len >= 1:
                var1_hyper_emb = None
                var2_hyper_emb = None
            if rel_len >= 2 :
                var1_hyper_emb = self.in1out1_hyper_agg(var1_neigh_rel_inds)
            if rel_len >= 3 :
                var2_hyper_emb = self.in1out1_hyper_agg(var2_neigh_rel_inds)

            var_hyper_emb = (var1_hyper_emb, var2_hyper_emb)

        return var_hyper_emb, ans_hyper_emb

    def in1out1_hyper_agg(self, node_neigh_rel_inds): #(bs, 2)
        # 2c, 3c, ci의 variable node
        bs = node_neigh_rel_inds.size(0)

        node_in_rel_tail_ents = torch.index_select(self.rel_tmap, dim=0, index=node_neigh_rel_inds[:, 0])  # (bs, mxf)
        node_in_rel_tail_ents_pad_mx = (torch.ones((bs, self.max_freq), requires_grad=False) * self.ent_pad_id).cuda()
        node_in_rel_tail_mask_mx = torch.eq(node_in_rel_tail_ents, node_in_rel_tail_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.entity_dim*2)

        node_out_rel_head_ents = torch.index_select(self.rel_hmap, dim=0, index=node_neigh_rel_inds[:, 1])  # (bs, mxf)
        node_out_rel_head_ents_pad_mx = (torch.ones((bs, self.max_freq), requires_grad=False) * self.ent_pad_id).cuda()
        node_out_rel_head_mask_mx = torch.eq(node_out_rel_head_ents, node_out_rel_head_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.entity_dim*2)

        node_in_rel_tail_ent_embs = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0,
                                                      index=node_in_rel_tail_ents.flatten())).view(bs, self.max_freq, -1)
        node_in_rel_tail_ent_embs_masked = node_in_rel_tail_ent_embs.masked_fill_(node_in_rel_tail_mask_mx, 0) #(bs, mxf, d)


        node_out_rel_head_ent_embs = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_out_rel_head_ents.flatten())).view(bs, self.max_freq, -1)
        node_out_rel_head_ent_embs_masked = node_out_rel_head_ent_embs.masked_fill_(node_out_rel_head_mask_mx, 0) #(bs, mxf, d)


        node_in_rel_tail_degree = self.rel_t_degree[node_neigh_rel_inds[:, 0]]  # (bs, 1)
        node_out_rel_head_degree = self.rel_h_degree[node_neigh_rel_inds[:, 1]]  # (bs, 1)

        node_hyper_head_emb = torch.sum(node_out_rel_head_ent_embs_masked, dim=1) / node_out_rel_head_degree  # (bs, d)
        node_hyper_tail_emb = torch.sum(node_in_rel_tail_ent_embs_masked, dim=1) / node_in_rel_tail_degree  # (bs, d)

        node_hyper_emb = (node_hyper_head_emb + node_hyper_tail_emb) / 2

        return node_hyper_emb

    def inNout0_hyper_agg(self, node_neigh_rel_inds): #(bs, N)
        #1c,2c,3c, 2i, 3i, ic,ci,2u,uc 의 answer node
        bs, N = node_neigh_rel_inds.size()
        node_in_rel_tail_ents = torch.index_select(self.rel_tmap, dim=0, index=node_neigh_rel_inds.flatten()).view(bs, -1) #(bs, N * mxf)

        node_in_rel_tail_ents_pad_mx = (torch.ones((bs, N * self.max_freq), requires_grad=False) * self.ent_pad_id).cuda()
        node_in_rel_tail_mask_mx = torch.eq(node_in_rel_tail_ents, node_in_rel_tail_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.entity_dim*2)
        # (bs, N*self.mxf, dim)

        node_in_rel_tail_ent_embs = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_in_rel_tail_ents.flatten())).view(bs, N*self.max_freq, -1)

        node_in_rel_tail_ent_embs_masked = node_in_rel_tail_ent_embs.masked_fill_(node_in_rel_tail_mask_mx, 0)
        #(bs, N * self.mxf, dim)
        node_in_rel_tail_degree = torch.sum(self.rel_t_degree[node_neigh_rel_inds.flatten()].view(bs, N),
                                            dim=1, keepdim=True) #(bs, 1)

        node_hyper_tail_emb = torch.sum(node_in_rel_tail_ent_embs_masked, dim=1) / node_in_rel_tail_degree

        return node_hyper_tail_emb #(bs, d)

    def in2out1_hyper_agg(self, node_neigh_rel_inds): #ic의 variable node
        #(bs, 3)
        bs = node_neigh_rel_inds.size(0)

        node_in_rel_tail_ents = torch.index_select(self.rel_tmap, dim=0,
                                                   index=node_neigh_rel_inds[:, :2].flatten()).view(bs, 2 * self.max_freq)  #(bs, 2*mxf)
        node_in_rel_tail_ents_pad_mx = (torch.ones((bs, 2 * self.max_freq), requires_grad=False) * self.ent_pad_id).cuda() #(bs, 2*mxf)
        node_in_rel_tail_mask_mx = torch.eq(node_in_rel_tail_ents, node_in_rel_tail_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.entity_dim*2)

        node_in_rel_tail_ent_embs = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_in_rel_tail_ents.flatten())).view(bs, 2 * self.max_freq, -1)

        node_out_rel_head_ents = torch.index_select(self.rel_hmap, dim=0,
                                                    index=node_neigh_rel_inds[:, 2])  # (bs, max_freq)
        node_out_rel_head_ents_pad_mx = (torch.ones((bs, self.max_freq), requires_grad=False) * self.ent_pad_id).cuda()
        node_out_rel_head_mask_mx = torch.eq(node_out_rel_head_ents, node_out_rel_head_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.entity_dim*2)

        node_out_rel_head_ent_embs = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_out_rel_head_ents.flatten())).view(bs, self.max_freq, -1)

        node_in_rel_tail_ent_embs_masked = node_in_rel_tail_ent_embs.masked_fill_(node_in_rel_tail_mask_mx, 0) #(bs, 2*mxf, d)
        node_out_rel_head_ent_embs_masked = node_out_rel_head_ent_embs.masked_fill_(node_out_rel_head_mask_mx, 0) #(bs, mxf, d)


        node_in_rel_tail_degree = torch.sum(self.rel_t_degree[node_neigh_rel_inds[:, :2].flatten()].view(bs, 2),
                                            dim=1, keepdim=True)  # (bs, 1)
        node_out_rel_head_degree = self.rel_h_degree[node_neigh_rel_inds[:, 2]]  # (bs, 1)

        node_hyper_head_emb = torch.sum(node_out_rel_head_ent_embs_masked, dim=1) / node_out_rel_head_degree  # (bs, d)
        node_hyper_tail_emb = torch.sum(node_in_rel_tail_ent_embs_masked, dim=1) / node_in_rel_tail_degree  # (bs, d)

        node_hyper_emb = (node_hyper_head_emb + node_hyper_tail_emb) / 2

        return node_hyper_emb

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                     batch_idxs_dict)

    def embed_query_beta_iterative(self, queries, query_structure):
        qtype = structure2name[query_structure]
        structural_alpha_embedding = self.structural_alpha_mapping(self.query2onehot[structure2name[query_structure]]).unsqueeze(0)
        structural_beta_embedding = self.structural_beta_mapping(self.query2onehot[structure2name[query_structure]]).unsqueeze(0)

        if qtype in ['1p', '2p', '3p'] :
            rel_len = int(qtype[0])
            ########################################################################################
            var1_neigh_rel_inds = None if rel_len == 1 else queries[:, [1, 2]]
            var2_neigh_rel_inds = queries[:, [2, 3]] if rel_len == 3 else None
            ans_neigh_rel_inds = queries[:, [rel_len]]

            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding((var1_neigh_rel_inds, var2_neigh_rel_inds),
                                                                     ans_neigh_rel_inds,
                                                                     qtype)
            # 1c : (None, None), (bs, d)
            # 2c : ((bs, d), None), (bs, d)
            # 3c : ((bs, d), (bs, d)), (bs, d)
            ########################################################################################

            embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, 0]))

            for i in range(rel_len):
                if i == rel_len-1 :
                    role_alpha_embedding = self.role_alpha_embedding[2]
                    role_beta_embedding = self.role_beta_embedding[2]

                    hyper_alpha_embedding = ans_hyper_emb[:, :self.entity_dim]
                    hyper_beta_embedding = ans_hyper_emb[:, self.entity_dim:]
                else :
                    role_alpha_embedding = self.role_alpha_embedding[1]
                    role_beta_embedding = self.role_beta_embedding[1]

                    hyper_alpha_embedding = var_hyper_emb[i][:, :self.entity_dim]
                    hyper_beta_embedding = var_hyper_emb[i][:, self.entity_dim:]

                r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, i+1])

                embedding = self.projection_net(embedding, r_embedding)

                alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

                ################################################################################################
                positional_alpha_embedding = self.entity_regularizer(self.pos_alpha_embedding[i + 1].unsqueeze(0))
                positional_beta_embedding = self.entity_regularizer(self.pos_beta_embedding[i + 1].unsqueeze(0))

                role_alpha_embedding = self.entity_regularizer(role_alpha_embedding.unsqueeze(0))
                role_beta_embedding = self.entity_regularizer(role_beta_embedding.unsqueeze(0))

                structural_alpha_embedding = self.entity_regularizer(structural_alpha_embedding)
                structural_beta_embedding = self.entity_regularizer(structural_beta_embedding)

                structure_alpha_embeddings = [positional_alpha_embedding, role_alpha_embedding,
                                        structural_alpha_embedding]
                structure_beta_embeddings = [positional_beta_embedding, role_beta_embedding,
                                       structural_beta_embedding]

                if self.fusion_all:
                    structure_alpha_embeddings.append(hyper_alpha_embedding)
                    structure_beta_embeddings.append(hyper_beta_embedding)

                    alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embeddings)
                    beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embeddings)
                else :
                    alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embeddings, hyper_alpha_embedding)
                    beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embeddings, hyper_beta_embedding)

                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
                ################################################################################################

            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

        elif ('2i' in qtype) or ('3i' in qtype):
            ########################################################################################
            """
            var_neigh_rel_inds, ans_neigh_rel_inds
            2i,2in : (None,               (bs, 2))
            3i,3in : (None,               (bs, 3))
            """
            var_neigh_rel_inds = None
            ans_neigh_rel_inds = queries[:, [i * 2 + 1 for i in range(int(qtype[0]))]]
            _, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds,
                                                                     ans_neigh_rel_inds,
                                                                     qtype)
            ########################################################################################

            alpha_embedding_list = []
            beta_embedding_list = []

            for i in range(len(query_structure)):  # 2i, 2in -> i=0,1 | 3i, 3in -> i=0,1,2
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, 2*i]))

                r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, 2*i+1])

                embedding = self.projection_net(embedding, r_embedding)

                alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
                ########################################################################################################
                positional_alpha_embedding = self.entity_regularizer(self.pos_alpha_embedding[1].unsqueeze(0))
                positional_beta_embedding = self.entity_regularizer(self.pos_beta_embedding[1].unsqueeze(0))

                role_alpha_embedding = self.entity_regularizer(self.role_alpha_embedding[2].unsqueeze(0))
                role_beta_embedding = self.entity_regularizer(self.role_beta_embedding[2].unsqueeze(0))

                structural_alpha_embedding = self.entity_regularizer(structural_alpha_embedding)
                structural_beta_embedding = self.entity_regularizer(structural_beta_embedding)

                hyper_alpha_embedding = ans_hyper_emb[:, :self.entity_dim]
                hyper_beta_embedding = ans_hyper_emb[:, self.entity_dim:]

                structure_alpha_embedding = [positional_alpha_embedding, role_alpha_embedding,
                                        structural_alpha_embedding]
                structure_beta_embedding = [positional_beta_embedding, role_beta_embedding,
                                       structural_beta_embedding]
                if self.fusion_all :
                    structure_alpha_embedding.append(hyper_alpha_embedding)
                    structure_beta_embedding.append(hyper_beta_embedding)

                    alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embedding)
                    beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embedding)

                else :
                    alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embedding, hyper_alpha_embedding)
                    beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embedding, hyper_beta_embedding)

                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
                ########################################################################################################
                if 'n' in query_structure[i][-1]:
                    embedding = 1./embedding

                alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)

            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                              torch.stack(beta_embedding_list))


        elif qtype in ['ip', 'inp'] :
            alpha_embedding_list = []
            beta_embedding_list = []

            ##########################################################################################
            last_rel_idx = self.last_rel_idx[qtype]
            var_neigh_rel_inds = queries[:, [1, 3, last_rel_idx]]  # (bs, 3)
            ans_neigh_rel_inds = queries[:, [last_rel_idx]]  # (bs, 1)
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds,
                                                                     ans_neigh_rel_inds,
                                                                     qtype)
            ##########################################################################################
            for i in range(len(query_structure[0])):
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, i*2]))

                r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, i*2+1])

                embedding = self.projection_net(embedding, r_embedding)

                alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
                ########################################################################################################
                positional_alpha_embedding = self.entity_regularizer(self.pos_alpha_embedding[1].unsqueeze(0))
                positional_beta_embedding = self.entity_regularizer(self.pos_beta_embedding[1].unsqueeze(0))

                role_alpha_embedding = self.entity_regularizer(self.role_alpha_embedding[1].unsqueeze(0))
                role_beta_embedding = self.entity_regularizer(self.role_beta_embedding[1].unsqueeze(0))

                structural_alpha_embedding = self.entity_regularizer(structural_alpha_embedding)
                structural_beta_embedding = self.entity_regularizer(structural_beta_embedding)

                hyper_alpha_embedding = var_hyper_emb[:, :self.entity_dim]
                hyper_beta_embedding = var_hyper_emb[:, self.entity_dim:]

                structure_alpha_embeddings = [positional_alpha_embedding, role_alpha_embedding,
                                              structural_alpha_embedding]

                structure_beta_embeddings = [positional_beta_embedding, role_beta_embedding,
                                             structural_beta_embedding]

                if self.fusion_all :
                    structure_alpha_embeddings.append(hyper_alpha_embedding)
                    structure_beta_embeddings.append(hyper_beta_embedding)

                    alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embeddings)
                    beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embeddings)
                else :
                    alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embeddings, hyper_alpha_embedding)
                    beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embeddings, hyper_beta_embedding)

                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
                ########################################################################################################

                if 'n' in query_structure[0][i][-1]:
                    embedding = 1. / embedding

                alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)

            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                              torch.stack(beta_embedding_list))

            embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)

            r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, last_rel_idx])

            embedding = self.projection_net(embedding, r_embedding)

            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
            ########################################################################################################
            positional_alpha_embedding = self.entity_regularizer(self.pos_alpha_embedding[2].unsqueeze(0))
            positional_beta_embedding = self.entity_regularizer(self.pos_beta_embedding[2].unsqueeze(0))

            role_alpha_embedding = self.entity_regularizer(self.role_alpha_embedding[2].unsqueeze(0))
            role_beta_embedding = self.entity_regularizer(self.role_beta_embedding[2].unsqueeze(0))

            structural_alpha_embedding = self.entity_regularizer(structural_alpha_embedding)
            structural_beta_embedding = self.entity_regularizer(structural_beta_embedding)

            hyper_alpha_embedding = ans_hyper_emb[:, :self.entity_dim]
            hyper_beta_embedding = ans_hyper_emb[:, self.entity_dim:]

            structure_alpha_embeddings = [positional_alpha_embedding, role_alpha_embedding,
                                          structural_alpha_embedding]
            structure_beta_embeddings = [positional_beta_embedding, role_beta_embedding,
                                         structural_beta_embedding]

            if self.fusion_all :
                structure_alpha_embeddings.append(hyper_alpha_embedding)
                structure_beta_embeddings.append(hyper_beta_embedding)

                alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embeddings)
                beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embeddings)

            else :
                alpha_embedding = self.alpha_embedding_trans(alpha_embedding, structure_alpha_embeddings,
                                                             hyper_alpha_embedding)
                beta_embedding = self.beta_embedding_trans(beta_embedding, structure_beta_embeddings,
                                                           hyper_beta_embedding)



        elif qtype in ['pi', 'pni', 'pin'] :
            ############################################################################################################
            e2_ind, r2_ind = self.ent_rel_2_idx[qtype]
            var_neigh_rel_inds = queries[:, [1, 2]]  # (bs, 2)
            ans_neigh_rel_inds = queries[:, [2, r2_ind]]  # (bs, 2)
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds, ans_neigh_rel_inds, qtype)
            ############################################################################################################

            alpha_embedding_list = []
            beta_embedding_list = []

            #r11
            embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, 0]))

            r_embedding_11 = torch.index_select(self.relation_embedding, dim=0, index=queries[:, 1])

            embedding_11 = self.projection_net(embedding, r_embedding_11)

            alpha_embedding_11, beta_embedding_11 = torch.chunk(embedding_11, 2, dim=-1)

            ############################################################################################################
            positional_alpha_embedding = self.entity_regularizer(self.pos_alpha_embedding[1].unsqueeze(0))
            positional_beta_embedding = self.entity_regularizer(self.pos_beta_embedding[1].unsqueeze(0))

            role_alpha_embedding = self.entity_regularizer(self.role_alpha_embedding[1].unsqueeze(0))
            role_beta_embedding = self.entity_regularizer(self.role_beta_embedding[1].unsqueeze(0))

            structural_alpha_embedding = self.entity_regularizer(structural_alpha_embedding)
            structural_beta_embedding = self.entity_regularizer(structural_beta_embedding)

            hyper_alpha_embedding = var_hyper_emb[:, :self.entity_dim]
            hyper_beta_embedding = var_hyper_emb[:, self.entity_dim:]

            structure_alpha_embeddings = [positional_alpha_embedding, role_alpha_embedding,
                                          structural_alpha_embedding]
            structure_beta_embeddings = [positional_beta_embedding, role_beta_embedding,
                                         structural_beta_embedding]
            if self.fusion_all :
                structure_alpha_embeddings.append(hyper_alpha_embedding)
                structure_beta_embeddings.append(hyper_beta_embedding)

                alpha_embedding_11 = self.alpha_embedding_trans(alpha_embedding_11, structure_alpha_embeddings)
                beta_embedding_11 = self.beta_embedding_trans(beta_embedding_11, structure_beta_embeddings)

            else :
                alpha_embedding_11 = self.alpha_embedding_trans(alpha_embedding_11, structure_alpha_embeddings, hyper_alpha_embedding)
                beta_embedding_11 = self.beta_embedding_trans(beta_embedding_11, structure_beta_embeddings, hyper_beta_embedding)

            embedding_11 = torch.cat([alpha_embedding_11, beta_embedding_11], dim=-1)
            ############################################################################################################

            #r12
            r_embedding_12 = torch.index_select(self.relation_embedding, dim=0, index=queries[:, 2])

            embedding_12 = self.projection_net(embedding_11, r_embedding_12)

            alpha_embedding_12, beta_embedding_12 = torch.chunk(embedding_12, 2, dim=-1)
            ############################################################################################################
            positional_alpha_embedding = self.entity_regularizer(self.pos_alpha_embedding[2].unsqueeze(0))
            positional_beta_embedding = self.entity_regularizer(self.pos_beta_embedding[2].unsqueeze(0))

            role_alpha_embedding = self.entity_regularizer(self.role_alpha_embedding[2].unsqueeze(0))
            role_beta_embedding = self.entity_regularizer(self.role_beta_embedding[2].unsqueeze(0))

            structural_alpha_embedding = self.entity_regularizer(structural_alpha_embedding)
            structural_beta_embedding = self.entity_regularizer(structural_beta_embedding)

            hyper_alpha_embedding = ans_hyper_emb[:, :self.entity_dim]
            hyper_beta_embedding = ans_hyper_emb[:, self.entity_dim:]

            structure_alpha_embeddings = [positional_alpha_embedding, role_alpha_embedding,
                                          structural_alpha_embedding]
            structure_beta_embeddings = [positional_beta_embedding, role_beta_embedding,
                                         structural_beta_embedding]
            if self.fusion_all :
                structure_alpha_embeddings.append(hyper_alpha_embedding)
                structure_beta_embeddings.append(hyper_beta_embedding)

                alpha_embedding_12 = self.alpha_embedding_trans(alpha_embedding_12, structure_alpha_embeddings)
                beta_embedding_12 = self.beta_embedding_trans(beta_embedding_12, structure_beta_embeddings)

            else :
                alpha_embedding_12 = self.alpha_embedding_trans(alpha_embedding_12, structure_alpha_embeddings, hyper_alpha_embedding)
                beta_embedding_12 = self.beta_embedding_trans(beta_embedding_12, structure_beta_embeddings, hyper_beta_embedding)

            embedding_12 = torch.cat([alpha_embedding_12, beta_embedding_12], dim=-1)
            ############################################################################################################
            if qtype == 'pni':
                embedding_12 = 1. / embedding_12
            alpha_embedding_12, beta_embedding_12 = torch.chunk(embedding_12, 2, dim=-1)

            alpha_embedding_list.append(alpha_embedding_12)
            beta_embedding_list.append(beta_embedding_12)

            #branch2
            embedding_2 = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, e2_ind]))

            r_embedding_2 = torch.index_select(self.relation_embedding, dim=0, index=queries[:, r2_ind])

            embedding_2 = self.projection_net(embedding_2, r_embedding_2)

            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)
            ############################################################################################################
            positional_alpha_embedding = self.entity_regularizer(self.pos_alpha_embedding[2].unsqueeze(0))
            positional_beta_embedding = self.entity_regularizer(self.pos_beta_embedding[2].unsqueeze(0))

            role_alpha_embedding = self.entity_regularizer(self.role_alpha_embedding[2].unsqueeze(0))
            role_beta_embedding = self.entity_regularizer(self.role_beta_embedding[2].unsqueeze(0))

            structural_alpha_embedding = self.entity_regularizer(structural_alpha_embedding)
            structural_beta_embedding = self.entity_regularizer(structural_beta_embedding)

            hyper_alpha_embedding = ans_hyper_emb[:, :self.entity_dim]
            hyper_beta_embedding = ans_hyper_emb[:, self.entity_dim:]

            structure_alpha_embeddings = [positional_alpha_embedding, role_alpha_embedding,
                                          structural_alpha_embedding]
            structure_beta_embeddings = [positional_beta_embedding, role_beta_embedding,
                                         structural_beta_embedding]
            if self.fusion_all :
                structure_alpha_embeddings.append(hyper_alpha_embedding)
                structure_beta_embeddings.append(hyper_beta_embedding)

                alpha_embedding_2 = self.alpha_embedding_trans(alpha_embedding_2, structure_alpha_embeddings)
                beta_embedding_2 = self.beta_embedding_trans(beta_embedding_2, structure_beta_embeddings)
            else :
                alpha_embedding_2 = self.alpha_embedding_trans(alpha_embedding_2, structure_alpha_embeddings, hyper_alpha_embedding)
                beta_embedding_2 = self.beta_embedding_trans(beta_embedding_2, structure_beta_embeddings, hyper_beta_embedding)

            embedding_2 = torch.cat([alpha_embedding_2, beta_embedding_2], dim=-1)
            ############################################################################################################
            if qtype == 'pin':
                embedding_2 = 1. / embedding_2

            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)
            alpha_embedding_list.append(alpha_embedding_2)
            beta_embedding_list.append(beta_embedding_2)


            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                              torch.stack(beta_embedding_list))

        else :
            raise ValueError('invalid query type')

        return alpha_embedding, beta_embedding

    def embed_query_beta_orig(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(torch.index_select(self.ref_ent_emb, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta_orig(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding
                else:
                    r_embedding = torch.index_select(self.ref_rel_emb, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta_orig(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        #######################################################################################
        all_var_losses, all_org_alpha_embeddings, all_org_beta_embeddings = [], [], []
        variance_loss = 0
        #######################################################################################
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []

        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding = self.embed_query_beta_iterative(self.transform_union_query(batch_queries_dict[query_structure],
                                                                                                             query_structure),
                                                                                  self.transform_union_structure(query_structure))
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)

            else:
                alpha_embedding, beta_embedding = self.embed_query_beta_iterative(batch_queries_dict[query_structure], query_structure)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)
                if type(positive_sample) != None and self.var_lambda:
                    #####################################################################################################
                    org_alpha_embedding, org_beta_embedding, _ = self.embed_query_beta_orig(batch_queries_dict[query_structure],
                                                                                         query_structure, 0)

                    all_org_alpha_embeddings.append(org_alpha_embedding)
                    all_org_beta_embeddings.append(org_beta_embedding)
                    #####################################################################################################

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
            #####################################################################################################
            if len(all_org_alpha_embeddings) > 0 :
                all_org_alpha_embeddings = torch.cat(all_org_alpha_embeddings, dim=0).unsqueeze(1)
                all_org_beta_embeddings = torch.cat(all_org_beta_embeddings, dim=0).unsqueeze(1)

                trans_variance = (all_alpha_embeddings * all_beta_embeddings) / \
                                 (((all_alpha_embeddings + all_beta_embeddings) ** 2 + 10e-7) *
                                  (all_alpha_embeddings + all_beta_embeddings + 1 + 10e-7))

                orig_variance = (all_org_alpha_embeddings * all_org_beta_embeddings) / \
                                (((all_org_alpha_embeddings + all_org_beta_embeddings) ** 2 + 10e-7) *
                                 (all_org_alpha_embeddings + all_org_beta_embeddings + 1 + 10e-7))

                variance_loss = torch.norm((orig_variance - trans_variance), p=2, dim=-1)
            #####################################################################################################

        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0] // 2, 2, 1,
                                                                         -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0] // 2, 2, 1,
                                                                       -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[
                    all_idxs]  # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[
                    all_union_idxs]  # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(
                        1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(
                        batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(
                        batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        ##########################################################################################################
        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, variance_loss
        ##########################################################################################################

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(query_structure),
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(
                    self.transform_union_query(batch_queries_dict[query_structure],
                                               query_structure),
                    self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        #train_path_iterator, train_other_iterator
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        ################################################################################################################
        positive_logit, negative_logit, subsampling_weight, _, variance_loss = model(positive_sample, negative_sample,
                                                                                     subsampling_weight, batch_queries_dict,
                                                                                     batch_idxs_dict)
        ################################################################################################################

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()
        ##############################################################################################
        if model.var_lambda :
            variance_loss = variance_loss.sum() * model.var_lambda
        ##############################################################################################

        ##############################################################################################
        loss = (positive_sample_loss + negative_sample_loss + variance_loss) / 3
        ##############################################################################################
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'variance_loss': variance_loss.item() if model.var_lambda else 0,
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader,
                                                                                      disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                ################################################################################################################
                _, negative_logit, _, idxs, _ = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                ################################################################################################################

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort,
                                               model.batch_entity_range)  # achieve the ranking of all entities
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics