from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from structural_table import *

pi = 3.14159265358979323846


def convert_to_arg(x):   # 0 ~ pi 사이로 변환
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x, value_range=pi):   # -pi ~ pi 사이로 변환
    y = torch.tanh(x) * value_range
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale

class StructureHyperComb(nn.Module): #projection 후에 cascading error 줄이기
    # torch min will just return a single element in the tensor
    # min, max도 결국 cent 처럼 좌표값이므로 같은 lin transformation을 사용해야하지 않을까?
    def __init__(self, entity_dim, angle_scale, axis_scale=1.0, num_layer=0):
        super(StructureHyperComb, self).__init__()
        self.entity_dim = entity_dim
        self.num_layer = num_layer
        self.convert = True if num_layer else False
        self.angle_scale = angle_scale
        self.axis_scale = axis_scale

        value_range = pi / 6 if not num_layer else pi

        for i in range(1, 1 + num_layer):
            setattr(self, 'layer{}'.format(i), nn.Linear(self.entity_dim, self.entity_dim))
        if not self.num_layer :
            self.layer0 = lambda x : self.angle_scale(x, self.axis_scale)
        self.layer0 = nn.Linear(self.entity_dim, self.entity_dim)

    def forward(self, pos_emb, role_emb, type_emb, hyper_emb): #(1, dim)
        if self.convert :
            pos_emb = self.angle_scale(pos_emb, self.axis_scale)
            pos_emb = convert_to_axis(pos_emb)

            role_emb = self.angle_scale(role_emb, self.axis_scale)
            role_emb = convert_to_axis(role_emb)

            type_emb = self.angle_scale(type_emb, self.axis_scale)
            type_emb =convert_to_axis(type_emb)

        x = pos_emb + role_emb + type_emb + hyper_emb

        for i in range(1, 1 + self.num_layer,):
            x = F.relu(getattr(self, 'layer{}'.format(i)))
        x = self.layer0(x)

        return x

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,

                 transform = True,
                 rel_hmap=None, rel_tmap=False,
                 rel_h_degree=None, rel_t_degree=None,
                 max_qry_len=4, num_role=3, structure_dim=108,
                 PE=False, RE=False, SE=False, table_normalization=False,
                 value_range=pi/6, regularization=0, num_layer=0,


                 test_batch_size=1, use_cuda=False, query_name_dict=None,
                 center_reg=None, drop=0.):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.cen = center_reg

        # entity only have axis but no arg
        ##########################################################################################
        """
        entity embedding은 padding 위해서 nentity+1 만큼
        initialization은 [:-1] 까지
        """
        if rel_hmap is not None:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity + 1, self.entity_dim),
                                                 requires_grad=True)  # axis for entities
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim),
                                                 requires_grad=True)  # axis for entities
        ##########################################################################################
        self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]

        self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

        self.axis_scale = 1.0
        self.arg_scale = 1.0

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        ##########################################################################################
        self.value_range = value_range
        self.regularization=regularization

        self.max_qry_len = max_qry_len
        self.num_role = num_role
        self.structure_dim = structure_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.table_normalization = True
        self.PE = PE
        self.RE = RE
        self.SE = SE


        self.transform = transform
        self.Hyper = False
        if rel_hmap is not None:
            self.Hyper = True
            self.hyper_dim = hidden_dim
            assert rel_tmap is not None and rel_h_degree is not None and rel_t_degree is not None

            self.rel_hmap = nn.Parameter(torch.tensor(rel_hmap), requires_grad=False).to('cuda')
            self.rel_tmap = nn.Parameter(torch.tensor(rel_tmap), requires_grad=False).to('cuda')
            self.max_freq = self.rel_hmap.size(1)
            self.rel_h_degree = nn.Parameter(torch.tensor(rel_h_degree), requires_grad=False).unsqueeze(1).to('cuda')
            self.rel_t_degree = nn.Parameter(torch.tensor(rel_t_degree), requires_grad=False).unsqueeze(1).to('cuda')

            self.ent_pad_id = nentity
            self.last_rel_idx = {'ip' : 4, 'inp' : 5}
            self.ent_rel_2_idx = {'pi': [3, 4], 'pin': [3, 4], 'pni': [4, 5]}

            logging.info('head of relation map : {}'.format(self.rel_hmap.size()))
            logging.info('tail of relation map : {}'.format(self.rel_tmap.size()))
            logging.info('head degree of relation : {}'.format(self.rel_h_degree.size()))
            logging.info('tail degree of relation : {}'.format(self.rel_t_degree.size()))
        else :
            self.hyper_dim = None

        """
        rel_hmap : kg_triple에서 r의 head가 되는 애들
        rel_tmap : kg_triple에서 r의 tail이 되는 애들
        """
        self.init_role_embedding()
        self.init_position_embedding()
        self.init_structure_embedding()

        self.embedding_trans = StructureHyperComb(entity_dim=self.entity_dim, angle_scale=self.angle_scale,
                                                  num_layer=num_layer)
        # self.embedding_trans = HyperComb(entity_dim=self.entity_dim)

        # logging.info('Use position embedding : {}'.format(True if self.pos0_embedding is not None else False))
        # logging.info('Use role embedding : {}'.format(True if self.variable_role_embedding is not None else False))
        # logging.info('Use structure embedding : {}'.format(self.SE))

        ##########################################################################################

        self.cone_proj = ConeProjection(self.entity_dim, 1600, 2)
        self.cone_intersection = ConeIntersection(self.entity_dim, drop)
        self.cone_negation = ConeNegation()

    def init_position_embedding(self):
        if self.PE:
            self.pos0_embedding = nn.Parameter(torch.zeros(1, self.structure_dim))
            self.pos1_embedding = nn.Parameter(torch.zeros(1, self.structure_dim))
            self.pos2_embedding = nn.Parameter(torch.zeros(1, self.structure_dim))
            self.pos3_embedding = nn.Parameter(torch.zeros(1, self.structure_dim))
            nn.init.uniform_(tensor=self.pos0_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.pos1_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.pos2_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.pos3_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
        else:
            self.pos0_embedding = None
            self.pos1_embedding = None
            self.pos2_embedding = None
            self.pos3_embedding = None

        self.rel2posemb = {0: self.pos0_embedding, 1: self.pos1_embedding, 2: self.pos2_embedding,
                           3: self.pos3_embedding}

    def init_role_embedding(self):
        if self.RE:
            self.anchor_role_embedding = nn.Parameter(torch.zeros(1, self.structure_dim))
            self.variable_role_embedding = nn.Parameter(torch.zeros(1, self.structure_dim))
            self.answer_role_embedding = nn.Parameter(torch.zeros(1, self.structure_dim))
            nn.init.uniform_(tensor=self.anchor_role_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.variable_role_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.answer_role_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
        else:
            self.anchor_role_embedding = None
            self.variable_role_embedding = None
            self.answer_role_embedding = None

    def init_structure_embedding(self):
        if self.SE:
            self.query2onehot = get_query2onehot(self.device, self.table_normalization)
            self.structural_embedding = nn.Linear(self.max_qry_len * self.num_role, self.structure_dim)

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
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds) #(bs, 1)

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
        node_in_rel_tail_mask_mx = torch.eq(node_in_rel_tail_ents, node_in_rel_tail_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.hidden_dim)

        node_out_rel_head_ents = torch.index_select(self.rel_hmap, dim=0, index=node_neigh_rel_inds[:, 1])  # (bs, mxf)
        node_out_rel_head_ents_pad_mx = (torch.ones((bs, self.max_freq), requires_grad=False) * self.ent_pad_id).cuda()
        node_out_rel_head_mask_mx = torch.eq(node_out_rel_head_ents, node_out_rel_head_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.hidden_dim)

        node_in_rel_tail_ent_embs = torch.index_select(self.entity_embedding, dim=0,
                                                      index=node_in_rel_tail_ents.flatten()).view(bs, self.max_freq, -1)
        node_in_rel_tail_ent_embs_masked = node_in_rel_tail_ent_embs.masked_fill_(node_in_rel_tail_mask_mx, 0) #(bs, mxf, d)


        node_out_rel_head_ent_embs = torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_out_rel_head_ents.flatten()).view(bs, self.max_freq, -1)
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
        node_in_rel_tail_mask_mx = torch.eq(node_in_rel_tail_ents, node_in_rel_tail_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        # (bs, N*self.mxf, dim)

        node_in_rel_tail_ent_embs = torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_in_rel_tail_ents.flatten()).view(bs, N*self.max_freq, -1)

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
        node_in_rel_tail_mask_mx = torch.eq(node_in_rel_tail_ents, node_in_rel_tail_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.hidden_dim)

        node_in_rel_tail_ent_embs = torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_in_rel_tail_ents.flatten()).view(bs, 2 * self.max_freq, -1)

        node_out_rel_head_ents = torch.index_select(self.rel_hmap, dim=0,
                                                    index=node_neigh_rel_inds[:, 2])  # (bs, max_freq)
        node_out_rel_head_ents_pad_mx = (torch.ones((bs, self.max_freq), requires_grad=False) * self.ent_pad_id).cuda()
        node_out_rel_head_mask_mx = torch.eq(node_out_rel_head_ents, node_out_rel_head_ents_pad_mx).unsqueeze(-1).repeat(1, 1, self.hidden_dim)

        node_out_rel_head_ent_embs = torch.index_select(self.entity_embedding, dim=0,
                                                       index=node_out_rel_head_ents.flatten()).view(bs, self.max_freq, -1)

        node_in_rel_tail_ent_embs_masked = node_in_rel_tail_ent_embs.masked_fill_(node_in_rel_tail_mask_mx, 0) #(bs, 2*mxf, d)
        node_out_rel_head_ent_embs_masked = node_out_rel_head_ent_embs.masked_fill_(node_out_rel_head_mask_mx, 0) #(bs, mxf, d)


        node_in_rel_tail_degree = torch.sum(self.rel_t_degree[node_neigh_rel_inds[:, :2].flatten()].view(bs, 2),
                                            dim=1, keepdim=True)  # (bs, 1)
        node_out_rel_head_degree = self.rel_h_degree[node_neigh_rel_inds[:, 2]]  # (bs, 1)

        node_hyper_head_emb = torch.sum(node_out_rel_head_ent_embs_masked, dim=1) / node_out_rel_head_degree  # (bs, d)
        node_hyper_tail_emb = torch.sum(node_in_rel_tail_ent_embs_masked, dim=1) / node_in_rel_tail_degree  # (bs, d)

        node_hyper_emb = (node_hyper_head_emb + node_hyper_tail_emb) / 2

        return node_hyper_emb

    def transform_union_query(self, queries, query_structure):
        """
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        """
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return 'e', ('r',)
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return 'e', ('r', 'r')

    def train_step(self, model, optimizer, train_iterator, args, step):
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

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()


        loss = (positive_sample_loss + negative_sample_loss) / 2

        ################################################################################################################
        regularization = model.regularization * (
                model.entity_embedding.norm(p=3) ** 3 + model.axis_embedding.norm(p=3) ** 3 +
                model.arg_embedding.norm(p=3) ** 3)
        loss += regularization
        ################################################################################################################

        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            ################################################################################################################
            'regularization_loss': regularization.item(),
            ################################################################################################################
            'loss': loss.item(),
        }
        return log

    def test_step(self, model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size:
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range)
                else:
                    if args.cuda:
                        ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).cuda())
                    else:
                        ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1))

                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
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
                    })
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics

    def embed_query_cone_iterative(self, queries, query_structure):
        qtype = structure2name[query_structure]

        type_embedding = (self.structural_embedding(self.query2onehot[structure2name[query_structure]])).unsqueeze(0)

        if qtype in ['1p', '2p', '3p'] :
            rel_len = int(qtype[0])
            ########################################################################################
            var1_neigh_rel_inds = None if rel_len == 1 else queries[:, [1, 2]]
            var2_neigh_rel_inds = queries[:, [2, 3]] if rel_len == 3 else None
            ans_neigh_rel_inds = queries[:, [rel_len]]
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding((var1_neigh_rel_inds, var2_neigh_rel_inds),
                                                                     ans_neigh_rel_inds,
                                                                     qtype)
            ########################################################################################
            axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, 0])
            axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
            axis_entity_embedding = convert_to_axis(axis_entity_embedding)

            if self.use_cuda:
                arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
            else:
                arg_entity_embedding = torch.zeros_like(axis_entity_embedding)

            axis_embedding = axis_entity_embedding
            arg_embedding = arg_entity_embedding

            for i in range(rel_len):
                if i == rel_len - 1 :
                    hyper_embedding = ans_hyper_emb
                    role_embedding = self.answer_role_embedding
                else :
                    hyper_embedding = var_hyper_emb[i]
                    role_embedding = self.variable_role_embedding
                pos_embedding = self.rel2posemb[i + 1]

                axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, i+1])
                arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, i+1])

                axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

                axis_r_embedding = convert_to_axis(axis_r_embedding)
                arg_r_embedding = convert_to_axis(arg_r_embedding)

                axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding,
                                                               arg_r_embedding)
                #emb, pos_emb=None, rol_emb=None, strc_emb=None, hyper_emb=None
                ########################################################################################################
                calib_embedding = self.embedding_trans(pos_embedding, role_embedding, type_embedding, hyper_embedding)
                calib_axis_embedding = convert_to_axis(calib_embedding, value_range=self.value_range)
                axis_embedding += calib_axis_embedding
                axis_embedding = torch.clamp(axis_embedding, min=-pi, max=pi)
                ########################################################################################################
        # 1p, 2p, 3p

        elif ('2i' in qtype) or ('3i' in qtype) :
            ########################################################################################
            var_neigh_rel_inds = None
            ans_neigh_rel_inds = queries[:, [i*2+1 for i in range(int(qtype[0]))]]
            _, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds,
                                                                     ans_neigh_rel_inds,
                                                                     qtype)
            ########################################################################################
            axis_embedding_list = []
            arg_embedding_list = []

            for i in range(len(query_structure)): #2i, 2in -> i=0,1 | 3i, 3in -> i=0,1,2
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, i*2])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding

                axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, i*2+1])
                arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, i*2+1])

                axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

                axis_r_embedding = convert_to_axis(axis_r_embedding)
                arg_r_embedding = convert_to_axis(arg_r_embedding)

                axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding,
                                                               arg_r_embedding)
                # emb, pos_emb=None, rol_emb=None, strc_emb=None, hyper_emb=None
                ########################################################################################################
                pos_embedding = self.rel2posemb[1]
                role_embedding = self.answer_role_embedding
                hyper_embedding = ans_hyper_emb

                calib_embedding = self.embedding_trans(pos_embedding, role_embedding, type_embedding, hyper_embedding)
                calib_axis_embedding = convert_to_axis(calib_embedding, value_range=self.value_range)
                axis_embedding += calib_axis_embedding
                axis_embedding = torch.clamp(axis_embedding, min=-pi, max=pi)
                ########################################################################################################
                """
                2i : (('e', ('r',)), ('e', ('r',)))
                3i : (('e', ('r',)), ('e', ('r',)), ('e', ('r',)))
                2in : (('e', ('r',)), ('e', ('r', 'n')))
                3in : (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
                """
                if 'n' in query_structure[i][-1] :
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)

                axis_embedding_list.append(axis_embedding) #(bs, d)
                arg_embedding_list.append(arg_embedding)   #(bs, d)

            stacked_axis_embeddings = torch.stack(axis_embedding_list) #(num_inter, bs, d)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)  # (num_inter, bs, d)
            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)
        # 2i, 2in, 3i, 3in

        elif qtype in ['ip', 'inp'] :
            """
            ip : [[657, 63, 8204, 12, 62]] -> [((('e', ('r',)), ('e', ('r',))), ('r',))] -> len : 2
            inp : [[356, 39, 540, 79, -2, 346]] -> [((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] -> len : 2
            """
            axis_embedding_list = []
            arg_embedding_list = []
            last_rel_idx = self.last_rel_idx[qtype]
            ##########################################################################################
            var_neigh_rel_inds = queries[:, [1, 3, last_rel_idx]]  # (bs, 3)
            ans_neigh_rel_inds = queries[:, [last_rel_idx]]  # (bs, 1)
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds,
                                                                     ans_neigh_rel_inds,
                                                                     qtype)
            ##########################################################################################
            for i in range(len(query_structure[0])):
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, i*2])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding

                axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, i*2+1])
                arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, i*2+1])

                axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

                axis_r_embedding = convert_to_axis(axis_r_embedding)
                arg_r_embedding = convert_to_axis(arg_r_embedding)

                axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding,
                                                               arg_r_embedding)
                # emb, pos_emb=None, rol_emb=None, strc_emb=None, hyper_emb=None
                ########################################################################################################
                pos_embedding = self.rel2posemb[1]
                role_embedding = self.variable_role_embedding
                hyper_embedding = var_hyper_emb

                calib_embedding = self.embedding_trans(pos_embedding, role_embedding, type_embedding, hyper_embedding)
                calib_axis_embedding = convert_to_axis(calib_embedding, value_range=self.value_range)
                axis_embedding += calib_axis_embedding
                axis_embedding = torch.clamp(axis_embedding, min=-pi, max=pi)
                ########################################################################################################

                if 'n' in query_structure[0][i][-1]:
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)

                axis_embedding_list.append(axis_embedding)  # (bs, d)
                arg_embedding_list.append(arg_embedding)  # (bs, d)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)  # (num_inter, bs, d)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)  # (num_inter, bs, d)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings,
                                                                   stacked_arg_embeddings)

            ########################### last projection of ip ##########################################
            #
            axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, last_rel_idx])
            arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, last_rel_idx])

            axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
            arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

            axis_r_embedding = convert_to_axis(axis_r_embedding)
            arg_r_embedding = convert_to_axis(arg_r_embedding)
            axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding,
                                                           arg_r_embedding)

            ############################################################################################################
            pos_embedding = self.rel2posemb[2]
            role_embedding = self.answer_role_embedding
            hyper_embedding = ans_hyper_emb

            calib_embedding = self.embedding_trans(pos_embedding, role_embedding, type_embedding, hyper_embedding)
            calib_axis_embedding = convert_to_axis(calib_embedding, value_range=self.value_range)
            axis_embedding += calib_axis_embedding
            axis_embedding = torch.clamp(axis_embedding, min=-pi, max=pi)
            ############################################################################################################
        # ip, inp

        elif qtype in ['pi', 'pni', 'pin'] :
            ############################################################################################################
            #self.ent_rel_2_idx = {'pi': [3, 4], 'pin': [3, 4], 'pni': [4, 5]}
            e2_ind, r2_ind = self.ent_rel_2_idx[qtype]

            var_neigh_rel_inds = queries[:, [1, 2]]  # (bs, 2)
            ans_neigh_rel_inds = queries[:, [2, r2_ind]]  # (bs, 2)
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds, ans_neigh_rel_inds, qtype)
            ############################################################################################################
            # (bs, d), (bs, d)
            # branch 1

            axis_embedding_list = []
            arg_embedding_list = []

            #branch1
            # r11
            axis_entity_embedding_1 = torch.index_select(self.entity_embedding, dim=0, index=queries[:, 0])
            axis_entity_embedding_1 = self.angle_scale(axis_entity_embedding_1, self.axis_scale)
            axis_entity_embedding_1 = convert_to_axis(axis_entity_embedding_1)
            if self.use_cuda:
                arg_entity_embedding_1 = torch.zeros_like(axis_entity_embedding_1).cuda()
            else:
                arg_entity_embedding_1 = torch.zeros_like(axis_entity_embedding_1)

            axis_embedding_1 = axis_entity_embedding_1
            arg_embedding_1 = arg_entity_embedding_1

            axis_r_embedding_11 = torch.index_select(self.axis_embedding, dim=0, index=queries[:, 1])
            arg_r_embedding_11 = torch.index_select(self.arg_embedding, dim=0, index=queries[:, 1])

            axis_r_embedding_11 = self.angle_scale(axis_r_embedding_11, self.axis_scale)
            arg_r_embedding_11 = self.angle_scale(arg_r_embedding_11, self.arg_scale)

            axis_r_embedding_11 = convert_to_axis(axis_r_embedding_11)
            arg_r_embedding_11 = convert_to_axis(arg_r_embedding_11)

            axis_embedding_11, arg_embedding_11 = self.cone_proj(axis_embedding_1, arg_embedding_1, axis_r_embedding_11,
                                                                 arg_r_embedding_11)

            ############################################################################################################
            pos_embedding = self.rel2posemb[1]
            role_embedding = self.variable_role_embedding
            hyper_embedding = var_hyper_emb

            calib_embedding_11 = self.embedding_trans(pos_embedding, role_embedding, type_embedding, hyper_embedding)
            calib_axis_embedding_11 = convert_to_axis(calib_embedding_11, value_range=self.value_range)
            axis_embedding_11 += calib_axis_embedding_11
            axis_embedding_11 = torch.clamp(axis_embedding_11, min=-pi, max=pi)
            ############################################################################################################


            # r12
            axis_r_embedding_12 = torch.index_select(self.axis_embedding, dim=0, index=queries[:, 2])
            arg_r_embedding_12 = torch.index_select(self.arg_embedding, dim=0, index=queries[:, 2])

            axis_r_embedding_12 = self.angle_scale(axis_r_embedding_12, self.axis_scale)
            arg_r_embedding_12 = self.angle_scale(arg_r_embedding_12, self.arg_scale)

            axis_r_embedding_12 = convert_to_axis(axis_r_embedding_12)
            arg_r_embedding_12 = convert_to_axis(arg_r_embedding_12)

            axis_embedding_12, arg_embedding_12 = self.cone_proj(axis_embedding_11, arg_embedding_11, axis_r_embedding_12,
                                                                 arg_r_embedding_12)
            ############################################################################################################
            pos_embedding = self.rel2posemb[2]
            role_embedding = self.answer_role_embedding
            hyper_embedding = ans_hyper_emb

            calib_embedding_12 = self.embedding_trans(pos_embedding, role_embedding, type_embedding, hyper_embedding)
            calib_axis_embedding_12 = convert_to_axis(calib_embedding_12, value_range=self.value_range)
            axis_embedding_12 += calib_axis_embedding_12
            axis_embedding_12 = torch.clamp(axis_embedding_12, min=-pi, max=pi)
            ############################################################################################################

            if qtype == 'pni' :
                axis_embedding_12, arg_embedding_12 = self.cone_negation(axis_embedding_12, arg_embedding_12)

            axis_embedding_list.append(axis_embedding_12)
            arg_embedding_list.append(arg_embedding_12)

            #branch2
            axis_entity_embedding_2 = torch.index_select(self.entity_embedding, dim=0, index=queries[:, e2_ind])
            axis_entity_embedding_2 = self.angle_scale(axis_entity_embedding_2, self.axis_scale)
            axis_entity_embedding_2 = convert_to_axis(axis_entity_embedding_2)
            if self.use_cuda:
                arg_entity_embedding_2 = torch.zeros_like(axis_entity_embedding_2).cuda()
            else:
                arg_entity_embedding_2 = torch.zeros_like(axis_entity_embedding_2)

            axis_embedding_2 = axis_entity_embedding_2
            arg_embedding_2 = arg_entity_embedding_2

            axis_r_embedding_2 = torch.index_select(self.axis_embedding, dim=0, index=queries[:, r2_ind])
            arg_r_embedding_2 = torch.index_select(self.arg_embedding, dim=0, index=queries[:, r2_ind])

            axis_r_embedding_2 = self.angle_scale(axis_r_embedding_2, self.axis_scale)
            arg_r_embedding_2 = self.angle_scale(arg_r_embedding_2, self.arg_scale)

            axis_r_embedding_2 = convert_to_axis(axis_r_embedding_2)
            arg_r_embedding_2 = convert_to_axis(arg_r_embedding_2)

            axis_embedding_2, arg_embedding_2 = self.cone_proj(axis_embedding_2, arg_embedding_2, axis_r_embedding_2,
                                                           arg_r_embedding_2)

            ############################################################################################################
            pos_embedding = self.rel2posemb[2]
            role_embedding = self.answer_role_embedding
            hyper_embedding = ans_hyper_emb

            calib_embedding_2 = self.embedding_trans(pos_embedding, role_embedding, type_embedding, hyper_embedding)
            calib_axis_embedding_2 = convert_to_axis(calib_embedding_2, value_range=self.value_range)
            axis_embedding_2 += calib_axis_embedding_2
            axis_embedding_2 = torch.clamp(axis_embedding_2, min=-pi, max=pi)
            ############################################################################################################

            if qtype == 'pin' :
                axis_embedding_2, arg_embedding_2 = self.cone_negation(axis_embedding_2, arg_embedding_2)

            axis_embedding_list.append(axis_embedding_2)
            arg_embedding_list.append(arg_embedding_2)
            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)
        #pi, pni, pin

        else :
            raise ValueError('Invalid qtype {}'.format(qtype))

        return axis_embedding, arg_embedding




    # implement distance function
    def cal_logit_cone(self, entity_embedding, query_axis_embedding, query_arg_embedding):
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        logit = self.gamma - distance * self.modulus

        return logit

    # implement formatting forward method
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_axis_embeddings, all_arg_embeddings = [], [], []
        all_union_idxs, all_union_axis_embeddings, all_union_arg_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis_embedding, arg_embedding = self.embed_query_cone_iterative(self.transform_union_query(batch_queries_dict[query_structure],
                                                                               query_structure),
                                                                                self.transform_union_structure(query_structure))
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
            else:
                axis_embedding, arg_embedding = self.embed_query_cone_iterative(batch_queries_dict[query_structure],
                                                                                query_structure)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)

        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(all_arg_embeddings, dim=0).unsqueeze(1)
        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_logit = self.cal_logit_cone(positive_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)


            if len(all_union_axis_embeddings) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_union_logit = self.cal_logit_cone(positive_embedding, all_union_axis_embeddings, all_union_arg_embeddings)

                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_union_logit = self.cal_logit_cone(negative_embedding, all_union_axis_embeddings, all_union_arg_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs



class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        return axis_embeddings, arg_embeddings


class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_arg1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings):
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))
        axis_attention = F.softmax(self.drop(self.layer_axis2(axis_layer1_act)), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        # when x_embeddings are very closed to zero, the tangent may be nan
        # no need to consider the sign of x_embeddings
        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        # DeepSets
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        return axis_embeddings, arg_embeddings


class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding
