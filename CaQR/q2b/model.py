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
from dataloader import *
import random
import pickle
import math
from structural_table import *


def Identity(x):
    return x

class SetIntersection(nn.Module):
    def __init__(self, mode_dims, expand_dims, agg_func=torch.min):
        super(SetIntersection, self).__init__()
        self.agg_func = agg_func
        self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat", self.pre_mats)
        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat", self.post_mats)
        self.pre_mats_im = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats_im)
        self.register_parameter("premat_im", self.pre_mats_im)
        self.post_mats_im = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats_im)
        self.register_parameter("postmat_im", self.post_mats_im)

    def forward(self, embeds1, embeds2, embeds3=[], name='real'):
        if name == 'real':
            temp1 = F.relu(embeds1.mm(self.pre_mats))
            temp2 = F.relu(embeds2.mm(self.pre_mats))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats)

        elif name == 'img':
            temp1 = F.relu(embeds1.mm(self.pre_mats_im))
            temp2 = F.relu(embeds2.mm(self.pre_mats_im))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats_im))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats_im)
        return combined

class CenterSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, agg_func=torch.min, bn='no', nat=1,
                 name='Real_center'):
        super(CenterSet, self).__init__()
        assert nat == 1, 'vanilla method only support 1 nat now'
        self.center_use_offset = center_use_offset
        self.agg_func = agg_func
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))

        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat_%s" % name, self.pre_mats)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn2 = nn.BatchNorm1d(mode_dims)
            self.bn3 = nn.BatchNorm1d(mode_dims)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s" % name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if self.center_use_offset:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1
            temp2 = embeds2
            if len(embeds3) > 0:
                temp3 = embeds3

        if self.bn == 'no':
            temp1 = F.relu(temp1.mm(self.pre_mats))
            temp2 = F.relu(temp2.mm(self.pre_mats))
        elif self.bn == 'before':
            temp1 = F.relu(self.bn1(temp1.mm(self.pre_mats)))
            temp2 = F.relu(self.bn2(temp2.mm(self.pre_mats)))
        elif self.bn == 'after':
            temp1 = self.bn1(F.relu(temp1.mm(self.pre_mats)))
            temp2 = self.bn2(F.relu(temp2.mm(self.pre_mats)))
        if len(embeds3) > 0:
            if self.bn == 'no':
                temp3 = F.relu(temp3.mm(self.pre_mats))
            elif self.bn == 'before':
                temp3 = F.relu(self.bn3(temp3.mm(self.pre_mats)))
            elif self.bn == 'after':
                temp3 = self.bn3(F.relu(temp3.mm(self.pre_mats)))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined

class MeanSet(nn.Module):
    def __init__(self):
        super(MeanSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.mean(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)
        else:
            return torch.mean(torch.stack([embeds1, embeds2], dim=0), dim=0)

class MinSet(nn.Module):
    def __init__(self):
        super(MinSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.min(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)[0]
        else:
            return torch.min(torch.stack([embeds1, embeds2], dim=0), dim=0)[0]

class OffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, agg_func=torch.min, name='Real_offset'):
        super(OffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        if offset_use_center:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s" % name, self.pre_mats)
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s" % name, self.pre_mats)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s" % name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if self.offset_use_center:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3_o) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1_o
            temp2 = embeds2_o
            if len(embeds3_o) > 0:
                temp3 = embeds3_o
        temp1 = F.relu(temp1.mm(self.pre_mats))
        temp2 = F.relu(temp2.mm(self.pre_mats))
        if len(embeds3_o) > 0:
            temp3 = F.relu(temp3.mm(self.pre_mats))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined

class InductiveOffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, off_reg, agg_func=torch.min, name='Real_offset'):
        super(InductiveOffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.OffsetSet_Module = OffsetSet(mode_dims, expand_dims, offset_use_center, self.agg_func)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if len(embeds3_o) > 0:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o, embeds3_o]), dim=0)[0]
        else:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o]), dim=0)[0]
        offset = offset_min * torch.sigmoid(
            self.OffsetSet_Module(embeds1, embeds1_o, embeds2, embeds2_o, embeds3, embeds3_o))
        return offset

class AttentionSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_reg=0., att_tem=1., att_type="whole", bn='no',
                 nat=1, name="Real"):
        super(AttentionSet, self).__init__()
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.Attention_module = Attention(mode_dims, expand_dims, center_use_offset, att_type=att_type, bn=bn, nat=nat)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        temp1 = (self.Attention_module(embeds1, embeds1_o) + self.att_reg) / (self.att_tem + 1e-4)
        temp2 = (self.Attention_module(embeds2, embeds2_o) + self.att_reg) / (self.att_tem + 1e-4)
        if len(embeds3) > 0:
            temp3 = (self.Attention_module(embeds3, embeds3_o) + self.att_reg) / (self.att_tem + 1e-4)
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2, temp3], dim=1), dim=1)
                center = embeds1 * (combined[:, 0].view(embeds1.size(0), 1)) + \
                         embeds2 * (combined[:, 1].view(embeds2.size(0), 1)) + \
                         embeds3 * (combined[:, 2].view(embeds3.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2, temp3]), dim=0)
                center = embeds1 * combined[0] + embeds2 * combined[1] + embeds3 * combined[2]
        else:
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2], dim=1), dim=1)
                center = embeds1 * (combined[:, 0].view(embeds1.size(0), 1)) + \
                         embeds2 * (combined[:, 1].view(embeds2.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2]), dim=0)
                center = embeds1 * combined[0] + embeds2 * combined[1]

        return center

class Attention(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_type, bn, nat, name="Real"):
        super(Attention, self).__init__()
        self.center_use_offset = center_use_offset
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
        else:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats1)
        self.register_parameter("atten_mats1_%s" % name, self.atten_mats1)
        if self.nat >= 2:
            self.atten_mats1_1 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_1)
            self.register_parameter("atten_mats1_1_%s" % name, self.atten_mats1_1)
        if self.nat >= 3:
            self.atten_mats1_2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_2)
            self.register_parameter("atten_mats1_2_%s" % name, self.atten_mats1_2)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn1_1 = nn.BatchNorm1d(mode_dims)
            self.bn1_2 = nn.BatchNorm1d(mode_dims)
        if att_type == 'whole':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, 1))
        elif att_type == 'ele':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats2)
        self.register_parameter("atten_mats2_%s" % name, self.atten_mats2)

    def forward(self, center_embed, offset_embed=None):
        if self.center_use_offset:
            temp1 = torch.cat([center_embed, offset_embed], dim=1)
        else:
            temp1 = center_embed
        if self.nat >= 1:
            if self.bn == 'no':
                temp2 = F.relu(temp1.mm(self.atten_mats1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1(temp1.mm(self.atten_mats1)))
            elif self.bn == 'after':
                temp2 = self.bn1(F.relu(temp1.mm(self.atten_mats1)))
        if self.nat >= 2:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_1(temp2.mm(self.atten_mats1_1)))
            elif self.bn == 'after':
                temp2 = self.bn1_1(F.relu(temp2.mm(self.atten_mats1_1)))
        if self.nat >= 3:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_2))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_2(temp2.mm(self.atten_mats1_2)))
            elif self.bn == 'after':
                temp2 = self.bn1_2(F.relu(temp2.mm(self.atten_mats1_2)))
        temp3 = temp2.mm(self.atten_mats2)
        return temp3

class MultilayerNN_MinMax(nn.Module): #projection 후에 cascading error 줄이기
    # torch min will just return a single element in the tensor
    def __init__(self, center_dim, offset_dim=None, pos_dim=None, rol_dim=None, strc_dim=None):
        super(MultilayerNN_MinMax, self).__init__()
        self.center_dim = center_dim
        self.offset_dim = offset_dim
        self.pos_dim = pos_dim
        self.rol_dim = rol_dim
        self.strc_dim = strc_dim
        if self.offset_dim is not None :
            assert self.center_dim == self.offset_dim
        #
        expand_dim = center_dim * 2
        pos_expand_dim = pos_dim * 2 if pos_dim else 0
        rol_expand_dim = rol_dim * 2 if rol_dim else 0
        strc_expand_dim = strc_dim * 2 if strc_dim else 0
        post_dim = expand_dim * 3 + pos_expand_dim + rol_expand_dim + strc_expand_dim

        self.mats1_center = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform_(self.mats1_center)
        self.register_parameter("mats1_center", self.mats1_center)

        self.post_mats_center = nn.Parameter(torch.FloatTensor(post_dim, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_center)
        self.register_parameter("post_mats_center", self.post_mats_center)

        if offset_dim is not None :
            self.mats1_min = nn.Parameter(torch.FloatTensor(offset_dim, expand_dim))
            nn.init.xavier_uniform_(self.mats1_min)
            self.register_parameter("mats1_min", self.mats1_min)

            self.post_mats_min = nn.Parameter(torch.FloatTensor(post_dim, offset_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_min)
            self.register_parameter("post_mats_min", self.post_mats_min)

            self.mats1_max = nn.Parameter(torch.FloatTensor(offset_dim, expand_dim))
            nn.init.xavier_uniform_(self.mats1_max)
            self.register_parameter("mats1_max", self.mats1_max)

            self.post_mats_max = nn.Parameter(torch.FloatTensor(post_dim, offset_dim))
            nn.init.xavier_uniform_(self.post_mats_max)
            self.register_parameter("post_mats_max", self.post_mats_max)

        if pos_dim is not None:
            self.mats1_pos = nn.Parameter(torch.FloatTensor(pos_dim, pos_expand_dim))
            nn.init.xavier_uniform_(self.mats1_pos)
            self.register_parameter("mats1_pos", self.mats1_pos)

            self.post_mats_pos = nn.Parameter(torch.FloatTensor(post_dim, pos_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_pos)
            self.register_parameter("post_mats_pos", self.post_mats_pos)

        if rol_dim is not None:
            self.mats1_rol = nn.Parameter(torch.FloatTensor(rol_dim, rol_expand_dim))
            nn.init.xavier_uniform_(self.mats1_rol)
            self.register_parameter("mats1_rol", self.mats1_rol)

            self.post_mats_rol = nn.Parameter(torch.FloatTensor(post_dim, rol_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_rol)
            self.register_parameter("post_mats_rol", self.post_mats_rol)

        if strc_dim is not None :
            self.mats1_strc = nn.Parameter(torch.FloatTensor(strc_dim, strc_expand_dim))
            nn.init.xavier_uniform_(self.mats1_strc)
            self.register_parameter("mats1_strc", self.mats1_strc)

            self.post_mats_strc = nn.Parameter(torch.FloatTensor(post_dim, strc_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_strc)
            self.register_parameter("post_mats_strc", self.post_mats_strc)


    def forward(self, center_emb, min_emb, max_emb, pos_emb=None, rol_emb=None, strc_emb=None):
        # query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)

        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        bs_branch = center_emb.size(0)
        post_concat = []
        post_center = F.relu(torch.matmul(center_emb, self.mats1_center))
        post_concat.append(post_center)

        post_min = F.relu(torch.matmul(min_emb, self.mats1_min))
        post_concat.append(post_min)

        post_max = F.relu(torch.matmul(max_emb, self.mats1_max))
        post_concat.append(post_max)

        if pos_emb is not None:
            pos_emb = pos_emb.repeat(bs_branch, 1, 1)
            post_pos = F.relu(torch.matmul(pos_emb, self.mats1_pos))
            post_concat.append(post_pos)
        if rol_emb is not None :
            rol_emb = rol_emb.repeat(bs_branch, 1, 1)
            post_rol = F.relu(torch.matmul(rol_emb, self.mats1_rol))
            post_concat.append(post_rol)
        if strc_emb is not None :
            strc_emb = strc_emb.repeat(bs_branch, 1, 1)
            post_strc = F.relu(torch.matmul(strc_emb, self.mats1_strc))
            post_concat.append(post_strc)

        post_concat = torch.cat(post_concat, dim=2)

        center_emb = torch.matmul(post_concat, self.post_mats_center)
        min_emb = torch.matmul(post_concat, self.post_mats_min)
        max_emb = torch.matmul(post_concat, self.post_mats_max)

        if pos_emb is not None:
            pos_emb = torch.matmul(post_concat, self.post_mats_pos)
        if rol_emb is not None:
            rol_emb = torch.matmul(post_concat, self.post_mats_rol)
        if strc_emb is not None:
            strc_emb = torch.matmul(post_concat, self.post_mats_strc)

        return (center_emb, min_emb, max_emb, pos_emb, rol_emb, strc_emb)

class MultilayerNN_MinMax2(nn.Module): #projection 후에 cascading error 줄이기
    # torch min will just return a single element in the tensor
    # min, max도 결국 cent 처럼 좌표값이므로 같은 lin transformation을 사용해야하지 않을까?
    def __init__(self, coord_dim, pos_dim=None, rol_dim=None, strc_dim=None):
        super(MultilayerNN_MinMax2, self).__init__()
        self.coord_dim = coord_dim
        self.pos_dim = pos_dim
        self.rol_dim = rol_dim
        self.strc_dim = strc_dim

        coo_expand_dim = coord_dim * 2
        pos_expand_dim = pos_dim * 2 if pos_dim else 0
        rol_expand_dim = rol_dim * 2 if rol_dim else 0
        strc_expand_dim = strc_dim * 2 if strc_dim else 0
        post_dim = coo_expand_dim * 3 + pos_expand_dim + rol_expand_dim + strc_expand_dim
        #cent, min, max -> coo_expand_dim

        self.mats1_coord = nn.Parameter(torch.FloatTensor(coord_dim, coo_expand_dim))
        nn.init.xavier_uniform_(self.mats1_coord)
        self.register_parameter("mats1_coord", self.mats1_coord)

        self.post_mats_coord = nn.Parameter(torch.FloatTensor(post_dim, coord_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_coord)
        self.register_parameter("post_mats_coord", self.post_mats_coord)

        if pos_dim is not None:
            self.mats1_pos = nn.Parameter(torch.FloatTensor(pos_dim, pos_expand_dim))
            nn.init.xavier_uniform_(self.mats1_pos)
            self.register_parameter("mats1_pos", self.mats1_pos)

            self.post_mats_pos = nn.Parameter(torch.FloatTensor(post_dim, pos_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_pos)
            self.register_parameter("post_mats_pos", self.post_mats_pos)

        if rol_dim is not None:
            self.mats1_rol = nn.Parameter(torch.FloatTensor(rol_dim, rol_expand_dim))
            nn.init.xavier_uniform_(self.mats1_rol)
            self.register_parameter("mats1_rol", self.mats1_rol)

            self.post_mats_rol = nn.Parameter(torch.FloatTensor(post_dim, rol_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_rol)
            self.register_parameter("post_mats_rol", self.post_mats_rol)

        if strc_dim is not None :
            self.mats1_strc = nn.Parameter(torch.FloatTensor(strc_dim, strc_expand_dim))
            nn.init.xavier_uniform_(self.mats1_strc)
            self.register_parameter("mats1_strc", self.mats1_strc)

            self.post_mats_strc = nn.Parameter(torch.FloatTensor(post_dim, strc_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_strc)
            self.register_parameter("post_mats_strc", self.post_mats_strc)


    def forward(self, center_emb, min_emb, max_emb, pos_emb=None, rol_emb=None, strc_emb=None):
        # query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)

        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        bs_branch = center_emb.size(0)
        post_concat = []

        post_center = F.relu(torch.matmul(center_emb, self.mats1_coord))
        post_concat.append(post_center)

        post_min = F.relu(torch.matmul(min_emb, self.mats1_coord))
        post_concat.append(post_min)

        post_max = F.relu(torch.matmul(max_emb, self.mats1_coord))
        post_concat.append(post_max)

        if pos_emb is not None:
            pos_emb = pos_emb.repeat(bs_branch, 1, 1)
            post_pos = F.relu(torch.matmul(pos_emb, self.mats1_pos))
            post_concat.append(post_pos)
        if rol_emb is not None :
            rol_emb = rol_emb.repeat(bs_branch, 1, 1)
            post_rol = F.relu(torch.matmul(rol_emb, self.mats1_rol))
            post_concat.append(post_rol)
        if strc_emb is not None :
            strc_emb = strc_emb.repeat(bs_branch, 1, 1)
            post_strc = F.relu(torch.matmul(strc_emb, self.mats1_strc))
            post_concat.append(post_strc)

        post_concat = torch.cat(post_concat, dim=2)

        center_emb = torch.matmul(post_concat, self.post_mats_coord)
        min_emb = torch.matmul(post_concat, self.post_mats_coord)
        max_emb = torch.matmul(post_concat, self.post_mats_coord)

        if pos_emb is not None:
            pos_emb = torch.matmul(post_concat, self.post_mats_pos)
        if rol_emb is not None:
            rol_emb = torch.matmul(post_concat, self.post_mats_rol)
        if strc_emb is not None:
            strc_emb = torch.matmul(post_concat, self.post_mats_strc)

        return (center_emb, min_emb, max_emb, pos_emb, rol_emb, strc_emb)

class MultilayerNN_Offset(nn.Module): #projection 후에 cascading error 줄이기
    # torch min will just return a single element in the tensor
    def __init__(self, center_dim, offset_dim=None, pos_dim=None, rol_dim=None, strc_dim=None):
        super(MultilayerNN_Offset, self).__init__()
        self.center_dim = center_dim
        self.offset_dim = offset_dim
        self.pos_dim = pos_dim
        self.rol_dim = rol_dim
        self.strc_dim = strc_dim
        if self.offset_dim is not None :
            assert self.center_dim == self.offset_dim
        #
        expand_dim = center_dim * 2
        pos_expand_dim = pos_dim * 2 if pos_dim else 0
        rol_expand_dim = rol_dim * 2 if rol_dim else 0
        strc_expand_dim = strc_dim * 2 if strc_dim else 0
        post_dim = expand_dim * 2 + pos_expand_dim + rol_expand_dim + strc_expand_dim

        self.mats1_center = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform_(self.mats1_center)
        self.register_parameter("mats1_center", self.mats1_center)

        self.post_mats_center = nn.Parameter(torch.FloatTensor(post_dim, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_center)
        self.register_parameter("post_mats_center", self.post_mats_center)

        if offset_dim is not None :
            self.mats1_offset = nn.Parameter(torch.FloatTensor(offset_dim, expand_dim))
            nn.init.xavier_uniform_(self.mats1_offset)
            self.register_parameter("mats1_offset", self.mats1_offset)

            self.post_mats_offset = nn.Parameter(torch.FloatTensor(post_dim, offset_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_offset)
            self.register_parameter("post_mats_offset", self.post_mats_offset)

        if pos_dim is not None:
            self.mats1_pos = nn.Parameter(torch.FloatTensor(pos_dim, pos_expand_dim))
            nn.init.xavier_uniform_(self.mats1_pos)
            self.register_parameter("mats1_pos", self.mats1_pos)

            self.post_mats_pos = nn.Parameter(torch.FloatTensor(post_dim, pos_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_pos)
            self.register_parameter("post_mats_pos", self.post_mats_pos)

        if rol_dim is not None:
            self.mats1_rol = nn.Parameter(torch.FloatTensor(rol_dim, rol_expand_dim))
            nn.init.xavier_uniform_(self.mats1_rol)
            self.register_parameter("mats1_rol", self.mats1_rol)

            self.post_mats_rol = nn.Parameter(torch.FloatTensor(post_dim, rol_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_rol)
            self.register_parameter("post_mats_rol", self.post_mats_rol)

        if strc_dim is not None :
            self.mats1_strc = nn.Parameter(torch.FloatTensor(strc_dim, strc_expand_dim))
            nn.init.xavier_uniform_(self.mats1_strc)
            self.register_parameter("mats1_strc", self.mats1_strc)

            self.post_mats_strc = nn.Parameter(torch.FloatTensor(post_dim, strc_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_strc)
            self.register_parameter("post_mats_strc", self.post_mats_strc)


    def forward(self, center_emb, offset, pos_emb=None, rol_emb=None, strc_emb=None):
        # query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)

        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        post_concat = []
        post_center = F.relu(torch.matmul(center_emb, self.mats1_center))
        post_concat.append(post_center)

        if offset is not None :
            post_offset = F.relu(torch.matmul(offset, self.mats1_offset))
            post_concat.append(post_offset)
        if pos_emb is not None:
            post_pos = F.relu(torch.matmul(pos_emb, self.mats1_pos))
            post_concat.append(post_pos)
        if rol_emb is not None :
            post_rol = F.relu(torch.matmul(rol_emb, self.mats1_rol))
            post_concat.append(post_rol)
        if strc_emb is not None :
            post_strc = F.relu(torch.matmul(strc_emb, self.mats1_strc))
            post_concat.append(post_strc)

        post_concat = torch.cat(post_concat, dim=2)

        center_emb = torch.matmul(post_concat, self.post_mats_center)
        if offset is not None :
            offset = torch.matmul(post_concat, self.post_mats_offset)
        if pos_emb is not None:
            pos_emb = torch.matmul(post_concat, self.post_mats_pos)
        if rol_emb is not None:
            rol_emb = torch.matmul(post_concat, self.post_mats_rol)
        if strc_emb is not None:
            strc_emb = torch.matmul(post_concat, self.post_mats_strc)

        return (center_emb, offset, pos_emb, rol_emb, strc_emb)

class MultilayerNN_Center(nn.Module): #projection 후에 cascading error 줄이기
    # torch min will just return a single element in the tensor
    # min, max도 결국 cent 처럼 좌표값이므로 같은 lin transformation을 사용해야하지 않을까?
    def __init__(self, center_dim, pos_dim=None, rol_dim=None, strc_dim=None, hyper_dim=None):
        super(MultilayerNN_Center, self).__init__()
        self.center_dim = center_dim
        self.pos_dim = pos_dim
        self.rol_dim = rol_dim
        self.strc_dim = strc_dim
        self.hyper_dim = hyper_dim

        expand_dim = center_dim * 2
        pos_expand_dim = pos_dim * 2 if pos_dim else 0
        rol_expand_dim = rol_dim * 2 if rol_dim else 0
        strc_expand_dim = strc_dim * 2 if strc_dim else 0
        hyper_expand_dim = hyper_dim * 2 if hyper_dim else 0

        post_dim = expand_dim + pos_expand_dim + rol_expand_dim + strc_expand_dim + hyper_expand_dim
        #cent, min, max -> coo_expand_dim

        self.mats1_cent = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform_(self.mats1_cent)
        self.register_parameter("mats1_cent", self.mats1_cent)

        self.post_mats_cent = nn.Parameter(torch.FloatTensor(post_dim, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_cent)
        self.register_parameter("post_mats_cent", self.post_mats_cent)

        if pos_dim is not None:
            self.mats1_pos = nn.Parameter(torch.FloatTensor(pos_dim, pos_expand_dim))
            nn.init.xavier_uniform_(self.mats1_pos)
            self.register_parameter("mats1_pos", self.mats1_pos)

            self.post_mats_pos = nn.Parameter(torch.FloatTensor(post_dim, pos_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_pos)
            self.register_parameter("post_mats_pos", self.post_mats_pos)

        if rol_dim is not None:
            self.mats1_rol = nn.Parameter(torch.FloatTensor(rol_dim, rol_expand_dim))
            nn.init.xavier_uniform_(self.mats1_rol)
            self.register_parameter("mats1_rol", self.mats1_rol)

            self.post_mats_rol = nn.Parameter(torch.FloatTensor(post_dim, rol_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_rol)
            self.register_parameter("post_mats_rol", self.post_mats_rol)

        if strc_dim is not None :
            self.mats1_strc = nn.Parameter(torch.FloatTensor(strc_dim, strc_expand_dim))
            nn.init.xavier_uniform_(self.mats1_strc)
            self.register_parameter("mats1_strc", self.mats1_strc)

            self.post_mats_strc = nn.Parameter(torch.FloatTensor(post_dim, strc_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_strc)
            self.register_parameter("post_mats_strc", self.post_mats_strc)

        if hyper_dim is not None :
            self.mats1_hyper = nn.Parameter(torch.FloatTensor(hyper_dim, hyper_expand_dim))
            nn.init.xavier_uniform_(self.mats1_hyper)
            self.register_parameter("mats1_hyper", self.mats1_hyper)

            self.post_mats_hyper = nn.Parameter(torch.FloatTensor(post_dim, hyper_dim))
            # every time the initial is different
            nn.init.xavier_uniform_(self.post_mats_hyper)
            self.register_parameter("post_mats_hyper", self.post_mats_hyper)



    def forward(self, center_emb, pos_emb=None, rol_emb=None, strc_emb=None, hyper_emb=None):
        #center_emb, hyper_emb : (nbranch*bs, 1, d), pos,reo,strc : (1, 1, d)
        # query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)

        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        bs_branch = center_emb.size(0)
        post_concat = []

        post_center = F.relu(torch.matmul(center_emb, self.mats1_cent))
        post_concat.append(post_center)

        if pos_emb is not None:
            pos_emb = pos_emb.repeat(bs_branch, 1, 1)
            post_pos = F.relu(torch.matmul(pos_emb, self.mats1_pos))
            post_concat.append(post_pos)
        if rol_emb is not None :
            rol_emb = rol_emb.repeat(bs_branch, 1, 1)
            post_rol = F.relu(torch.matmul(rol_emb, self.mats1_rol))
            post_concat.append(post_rol)
        if strc_emb is not None :
            strc_emb = strc_emb.repeat(bs_branch, 1, 1)
            post_strc = F.relu(torch.matmul(strc_emb, self.mats1_strc))
            post_concat.append(post_strc)
        if hyper_emb is not None :
            post_hyper = F.relu(torch.matmul(hyper_emb, self.mats1_hyper))
            post_concat.append(post_hyper)

        post_concat = torch.cat(post_concat, dim=2) #(bs, 1, 2*(5d))
        #####################################################################################
        center_emb = torch.matmul(post_concat, self.post_mats_cent)

        if pos_emb is not None:
            pos_emb = torch.matmul(post_concat, self.post_mats_pos)
        if rol_emb is not None:
            rol_emb = torch.matmul(post_concat, self.post_mats_rol)
        if strc_emb is not None:
            strc_emb = torch.matmul(post_concat, self.post_mats_strc)
        if hyper_emb is not None :
            hyper_emb = torch.matmul(post_concat, self.post_mats_hyper)

        return (center_emb, pos_emb, rol_emb, strc_emb, hyper_emb)





class Query2box(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 writer=None, geo=None,
                 anchor_not_center_trans=False,
                 rel_hmap=None, rel_tmap=None,
                 rel_h_degree=None, rel_t_degree=None,
                 max_qry_len=4, num_role=3, structure_dim=108,
                 PE=False, RE=False, SE=False, table_normalization=False,
                 cen=None, offset_deepsets=None,
                 center_deepsets=None, offset_use_center=None, center_use_offset=None,
                 att_reg=0., off_reg=0., att_tem=1., euo=False,
                 gamma2=0, bn='no', nat=1, activation='relu'):
        super(Query2box, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.writer = writer
        self.geo = geo
        self.cen = cen
        self.offset_deepsets = offset_deepsets
        self.center_deepsets = center_deepsets
        self.offset_use_center = offset_use_center
        self.center_use_offset = center_use_offset
        ############################################################
        self.max_qry_len = max_qry_len
        self.num_role = num_role
        self.structure_dim = structure_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.table_normalization = table_normalization
        ############################################################
        self.att_reg = att_reg
        self.off_reg = off_reg
        self.att_tem = att_tem
        ###############################################################################################################
        self.anchor_not_center_trans=anchor_not_center_trans
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

        """
        rel_hmap : kg_triple에서 r의 head가 되는 애들
        rel_tmap : kg_triple에서 r의 tail이 되는 애들
        """
        ###############################################################################################################
        self.euo = euo
        self.his_step = 0
        self.bn = bn
        self.nat = nat
        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        if gamma2 == 0:
            gamma2 = gamma

        self.gamma2 = nn.Parameter(
            torch.Tensor([gamma2]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        ###############################################################################################################
        """
        entity embedding은 padding 위해서 nentity+1 만큼
        initialization은 [:-1] 까지
        """
        self.entity_embedding = nn.Parameter(torch.zeros(nentity+1, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding[:-1],
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        ###############################################################################################################
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        ###############################################################################################################
        self.PE = PE
        self.RE = RE
        self.SE = SE

        if not (self.PE and self.RE) :
            assert not self.SE

        if self.table_normalization :
            assert self.SE
        self.pos_dim = self.structure_dim if self.PE else None
        self.rol_dim = self.structure_dim if self.RE else None
        self.strc_dim = self.structure_dim if self.SE else None

        self.init_role_embedding()
        self.init_position_embedding()
        self.init_structure_embedding()

        self.center_trans = MultilayerNN_Center(center_dim=self.hidden_dim, pos_dim=self.pos_dim, rol_dim=self.rol_dim,
                                                 strc_dim=self.strc_dim, hyper_dim=self.hyper_dim)

        logging.info('Use position embedding : {}'.format(True if self.pos0_embedding is not None else False))
        logging.info('Use role embedding : {}'.format(True if self.variable_role_embedding is not None else False))
        logging.info('Use structure embedding : {}'.format(self.SE))

        ###############################################################################################################


        if self.geo == 'vec':
            if self.center_deepsets == 'vanilla':
                self.deepsets = CenterSet(self.relation_dim, self.relation_dim, False, agg_func=torch.mean, bn=bn,
                                          nat=nat)
            elif self.center_deepsets == 'attention':
                self.deepsets = AttentionSet(self.relation_dim, self.relation_dim, False,
                                             att_reg=self.att_reg, att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'eleattention':
                self.deepsets = AttentionSet(self.relation_dim, self.relation_dim, False,
                                             att_reg=self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'mean':
                self.deepsets = MeanSet()
            else:
                assert False

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            if self.euo:
                self.entity_offset_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
                nn.init.uniform_(
                    tensor=self.entity_offset_embedding,
                    a=0.,
                    b=self.embedding_range.item()
                )

            if self.center_deepsets == 'vanilla':
                self.center_sets = CenterSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                             agg_func=torch.mean, bn=bn, nat=nat)
            elif self.center_deepsets == 'attention':
                self.center_sets = AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                                att_reg=self.att_reg, att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'eleattention':
                self.center_sets = AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                                att_reg=self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn,
                                                nat=nat)
            elif self.center_deepsets == 'mean':
                self.center_sets = MeanSet()
            else:
                assert False

            if self.offset_deepsets == 'vanilla':
                self.offset_sets = OffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center,
                                             agg_func=torch.mean)
            elif self.offset_deepsets == 'inductive':
                self.offset_sets = InductiveOffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center,
                                                      self.off_reg, agg_func=torch.mean)
            elif self.offset_deepsets == 'min':
                self.offset_sets = MinSet()
            else:
                assert False

        if model_name not in ['TransE', 'BoxTransE']:
            raise ValueError('model %s not supported' % model_name)

    def init_position_embedding(self):
        if self.PE :
            self.pos0_embedding = nn.Parameter(torch.zeros(1, 1, self.pos_dim))
            self.pos1_embedding = nn.Parameter(torch.zeros(1, 1, self.pos_dim))
            self.pos2_embedding = nn.Parameter(torch.zeros(1, 1, self.pos_dim))
            self.pos3_embedding = nn.Parameter(torch.zeros(1, 1, self.pos_dim))
            nn.init.uniform_(tensor=self.pos0_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.pos1_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.pos2_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.pos3_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            self.rel2posemb = {0: self.pos0_embedding, 1: self.pos1_embedding, 2: self.pos2_embedding,
                               3: self.pos3_embedding}
        else :
            self.pos0_embedding = None
            self.pos1_embedding = None
            self.pos2_embedding = None
            self.pos3_embedding = None

    def init_role_embedding(self):
        if self.RE :
            self.anchor_role_embedding = nn.Parameter(torch.zeros(1, 1, self.rol_dim))
            self.variable_role_embedding = nn.Parameter(torch.zeros(1, 1, self.rol_dim))
            self.answer_role_embedding = nn.Parameter(torch.zeros(1, 1, self.rol_dim))
            nn.init.uniform_(tensor=self.anchor_role_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.variable_role_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.answer_role_embedding, a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
        else :
            self.anchor_role_embedding = None
            self.variable_role_embedding = None
            self.answer_role_embedding = None

    def init_structure_embedding(self):
        if self.SE:
            self.query2onehot = get_query2onehot(self.device, self.table_normalization)
            self.structural_embedding = nn.Linear(self.max_qry_len * self.num_role, self.structure_dim)

    ################################################################################################

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
        if qtype == 'chain-inter' :
            #var_neigh_rel_inds : (bs, 2), ans_neigh_rel_inds : (bs, 2)

            var_hyper_emb = self.in1out1_hyper_agg(var_neigh_rel_inds)
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds)

        elif qtype == 'inter-chain' or qtype == 'union-chain' :
            #var_neigh_rel_inds : (bs, 3), ans_neigh_rel_inds : (bs, 1)

            var_hyper_emb = self.in2out1_hyper_agg(var_neigh_rel_inds)
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds)

        elif 'inter' in qtype or 'union' in qtype :
            #var_neigh_rel_inds : None, ans_neigh_rel_inds : (bs, N)

            var_hyper_emb = None
            ans_hyper_emb = self.inNout0_hyper_agg(ans_neigh_rel_inds)

        else :
            # var_neigh_rel_inds := 1c : (None, None), 2c : ((bs, 2 ), None), 3c : ((bs, 2), (bs, 2))
            # ans_neigh_rel_inds : (bs, 1)

            rel_len = int(qtype.split('-')[0])
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

        return node_hyper_tail_emb

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


    def forward(self, sample, rel_len, qtype, mode='single'):
        if self.SE :
            structural_embedding = self.structural_embedding(self.query2onehot[qtype]).unsqueeze(0).unsqueeze(0)
        else :
            structural_embedding = None

        if qtype == 'chain-inter':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(
                1)  # (bs, 1, ent_dim)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                1)  # (bs, 1, ent_dim)

            ########################################################################################################
            var_neigh_rel_inds = head_part[:, [1, 2]] #(bs, 2)
            ans_neigh_rel_inds = head_part[:, [2, 4]] #(bs, 2)
            ########################################################################################################

            if self.euo and self.geo == 'box':
                head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 3]).unsqueeze(1)
            else :
                head_offset_1 = torch.zeros_like(head_1)
                head_offset_2 = torch.zeros_like(head_2)

            # head_min_1 = head_1 - 0.5 * self.func(head_offset_1)
            # head_max_1 = head_1 + 0.5 * self.func(head_offset_1)
            #
            # head_min_2 = head_1 - 0.5 * self.func(head_offset_2)
            # head_max_2 = head_1 + 0.5 * self.func(head_offset_2)
            if (self.PE or self.RE or self.SE) and not self.anchor_not_center_trans :

                head_1, _, _, _, _ = self.center_trans(head_1, self.pos0_embedding,
                                                       self.anchor_role_embedding,
                                                       structural_embedding)


                head_2, _, _, _, _ = self.center_trans(head_2, self.pos1_embedding,
                                                       self.anchor_role_embedding,
                                                       structural_embedding)

            head = torch.cat([head_1, head_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)

            relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == 'inter-chain' or qtype == 'union-chain':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)

            # head = torch.cat([head_1, head_2], dim=0)
            if self.euo and self.geo == 'box':
                head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 2]).unsqueeze(1)
            else :
                head_offset_1 = torch.zeros_like(head_1)
                head_offset_2 = torch.zeros_like(head_2)

            # head_min_1 = head_1 - 0.5 * self.func(head_offset_1)
            # head_max_1 = head_1 + 0.5 * self.func(head_offset_1)
            #
            # head_min_2 = head_1 - 0.5 * self.func(head_offset_2)
            # head_max_2 = head_1 + 0.5 * self.func(head_offset_2)

            ########################################################################################################
            var_neigh_rel_inds = head_part[:, [1, 3, 4]] #(bs, 3)
            ans_neigh_rel_inds = head_part[:, [4]] #(bs, 1)
            ########################################################################################################
            if (self.PE or self.RE or self.SE) and not self.anchor_not_center_trans:
                head_1, _, _, _, _ = self.center_trans(head_1, self.pos0_embedding,
                                                       self.anchor_role_embedding,
                                                       structural_embedding)

                head_2, _, _, _, _ = self.center_trans(head_2, self.pos0_embedding,
                                                       self.anchor_role_embedding,
                                                       structural_embedding)


            head = torch.cat([head_1, head_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)

            relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

                if self.euo and self.geo == 'box':
                    head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                       index=sample[:, 0]).unsqueeze(1)
                    head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                       index=sample[:, 2]).unsqueeze(1)

                else:
                    head_offset_1 = torch.zeros_like(head_1)
                    head_offset_2 = torch.zeros_like(head_2)

                # head_min_1 = head_1 - 0.5 * self.func(head_offset_1)
                # head_max_1 = head_1 + 0.5 * self.func(head_offset_2)
                #
                # head_min_2 = head_2 - 0.5 * self.func(head_offset_2)
                # head_max_2 = head_2 + 0.5 * self.func(head_offset_2)
                ################################################################################################
                var_neigh_rel_inds = None
                ans_neigh_rel_inds = sample[:, [1, 3]]
                ################################################################################################
                if (self.PE or self.RE or self.SE) and not self.anchor_not_center_trans:
                    head_1, _, _, _, _ = self.center_trans(head_1, self.pos0_embedding,
                                                           self.anchor_role_embedding,
                                                           structural_embedding)

                    head_2, _, _, _, _ = self.center_trans(head_2, self.pos0_embedding,
                                                        self.anchor_role_embedding,
                                                        structural_embedding)


                head = torch.cat([head_1, head_2], dim=0)

                if rel_len == 3:
                    ################################################################################################
                    ans_neigh_rel_inds = sample[:, [1, 3, 5]]
                    ################################################################################################
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)

                    if self.euo and self.geo == 'box':
                        head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                           index=sample[:, 4]).unsqueeze(1)
                    else :
                        head_offset_3 = torch.zeros_like(head_3)

                    # head_min_3 = head_3 - 0.5 * self.func(head_offset_3)
                    # head_max_3 = head_3 + 0.5 * self.func(head_offset_3)
                    if (self.PE or self.RE or self.SE) and not self.anchor_not_center_trans:
                        head_3, _, _, _, _ = self.center_trans(head_3, self.pos0_embedding,
                                                               self.anchor_role_embedding,
                                                               structural_embedding)


                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    offset_2 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset_1, offset_2], dim=0)
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset_3], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                ################################################################################################
                var_neigh_rel_inds = None
                ans_neigh_rel_inds = head_part[:, [1, 3]]
                ################################################################################################
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)

                if self.euo and self.geo == 'box':
                    head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                       index=head_part[:, 0]).unsqueeze(1)


                    head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                       index=head_part[:, 2]).unsqueeze(1)
                else:
                    head_offset_1 = torch.zeros_like(head_1)
                    head_offset_2 = torch.zeros_like(head_2)

                # head_min_1 = head_1 - 0.5 * self.func(head_offset_1)
                # head_max_1 = head_1 + 0.5 * self.func(head_offset_1)
                #
                # head_min_2 = head_2 - 0.5 * self.func(head_offset_2)
                # head_max_2 = head_2 + 0.5 * self.func(head_offset_2)

                if (self.SE or self.PE or self.RE) and not self.anchor_not_center_trans:
                    head_1, _, _, _, _ = self.center_trans(head_1, self.pos0_embedding,
                                                           self.anchor_role_embedding,
                                                           structural_embedding)


                    head_2, _, _, _, _ = self.center_trans(head_2, self.pos0_embedding,
                                                        self.anchor_role_embedding,
                                                        structural_embedding)


                head = torch.cat([head_1, head_2], dim=0)

                if rel_len == 3:
                    ################################################################################################
                    ans_neigh_rel_inds = head_part[:, [1, 3, 5]]
                    ################################################################################################
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)

                    if self.euo and self.geo == 'box':
                        head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                           index=head_part[:, 4]).unsqueeze(1)
                    else :
                        head_offset_3 = torch.zeros_like(head_3)

                    # head_min_3 = head_3 - 0.5 * self.func(head_offset_3)
                    # head_max_3 = head_3 + 0.5 * self.func(head_offset_3)
                    if (self.PE or self.RE or self.SE) and not self.anchor_not_center_trans:
                        head_3, _, _, _, _ = self.center_trans(head_3, self.pos0_embedding,
                                                               self.anchor_role_embedding,
                                                               structural_embedding)


                    head = torch.cat([head, head_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset_1, offset_2], dim=0)
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset_3], dim=0)

        elif qtype == '1-chain' or qtype == '2-chain' or qtype == '3-chain':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                ################################################################################################
                var1_neigh_rel_inds = None
                var2_neigh_rel_inds = None
                ans_neigh_rel_inds = sample[:, [1]] #(bs, 1)
                ################################################################################################

                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

                relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                if self.geo == 'box':
                    offset = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0,
                                                         index=sample[:, 0]).unsqueeze(1)

                    else:
                        head_offset = torch.zeros_like(head)

                # head_min = head - 0.5 * self.func(head_offset)
                # head_max = head + 0.5 * self.func(head_offset)

                if (self.SE or self.PE or self.RE) and not self.anchor_not_center_trans:
                    head, _, _, _, _ = self.center_trans(head, self.pos0_embedding,
                                                         self.anchor_role_embedding,
                                                         structural_embedding)


                if rel_len == 2 or rel_len == 3:
                    ################################################################################################
                    var1_neigh_rel_inds = sample[:, [1, 2]]
                    ans_neigh_rel_inds = sample[:, [2]]
                    ################################################################################################

                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    if self.geo == 'box':
                        offset2 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    ################################################################################################
                    var2_neigh_rel_inds = sample[:, [2, 3]]
                    ans_neigh_rel_inds = sample[:, [3]]
                    ################################################################################################
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)
                    if self.geo == 'box':
                        offset3 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset3], 1)


                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)


            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                ################################################################################################
                var1_neigh_rel_inds = None
                var2_neigh_rel_inds = None
                ans_neigh_rel_inds = head_part[:, [1]]
                ################################################################################################
                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)

                relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                if self.geo == 'box':
                    offset = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0,
                                                         index=head_part[:, 0]).unsqueeze(1)
                    else :
                        head_offset = torch.zeros_like(head)

                # head_min = head - 0.5 * self.func(head_offset)
                # head_max = head + 0.5 * self.func(head_offset)
                if (self.PE or self.RE or self.SE) and not self.anchor_not_center_trans:
                    head, _, _, _, _ = self.center_trans(head, self.pos0_embedding,
                                                         self.anchor_role_embedding,
                                                         structural_embedding)


                if rel_len == 2 or rel_len == 3:
                    ################################################################################################
                    var1_neigh_rel_inds = head_part[:, [1, 2]]
                    ans_neigh_rel_inds = head_part[:, [2]]
                    ################################################################################################
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    if self.geo == 'box':
                        offset2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    ################################################################################################
                    var2_neigh_rel_inds = head_part[:, [2, 3]]
                    ans_neigh_rel_inds = head_part[:, [3]]
                    ################################################################################################
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)
                    if self.geo == 'box':
                        offset3 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset3], 1)

                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
            var_neigh_rel_inds = (var1_neigh_rel_inds, var2_neigh_rel_inds)
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'BoxTransE': self.BoxTransE,
            'TransE': self.TransE,
        }
        if self.geo == 'vec':
            offset = None
            head_offset = None
        if self.geo == 'box':
            if not self.euo:
                head_offset = None

        if self.model_name in model_func:
            if qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union':
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail,
                                                                                               var_neigh_rel_inds,
                                                                                               ans_neigh_rel_inds,
                                                                                               mode, offset,
                                                                                               head_offset, 1,
                                                                                               qtype)
            else:
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail,
                                                                                               var_neigh_rel_inds,
                                                                                               ans_neigh_rel_inds,
                                                                                               mode, offset,
                                                                                               head_offset, rel_len,
                                                                                               qtype)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score, score_cen, offset_norm, score_cen_plus, None, None

    def BoxTransE(self, head, relation, tail, var_neigh_rel_inds, ans_neigh_rel_inds,
                  mode, offset, head_offset, rel_len, qtype):
        # projection : head : (bs, 1, dim), intersection : (nbranch*bs, 1, dim)
        if self.SE :
            structural_embedding = self.structural_embedding(self.query2onehot[qtype]).unsqueeze(0).unsqueeze(0)
            # (1,1,108)
        else :
            structural_embedding = None

        if qtype == 'chain-inter':
            ##############################################################################
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds, ans_neigh_rel_inds, qtype)
            #(bs, d), (bs, d)
            var_hyper_emb = var_hyper_emb.unsqueeze(1)
            ans_hyper_emb = ans_hyper_emb.unsqueeze(1)
            ##############################################################################
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)

            heads = torch.chunk(head, 2, dim=0) #((bs, 1, d), (bs, 1, d))
            bs = heads[0].size(0)

            query_center_11 = heads[0] + relations[0][:, 0, :, :] #(bs, 1, d)
            assert (var_hyper_emb.size() == heads[0].size() == ans_hyper_emb.size() == heads[0].size())

            variable_11, _, _, _, _ = self.center_trans(query_center_11, self.pos1_embedding,
                                                        self.variable_role_embedding,
                                                        structural_embedding,
                                                        var_hyper_emb)

            query_center_12 = variable_11 + relations[1][:, 0, :, :]

            variable_12, _, _, _, _ = self.center_trans(query_center_12, self.pos2_embedding,
                                                        self.answer_role_embedding,
                                                        structural_embedding,
                                                        ans_hyper_emb)

            query_center_1 = variable_12

            query_center_2 = heads[1] + relations[2][:, 0, :, :]

            variable_2,  _, _, _, _ = self.center_trans(query_center_2, self.pos2_embedding,
                                                        self.answer_role_embedding,
                                                        structural_embedding,
                                                        ans_hyper_emb)

            query_center_2 = variable_2

            if self.euo:
                query_min_1 = query_center_1 - 0.5 * self.func(head_offsets[0]) - 0.5 * self.func(
                    offsets[0][:, 0, :, :]) - 0.5 * self.func(offsets[1][:, 0, :, :])
                query_min_2 = query_center_2 - 0.5 * self.func(head_offsets[1]) - 0.5 * self.func(
                    offsets[2][:, 0, :, :])
                query_max_1 = query_center_1 + 0.5 * self.func(head_offsets[0]) + 0.5 * self.func(
                    offsets[0][:, 0, :, :]) + 0.5 * self.func(offsets[1][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(head_offsets[1]) + 0.5 * self.func(
                    offsets[2][:, 0, :, :])
            else:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:, 0, :, :]) - 0.5 * self.func(
                    offsets[1][:, 0, :, :])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[2][:, 0, :, :])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:, 0, :, :]) + 0.5 * self.func(
                    offsets[1][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[2][:, 0, :, :])

            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)

            offset_1 = (query_max_1 - query_min_1).squeeze(1)
            offset_2 = (query_max_2 - query_min_2).squeeze(1)

            new_query_center = self.center_sets(query_center_1, offset_1, query_center_2, offset_2)
            new_offset = self.offset_sets(query_center_1, offset_1, query_center_2, offset_2)
            new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
            new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center.unsqueeze(1) - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1)

        elif qtype == 'inter-chain':
            ##############################################################################
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds, ans_neigh_rel_inds, qtype)
            # (bs, d), (bs, d)
            var_hyper_emb = var_hyper_emb.unsqueeze(1) #(bs, 1, d)
            ans_hyper_emb = ans_hyper_emb.unsqueeze(1) #(bs, 1, d)
            ##############################################################################
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)

            heads = torch.chunk(head, 2, dim=0) #((bs, 1, d), (bs, 1, d))

            assert var_hyper_emb.size() == ans_hyper_emb.size() == heads[0].size()

            bs = heads[0].size(0)

            query_center_1 = heads[0] + relations[0][:, 0, :, :]
            assert var_hyper_emb.size() == heads[0].size()
            query_center_1, _, _, _, _ = self.center_trans(query_center_1, self.pos1_embedding,
                                                            self.variable_role_embedding,
                                                            structural_embedding,
                                                            var_hyper_emb)

            query_center_2 = head[1] + relations[1][:, 0, :, :]

            query_center_2, _, _, _, _ = self.center_trans(query_center_2, self.pos1_embedding,
                                                        self.variable_role_embedding,
                                                        structural_embedding,
                                                        var_hyper_emb)

            if self.euo:
                query_min_1 = query_center_1 - 0.5 * self.func(head_offsets[0]) - 0.5 * self.func(
                    offsets[0][:, 0, :, :])
                query_min_2 = query_center_2 - 0.5 * self.func(head_offsets[1]) - 0.5 * self.func(
                    offsets[1][:, 0, :, :])
                query_max_1 = query_center_1 + 0.5 * self.func(head_offsets[0]) + 0.5 * self.func(
                    offsets[0][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(head_offsets[1]) + 0.5 * self.func(
                    offsets[1][:, 0, :, :])
            else:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:, 0, :, :])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[1][:, 0, :, :])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[1][:, 0, :, :])
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            offset_1 = (query_max_1 - query_min_1).squeeze(1)
            offset_2 = (query_max_2 - query_min_2).squeeze(1)

            conj_query_center = self.center_sets(query_center_1, offset_1, query_center_2, offset_2).unsqueeze(1)
            new_offset = self.offset_sets(query_center_1, offset_1, query_center_2, offset_2).unsqueeze(1)

            query_center_3 = conj_query_center + relations[2][:, 0, :, :]
            new_query_center, _, _, _, _ = self.center_trans(query_center_3, self.pos2_embedding,
                                                             self.answer_role_embedding,
                                                             structural_embedding,
                                                             ans_hyper_emb)
            ############################################################################################################
            ############################################################################################################
            """
            intersection 후에 projection 수행할 떄 두 가지 option 있음
            option1 일 경우 center_set에 의해 변형된 conj_query_center에 관하여 box_min, box_max가 결정됨
            option2 일 경우 conj_query_center와 상관없이 이전에 구한 box_min, box_max에 의해서 새로운 min, max가 결정됨.
            """
            #original
            new_query_min = new_query_center - 0.5 * self.func(new_offset) - 0.5 * self.func(offsets[2][:, 0, :, :])
            new_query_max = new_query_center + 0.5 * self.func(new_offset) + 0.5 * self.func(offsets[2][:, 0, :, :])
            # option 1
            # query_min_3 = query_center_3 - conj_query_offset
            # query_max_3 = query_center_3 + conj_query_offset

            #option 2
            # query_min_3 = query_min_2 + relations[2][:, 0, :, :] - conj_query_offset
            # query_max_3 = query_max_2 + relations[2][:, 0, :, :] + conj_query_offset
            ############################################################################################################
            ############################################################################################################



            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center

        elif qtype == 'union-chain':
            ##############################################################################
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds, ans_neigh_rel_inds, qtype)
            # (bs, d), (bs, d)
            var_hyper_emb = var_hyper_emb.unsqueeze(1) #(bs, 1, d)
            ans_hyper_emb = ans_hyper_emb.unsqueeze(1) #(bs, 1, d)
            ##############################################################################
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)

            heads = torch.chunk(head, 2, dim=0)
            assert var_hyper_emb.size() == ans_hyper_emb.size() == heads[0].size()
            bs = heads[0].size(0)

            query_center_11 = heads[0] + relations[0][:, 0, :, :]

            query_center_11, _, _, _, _ = self.center_trans(query_center_11, self.pos1_embedding,
                                                            self.variable_role_embedding,
                                                            structural_embedding,
                                                            var_hyper_emb)

            query_center_12 = query_center_11 + relations[2][:, 0, :, :]

            query_center_12, _, _, _, _ = self.center_trans(query_center_12, self.pos2_embedding,
                                                         self.answer_role_embedding,
                                                         structural_embedding,
                                                         ans_hyper_emb)

            query_center_21 = heads[1] + relations[1][:, 0, :, :]

            query_center_21, _, _, _, _ = self.center_trans(query_center_21, self.pos1_embedding,
                                                            self.variable_role_embedding,
                                                            structural_embedding,
                                                            var_hyper_emb)

            query_center_22 = query_center_21 + relations[2][:, 0, :, :]

            query_center_22, _, _, _, _ = self.center_trans(query_center_22, self.pos2_embedding,
                                                            self.answer_role_embedding,
                                                            structural_embedding,
                                                            ans_hyper_emb)

            query_center_1 = query_center_12
            query_center_2 = query_center_22

            if self.euo:
                query_min_1 = query_center_1 - 0.5 * self.func(head_offsets[0]) - 0.5 * self.func(
                    offsets[0][:, 0, :, :]) - 0.5 * self.func(offsets[2][:, 0, :, :])
                query_min_2 = query_center_2 - 0.5 * self.func(head_offsets[1]) - 0.5 * self.func(
                    offsets[1][:, 0, :, :]) - 0.5 * self.func(offsets[2][:, 0, :, :])
                query_max_1 = query_center_1 + 0.5 * self.func(head_offsets[0]) + 0.5 * self.func(
                    offsets[0][:, 0, :, :]) + 0.5 * self.func(offsets[2][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(head_offsets[1]) + 0.5 * self.func(
                    offsets[1][:, 0, :, :]) + 0.5 * self.func(offsets[2][:, 0, :, :])
            else:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:, 0, :, :]) - 0.5 * self.func(
                    offsets[2][:, 0, :, :])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[1][:, 0, :, :]) - 0.5 * self.func(
                    offsets[2][:, 0, :, :])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:, 0, :, :]) + 0.5 * self.func(
                    offsets[2][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[1][:, 0, :, :]) + 0.5 * self.func(
                    offsets[2][:, 0, :, :])

            new_query_min = torch.stack([query_min_1, query_min_2], dim=0)
            new_query_max = torch.stack([query_max_1, query_max_2], dim=0)
            new_query_center = torch.stack([query_center_1, query_center_2], dim=0)
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center

        else:
            """
            head
            projection : (bs, 1, dim), intersection : (nbranch*bs, 1, dim),

            relation
            projection : (bs, nrelation, 1, dim), intersection : (nbranch*bs, 1, 1, dim)
            """
            #########################################################################################################
            var_hyper_emb, ans_hyper_emb = self.node_hyper_embedding(var_neigh_rel_inds, ans_neigh_rel_inds, qtype)
            """
            1c : (None, None), (bs, d)
            2c : ((bs, d), None), (bs, d)
            3c : ((bs, d), (bs, d)), (bs, d)
            2i : None, (bs, d)
            3i : None, (bs, d)
            """
            ans_hyper_emb = ans_hyper_emb.unsqueeze(1) #(bs, 1, d)

            ###########################################################################################################
            query_center = head #(N*bs, 1, d)
            #relation : (N*bs, 1, 1, d)
            bs_branch = head.size(0)
            for rel in range(rel_len):  # 1p -> 0, 2p -> 0,1, 3p -> 0,1,2   2i,3i -> 0
                query_center = query_center + relation[:, rel, :, :]
                if rel == rel_len - 1:  # 마지막 -> answer_role_embedding
                    if self.RE:
                        role_embedding = self.answer_role_embedding
                    else :
                        role_embedding = None

                    if self.Hyper:
                        hyper_embedding = ans_hyper_emb
                    else :
                        hyper_embedding = None
                else:
                    if self.RE :
                        role_embedding = self.variable_role_embedding
                    else :
                        role_embedding = None

                    if self.Hyper :
                        hyper_embedding = var_hyper_emb[rel].unsqueeze(1)
                    else :
                        hyper_embedding = None

                if self.PE :
                    positional_embedding = self.rel2posemb[rel + 1]
                else :
                    positional_embedding = None

                query_center, _, _, _, _ = self.center_trans(query_center, positional_embedding,
                                                             role_embedding, structural_embedding,
                                                             hyper_embedding.repeat(bs_branch//hyper_embedding.size(0),1,1))

            if self.euo:
                query_min = query_center - 0.5 * self.func(head_offset)
                query_max = query_center + 0.5 * self.func(head_offset)
            else:
                query_min = query_center
                query_max = query_center

            for rel in range(0, rel_len):
                query_min = query_min - 0.5 * self.func(offset[:, rel, :, :])
                query_max = query_max + 0.5 * self.func(offset[:, rel, :, :])

            if 'inter' not in qtype and 'union' not in qtype:
                score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
                score_center = query_center - tail
                score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - query_center
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                queries_min = torch.chunk(query_min, rel_len, dim=0)
                queries_max = torch.chunk(query_max, rel_len, dim=0)
                queries_center = torch.chunk(query_center, rel_len, dim=0)

                tails = torch.chunk(tail, rel_len, dim=0)
                offsets = query_max - query_min
                offsets = torch.chunk(offsets, rel_len, dim=0)
                if 'inter' in qtype:
                    if rel_len == 2:
                        new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                            queries_center[1].squeeze(1), offsets[1].squeeze(1))
                        new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                      queries_center[1].squeeze(1), offsets[1].squeeze(1))

                    elif rel_len == 3:
                        new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                            queries_center[1].squeeze(1), offsets[1].squeeze(1),
                                                            queries_center[2].squeeze(1), offsets[2].squeeze(1))
                        new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                      queries_center[1].squeeze(1), offsets[1].squeeze(1),
                                                      queries_center[2].squeeze(1), offsets[2].squeeze(1))
                    new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
                    new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)
                    score_offset = F.relu(new_query_min - tails[0]) + F.relu(tails[0] - new_query_max)
                    score_center = new_query_center.unsqueeze(1) - tails[0]
                    score_center_plus = torch.min(new_query_max,
                                                  torch.max(new_query_min, tails[0])) - new_query_center.unsqueeze(1)
                elif 'union' in qtype:
                    new_query_min = torch.stack(queries_min, dim=0)
                    new_query_max = torch.stack(queries_max, dim=0)
                    new_query_center = torch.stack(queries_center, dim=0)
                    score_offset = F.relu(new_query_min - tails[0]) + F.relu(tails[0] - new_query_max)
                    score_center = new_query_center - tails[0]
                    score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tails[0])) - new_query_center
                else:
                    assert False, 'qtype not exists: %s' % qtype

        score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)
        score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)
        score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(
            score_center_plus, p=1, dim=-1)
        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
            score_center = torch.max(score_center, dim=0)[0]
            score_center_plus = torch.max(score_center_plus, dim=0)[0]

        return score, score_center, torch.mean(torch.norm(offset, p=2, dim=2).squeeze(1)), score_center_plus, None

    def TransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):

        if qtype == 'chain-inter':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:, 0, :, :] + relations[1][:, 0, :, :]).squeeze(1)
            score_2 = (heads[1] + relations[2][:, 0, :, :]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score - tail
        elif qtype == 'inter-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:, 0, :, :]).squeeze(1)
            score_2 = (heads[1] + relations[1][:, 0, :, :]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score + relations[2][:, 0, :, :] - tail
        elif qtype == 'union-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = heads[0] + relations[0][:, 0, :, :] + relations[2][:, 0, :, :]
            score_2 = heads[1] + relations[1][:, 0, :, :] + relations[2][:, 0, :, :]
            conj_score = torch.stack([score_1, score_2], dim=0)
            score = conj_score - tail
        else:
            score = head
            for rel in range(rel_len):
                score = score + relation[:, rel, :, :]

            if 'inter' not in qtype and 'union' not in qtype:
                score = score - tail
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                score = score.squeeze(1)
                scores = torch.chunk(score, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                if 'inter' in qtype:
                    if rel_len == 2:
                        conj_score = self.deepsets(scores[0], None, scores[1], None)
                    elif rel_len == 3:
                        conj_score = self.deepsets(scores[0], None, scores[1], None, scores[2], None)
                    conj_score = conj_score.unsqueeze(1)
                    score = conj_score - tails[0]
                elif 'union' in qtype:
                    conj_score = torch.stack(scores, dim=0)
                    score = conj_score - tails[0]
                else:
                    assert False, 'qtype not exist: %s' % qtype

        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
        if qtype == '2-union':
            score = score.unsqueeze(0)
        return score, None, None, 0., []

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        rel_len = int(train_iterator.qtype.split('-')[0])
        qtype = train_iterator.qtype
        negative_score, negative_score_cen, negative_offset, negative_score_cen_plus, _, _ = model(
            (positive_sample, negative_sample), rel_len, qtype, mode=mode)

        if model.geo == 'box':
            negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score, positive_score_cen, positive_offset, positive_score_cen_plus, _, _ = model(positive_sample,
                                                                                                   rel_len, qtype)
        if model.geo == 'box':
            positive_score = F.logsigmoid(positive_score_cen_plus).squeeze(dim=1)
        else:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, test_ans, test_ans_hard, args):
        qtype = test_triples[0][-1]
        if qtype == 'chain-inter' or qtype == 'inter-chain' or qtype == 'union-chain':
            rel_len = 2
        else:
            rel_len = int(test_triples[0][-1].split('-')[0])

        model.eval()

        if qtype == 'inter-chain' or qtype == 'union-chain':
            test_dataloader_tail = DataLoader(
                TestInterChainDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        elif qtype == 'chain-inter':
            test_dataloader_tail = DataLoader(
                TestChainInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        elif 'inter' in qtype or 'union' in qtype:
            test_dataloader_tail = DataLoader(
                TestInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        else:
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )

        test_dataset_list = [test_dataloader_tail]
        # test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        logs = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode, query in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    assert batch_size == 1, batch_size

                    if 'inter' in qtype:
                        if model.geo == 'box':
                            _, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len,
                                                                          qtype, mode=mode)
                        else:
                            score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample),
                                                                              rel_len, qtype, mode=mode)
                    else:
                        score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len,
                                                                          qtype, mode=mode)

                    if model.geo == 'box':
                        score = score_cen
                        score2 = score_cen_plus

                    score -= (torch.min(score) - 1)
                    ans = test_ans[query]
                    hard_ans = test_ans_hard[query]
                    all_idx = set(range(args.nentity))
                    false_ans = all_idx - ans
                    ans_list = list(ans)
                    hard_ans_list = list(hard_ans)
                    false_ans_list = list(false_ans)
                    ans_idxs = np.array(hard_ans_list)
                    vals = np.zeros((len(ans_idxs), args.nentity))
                    vals[np.arange(len(ans_idxs)), ans_idxs] = 1
                    axis2 = np.tile(false_ans_list, len(ans_idxs))
                    axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
                    vals[axis1, axis2] = 1
                    b = torch.Tensor(vals) if not args.cuda else torch.Tensor(vals).cuda()
                    filter_score = b * score
                    argsort = torch.argsort(filter_score, dim=1, descending=True)
                    ans_tensor = torch.LongTensor(hard_ans_list) if not args.cuda else torch.LongTensor(
                        hard_ans_list).cuda()
                    argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
                    ranking = (argsort == 0).nonzero()
                    ranking = ranking[:, 1]
                    ranking = ranking + 1
                    if model.geo == 'box':
                        score2 -= (torch.min(score2) - 1)
                        filter_score2 = b * score2
                        argsort2 = torch.argsort(filter_score2, dim=1, descending=True)
                        argsort2 = torch.transpose(torch.transpose(argsort2, 0, 1) - ans_tensor, 0, 1)
                        ranking2 = (argsort2 == 0).nonzero()
                        ranking2 = ranking2[:, 1]
                        ranking2 = ranking2 + 1

                    ans_vec = np.zeros(args.nentity)
                    ans_vec[ans_list] = 1
                    hits1 = torch.sum((ranking <= 1).to(torch.float)).item()
                    hits3 = torch.sum((ranking <= 3).to(torch.float)).item()
                    hits10 = torch.sum((ranking <= 10).to(torch.float)).item()
                    mr = float(torch.sum(ranking).item())
                    mrr = torch.sum(1. / ranking.to(torch.float)).item()
                    hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
                    hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
                    hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
                    mrm = torch.mean(ranking.to(torch.float)).item()
                    mrrm = torch.mean(1. / ranking.to(torch.float)).item()
                    num_ans = len(hard_ans_list)
                    if model.geo == 'box':
                        hits1m_newd = torch.mean((ranking2 <= 1).to(torch.float)).item()
                        hits3m_newd = torch.mean((ranking2 <= 3).to(torch.float)).item()
                        hits10m_newd = torch.mean((ranking2 <= 10).to(torch.float)).item()
                        mrm_newd = torch.mean(ranking2.to(torch.float)).item()
                        mrrm_newd = torch.mean(1. / ranking2.to(torch.float)).item()
                    else:
                        hits1m_newd = hits1m
                        hits3m_newd = hits3m
                        hits10m_newd = hits10m
                        mrm_newd = mrm
                        mrrm_newd = mrrm

                    logs.append({
                        'MRRm_new': mrrm_newd,
                        'MRm_new': mrm_newd,
                        'HITS@1m_new': hits1m_newd,
                        'HITS@3m_new': hits3m_newd,
                        'HITS@10m_new': hits10m_newd,
                        'num_answer': num_ans
                    })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        num_answer = sum([log['num_answer'] for log in logs])
        for metric in logs[0].keys():
            if metric == 'num_answer':
                continue
            if 'm' in metric:
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            else:
                metrics[metric] = sum([log[metric] for log in logs]) / num_answer
        return metrics