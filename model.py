#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

class GCNConv_SH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SH, self).__init__(aggr='mean')  # 对邻居节点特征进行平均操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # 公式2
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)  # m = e*T 公式1
        return x_j

class GCNConv_SS_HH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SS_HH, self).__init__(aggr='add')  # 对邻居节点特征进行sum操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # 公式10
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)
        return x_j

class KDHR(torch.nn.Module):
    def __init__(self, ss_num, hh_num, sh_num, embedding_dim, batchSize, drop):
        super(KDHR, self).__init__()
        self.batchSize = batchSize
        self.dropout = drop
        self.SH_embedding = torch.nn.Embedding(sh_num, embedding_dim)
        # S-H 图所需的网络
        # S
        self.convSH_TostudyS_1 = GCNConv_SH(embedding_dim, embedding_dim)

        self.convSH_TostudyS_2 = GCNConv_SH(embedding_dim, embedding_dim)

        # self.convSH_TostudyS_3 = GCNConv_SH(embedding_dim, embedding_dim)

        self.SH_mlp_1 = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1 = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1 = torch.nn.Tanh()
        # H
        self.convSH_TostudyS_1_h = GCNConv_SH(embedding_dim, embedding_dim)

        self.convSH_TostudyS_2_h = GCNConv_SH(embedding_dim, embedding_dim)

        # self.convSH_TostudyS_3_h = GCNConv_SH(embedding_dim, embedding_dim)

        self.SH_mlp_1_h = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1_h = torch.nn.Tanh()
        # S-S图网络
        self.convSS = GCNConv_SS_HH(embedding_dim, 256)
        # H-H图网络  维度加上嵌入KG特征的维度
        self.convHH = GCNConv_SS_HH(embedding_dim+27, 256)
        # self.convHH = GCNConv_SS_HH(embedding_dim, 256)
        # SI诱导层
        # SUM
        self.mlp = torch.nn.Linear(256, 256)
        # cat
        # self.mlp = torch.nn.Linear(512, 512)
        self.SI_bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()

    def forward(self, x_SH, edge_index_SH, x_SS, edge_index_SS, x_HH, edge_index_HH, prescription, kgOneHot):
        # S-H图搭建
        # 第一层
        x_SH1 = self.SH_embedding(x_SH.long())
        x_SH2 = self.convSH_TostudyS_1(x_SH1.float(), edge_index_SH)
        # 第二层
        x_SH6 = self.convSH_TostudyS_2(x_SH2, edge_index_SH)
        # x_SH6 = x_SH6.view(-1, 256)
        # 第三层
        # x_SH7 = self.convSH_TostudyS_3(x_SH6, edge_index_SH)

        x_SH9 = (x_SH1 + x_SH2 + x_SH6 ) / 3.0
        x_SH9 = self.SH_mlp_1(x_SH9)
        x_SH9 = x_SH9.view(1195, -1)
        x_SH9 = self.SH_bn_1(x_SH9)
        x_SH9 = self.SH_tanh_1(x_SH9)
        # SH H
        x_SH11 = self.SH_embedding(x_SH.long())
        x_SH22 = self.convSH_TostudyS_1_h(x_SH11.float(), edge_index_SH)
        # 第二层
        x_SH66 = self.convSH_TostudyS_2_h(x_SH22, edge_index_SH)
        # x_SH66 = x_SH66.view(-1, 256)
        # 第三层
        # x_SH77 = self.convSH_TostudyS_3_h(x_SH66, edge_index_SH)

        x_SH99 = (x_SH11 + x_SH22 +x_SH66 ) / 3.0
        x_SH99 = self.SH_mlp_1_h(x_SH99)
        x_SH99 = x_SH99.view(1195, -1)
        x_SH99 = self.SH_bn_1_h(x_SH99)
        x_SH99 = self.SH_tanh_1_h(x_SH99)

        # S-S图搭建
        x_ss0 = self.SH_embedding(x_SS.long())
        x_ss1 = self.convSS(x_ss0.float(), edge_index_SS) # S_S图中 s的嵌入
        x_ss1 = x_ss1.view(390, -1)
        # H-H图搭建
        x_hh0 = self.SH_embedding(x_HH.long())
        x_hh0 = x_hh0.view(-1, 64)
        x_hh0 = torch.cat((x_hh0.float(), kgOneHot), dim=-1)
        x_hh1 = self.convHH(x_hh0.float(), edge_index_HH)  # H_H图中 h的嵌入
        x_hh1 = x_hh1.view(805, -1)
        # 信息融合
        # sum
        es = x_SH9[:390] + x_ss1  # 1195,1,64  390,1,64
        eh = x_SH99[390:] + x_hh1 # 805*dim
        # cat
        # es = torch.cat((x_SH9[:390], x_ss1), dim=-1)
        # eh = torch.cat((x_SH99[390:], x_hh1), dim=-1)
        # SI 集成多个症状为一个症状表示 batch*390 390*dim => batch*dim
        es = es.view(390,-1)
        e_synd = torch.mm(prescription, es)  # prescription * es
        # batch*1
        preSum = prescription.sum(dim=1).view(-1, 1)
        # batch*dim
        e_synd_norm = e_synd / preSum
        e_synd_norm = self.mlp(e_synd_norm)
        e_synd_norm = e_synd_norm.view(-1, 256)
        e_synd_norm = self.SI_bn(e_synd_norm)
        e_synd_norm = self.relu(e_synd_norm)  # batch*dim
        # batch*dim dim*805 => batch*805
        eh = eh.view(805, -1)
        pre = torch.mm(e_synd_norm, eh.t())

        return  pre








