#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong

from utils import *
from model import *
import sys
import os
import parameter
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

para = parameter.para(lr=2e-4, rec=7e-3, drop=0.0, batchSize=512, epoch=100, ratio=0.8)

path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('KDHR.txt')

print(para.lr, para.rec, para.drop, para.batchSize, para.epoch, para.ratio)

"""创建3种图数据"""
# 读取S-H图
sh_edge = np.load('./data/sh_graph.npy')
sh_edge = sh_edge.tolist()
sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
sh_x = torch.tensor([[i] for i in range(1195)], dtype=torch.float)
sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous())

# S-S G
ss_edge = np.load('./data/ss_graph.npy')
ss_edge = ss_edge.tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
ss_x = torch.tensor([[i] for i in range(390)], dtype=torch.float)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous())

# H-H G
hh_edge = np.load('./data/hh_graph.npy').tolist()
hh_edge_index = torch.tensor(hh_edge, dtype=torch.long) - 390  # 边索引需要减去390
hh_x = torch.tensor([[i] for i in range(390, 1195)], dtype=torch.float)
hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous())

# 读取处方数据
prescript = pd.read_csv('./data/prescript_1195.csv', encoding='utf-8')
pLen = len(prescript) # 数据集的数量
# 症状的one-hot 矩阵
pS_list = [[0]*390 for _ in range(pLen)]
pS_array = np.array(pS_list)
# 草药的one-hot 矩阵
pH_list = [[0] * 805 for _ in range(pLen)]
pH_array = np.array(pH_list)
# 迭代数据集， 赋值
for i in range(pLen):
    j = eval(prescript.iloc[i, 0])
    pS_array[i, j] = 1

    k = eval(prescript.iloc[i, 1])
    k = [x - 390 for x in k]
    pH_array[i, k] = 1

# 读取中草药频率
herbCount = load_obj('./data/herbID2count')
herbCount = np.array(list(herbCount.values()))

# 读取KG中知识的独热编码
kg_oneHot = np.load('./data/herb_805_27_oneHot.npy')
kg_oneHot = torch.from_numpy(kg_oneHot).float()

# 训练集合测试集的下标
p_list = [x for x in range(pLen)]
x_train, x_test = train_test_split(p_list, test_size=1 - para.ratio, shuffle=False, random_state=2021)
print(len(x_train), len(x_test))# 27012 6753
train_dataset = presDataset(pS_array[x_train], pH_array[x_train])
test_dataset = presDataset(pS_array[x_test], pH_array[x_test])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=para.batchSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=para.batchSize)
# print(len(test_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KDHR(390, 805, 1195, 64, para.batchSize, para.drop)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.rec)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10 , gamma=0.8)
print('device: ', device)

loss_list = []
test_loss_list = []
test_p5_list = []
test_p10_list = []
test_p20_list = []
test_r5_list = []
test_r10_list = []
test_r20_list = []
test_n5_list = []
test_n10_list = []
test_n20_list = []
for epoch in range(para.epoch):

    model.train()
    running_loss = 0.0
    for i, (sid, hid) in enumerate(train_loader):
        # sid, hid = sid.to(device), hid.to(device)
        sid, hid = sid.float(), hid.float()
        optimizer.zero_grad()
        # batch*805 概率矩阵
        outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                        hh_data.x, hh_data.edge_index, sid, kg_oneHot)
        loss = criterion(outputs, hid)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print train loss per every epoch
    print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
    loss_list.append(running_loss / len(x_train))

    model.eval()
    test_loss = 0
    test_p5 = 0
    test_p10 = 0
    test_p20 = 0
    test_r5 = 0
    test_r10 = 0
    test_r20 = 0
    for tsid, thid in test_loader:
        tsid, thid = tsid.float(), thid.float()
        # batch*805 概率矩阵
        outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                        hh_data.x, hh_data.edge_index, tsid, kg_oneHot)
        test_loss += criterion(outputs, thid).item()

        # thid batch*805
        for i, hid in enumerate(thid):
            trueLabel = [] # 对应存在草药的索引
            for idx, val in enumerate(hid):  # 获得thid中值为一的索引
                if val == 1:
                    trueLabel.append(idx)


            top5 = torch.topk(outputs[i], 5)[1] # 预测值前5索引
            count = 0
            for m in top5:
                if m in trueLabel:
                    count += 1
            test_p5 += count / 5
            test_r5 += count / len(trueLabel)

            top10 = torch.topk(outputs[i], 10)[1]  # 预测值前10索引
            count = 0
            for m in top10:
                if m in trueLabel:
                    count += 1
            test_p10 += count / 10
            test_r10 += count / len(trueLabel)

            top20 = torch.topk(outputs[i], 20)[1]  # 预测值前20索引
            count = 0
            for m in top20:
                if m in trueLabel:
                    count += 1
            test_p20 += count / 20
            test_r20 += count / len(trueLabel)

    scheduler.step()

    print('[Epoch {}]test_loss: '.format(epoch + 1), test_loss / len(test_loader))
    print('p5-10-20:', test_p5 / len(x_test), test_p10 / len(x_test), test_p20 / len(x_test))
    print('r5-10-20:', test_r5 / len(x_test), test_r10 / len(x_test), test_r20 / len(x_test))
    test_loss_list.append(test_loss / len(x_test))

    test_p5_list.append(test_p5 / len(x_test))
    test_p10_list.append(test_p10 / len(x_test))
    test_p20_list.append(test_p20 / len(x_test))
    test_r5_list.append(test_r5 / len(x_test))
    test_r10_list.append(test_r10 / len(x_test))
    test_r20_list.append(test_r20 / len(x_test))


