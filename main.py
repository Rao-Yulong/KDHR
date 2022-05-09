#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong

from utils import *
from model import *
# from model_compara import Compare
# from model_SMGCN import SMGCN
import sys
import os
import parameter
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping
import time
from sklearn.metrics import roc_auc_score
import types
from torch_sparse import SparseTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if int(os.environ.get('CPU', 0)) == 1:
    device = torch.device('cpu')

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

para = parameter.para(lr=3e-4, rec=7e-3, drop=0.0, batchSize=int(os.environ.get('BATCH', 512)), epoch=200, dev_ratio=0.2, test_ratio=0.2)

path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('khdr.txt')

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
print("lr: ",para.lr, " rec: ", para.rec, " dropout: ",para.drop, " batchsize: ",
      para.batchSize, " epoch: ",para.epoch, " dev_ratio: ",para.dev_ratio, " test_ratio: ", para.test_ratio)


"""创建3种图数据"""
# 读取S-H图
sh_edge = np.load('./data/sh_graph.npy')
sh_edge = sh_edge.tolist()
sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
sh_x = torch.tensor([[i] for i in range(1195)], dtype=torch.float)
sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous())
sh_data_adj = SparseTensor(row=sh_data.edge_index[0], col=sh_data.edge_index[1],
                           sparse_sizes=(1195, 1195))
# S-S G
ss_edge = np.load('./data/ss_graph.npy')
ss_edge = ss_edge.tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
ss_x = torch.tensor([[i] for i in range(390)], dtype=torch.float)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous())
ss_data_adj = SparseTensor(row=ss_data.edge_index[0], col=ss_data.edge_index[1],
                           sparse_sizes=(390, 390))

# H-H G
hh_edge = np.load('./data/hh_graph.npy').tolist()
hh_edge_index = torch.tensor(hh_edge, dtype=torch.long) - 390  # 边索引需要减去390
hh_x = torch.tensor([[i] for i in range(390, 1195)], dtype=torch.float)
hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous())
hh_data_adj = SparseTensor(row=hh_data.edge_index[0], col=hh_data.edge_index[1],
                           sparse_sizes=(805, 805))

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
pS_array = torch.from_numpy(pS_array).to(device).float()
pH_array = torch.from_numpy(pH_array).to(device).float()
# 读取中草药频率
herbCount = load_obj('./data/herbID2count')
herbCount = np.array(list(herbCount.values()))

# 读取KG中知识的独热编码
kg_oneHot = np.load('./data/herb_805_27_oneHot.npy')
kg_oneHot = torch.from_numpy(kg_oneHot).float().to(device)

# 训练集开发集测试集的下标
p_list = [x for x in range(pLen)]
x_train, x_dev_test = train_test_split(p_list, test_size= (para.dev_ratio+para.test_ratio), shuffle=False,
                                       random_state=2021)
# print(len(x_train), len(x_dev_test))#

x_dev, x_test = train_test_split(x_dev_test, test_size=1 - 0.5, shuffle=False, random_state=2021)
print("train_size: ", len(x_train), "dev_size: ", len(x_dev), "test_size: ", len(x_test))


train_dataset = presDataset(pS_array[x_train], pH_array[x_train])
dev_dataset = presDataset(pS_array[x_dev], pH_array[x_dev])
test_dataset = presDataset(pS_array[x_test], pH_array[x_test])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=para.batchSize)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=para.batchSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=para.batchSize)
# print(len(test_loader))

model = KDHR(390, 805, 1195, 64, para.batchSize, para.drop).to(device)
# model = KDHR(390, 805, 1195, 64)



criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")


optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.rec)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7 , gamma=0.8)

early_stopping = EarlyStopping(patience=7, verbose=True)
print('device: ', device)


epsilon = 1e-13
sh_data = sh_data.to(device)
ss_data = ss_data.to(device)
hh_data = hh_data.to(device)
for epoch in range(para.epoch):

    model.train()
    running_loss = 0.0
    for i, (sid, hid) in enumerate(train_loader):
        optimizer.zero_grad()
        # batch*805 概率矩阵
        outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                        hh_data.x, hh_data.edge_index, sid, kg_oneHot)
        # outputs = model(sh_data.x, sh_data_adj, ss_data.x, ss_data_adj, hh_data.x, hh_data_adj, sid)
        loss = criterion(outputs, hid)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print train loss per every epoch
    print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
    # print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(x_train))
    # loss_list.append(running_loss / len(train_loader))

    model.eval()
    dev_loss = 0

    dev_p5 = 0
    dev_p10 = 0
    dev_p20 = 0

    dev_r5 = 0
    dev_r10 = 0
    dev_r20 = 0

    dev_f1_5 = 0
    dev_f1_10 = 0
    dev_f1_20 = 0
    for tsid, thid in dev_loader:
        # batch*805 概率矩阵
        torch.cuda.synchronize()
        s = time.time()
        outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                        hh_data.x, hh_data.edge_index, tsid, kg_oneHot)
        torch.cuda.synchronize()
        e = time.time()
        print('s',e-s)
        # outputs = model(sh_data.x, sh_data_adj, ss_data.x, ss_data_adj, hh_data.x, hh_data_adj, tsid)
        dev_loss += criterion(outputs, thid).item()

        torch.cuda.synchronize()
        s = time.time()
        # thid batch*805
        for i, hid in enumerate(thid):
            trueLabel = (hid==1).nonzero().flatten()
            top5 = torch.topk(outputs[i], 5)[1] # 预测值前5索引
            count = 0
            for m in top5:
                if m in trueLabel:
                    count += 1
            dev_p5 += count / 5
            dev_r5 += count / len(trueLabel)
            # dev_f1_5 += 2*(count / 5)*(count / len(trueLabel)) / ((count / 5) + (count / len(trueLabel)) + epsilon)

            top10 = torch.topk(outputs[i], 10)[1]  # 预测值前10索引
            count = 0
            for m in top10:
                if m in trueLabel:
                    count += 1
            dev_p10 += count / 10
            dev_r10 += count / len(trueLabel)
            # dev_f1_10 += 2 * (count / 10) * (count / len(trueLabel)) / ((count / 10) + (count / len(trueLabel)) + epsilon)

            top20 = torch.topk(outputs[i], 20)[1]  # 预测值前20索引
            count = 0
            for m in top20:
                if m in trueLabel:
                    count += 1
            dev_p20 += count / 20
            dev_r20 += count / len(trueLabel)
            # dev_f1_20 += 2 * (count / 20) * (count / len(trueLabel)) / ((count / 20) + (count / len(trueLabel)) + epsilon)
        torch.cuda.synchronize()
        e = time.time()
        print('v',e-s)

    scheduler.step()

    print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(dev_loader))
    # print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(x_dev))
    print('p5-10-20:', dev_p5 / len(x_dev), dev_p10 / len(x_dev), dev_p20 / len(x_dev))
    print('r5-10-20:', dev_r5 / len(x_dev), dev_r10 / len(x_dev), dev_r20 / len(x_dev))
    # print('f1_5-10-20: ', dev_f1_5 / len(x_dev), dev_f1_10 / len(x_dev), dev_f1_20 / len(x_dev))
    print('f1_5-10-20: ',
          2 * (dev_p5 / len(x_dev)) *(dev_r5 / len(x_dev))/((dev_p5 / len(x_dev))+(dev_r5 / len(x_dev))+ epsilon),
          2 * (dev_p10 / len(x_dev)) *(dev_r10 / len(x_dev))/((dev_p10 / len(x_dev))+(dev_r10 / len(x_dev))+ epsilon),
          2 * (dev_p20 / len(x_dev)) *(dev_r20 / len(x_dev))/((dev_p20 / len(x_dev))+(dev_r20 / len(x_dev))+ epsilon))

    early_stopping(dev_loss / len(dev_loader), model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
# 获得 early stopping 时的模型参数
model.load_state_dict(torch.load('checkpoint.pt'))

model.eval()
test_loss = 0

test_p5 = 0
test_p10 = 0
test_p20 = 0

test_r5 = 0
test_r10 = 0
test_r20 = 0

test_f1_5 = 0
test_f1_10 = 0
test_f1_20 = 0


for tsid, thid in test_loader:
    # batch*805 概率矩阵
    outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,hh_data.x, hh_data.edge_index, tsid, kg_oneHot)

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

print("----------------------------------------------------------------------------------------------------")

print('test_loss: ', test_loss / len(test_loader))

print('p5-10-20:', test_p5 / len(x_test), test_p10 / len(x_test), test_p20 / len(x_test))
print('r5-10-20:', test_r5 / len(x_test), test_r10 / len(x_test), test_r20 / len(x_test))

print('f1_5-10-20: ',
      2 * (test_p5 / len(x_test)) * (test_r5 / len(x_test)) / ((test_p5 / len(x_test)) + (test_r5 / len(x_test))),
      2 * (test_p10 / len(x_test)) * (test_r10 / len(x_test)) / ((test_p10 / len(x_test)) + (test_r10 / len(x_test))),
      2 * (test_p20 / len(x_test)) * (test_r20 / len(x_test)) / ((test_p20 / len(x_test)) + (test_r20 / len(x_test))))

torch.save(model.state_dict(), '/vc_data/users/t-zilongwang/temp.pt')


