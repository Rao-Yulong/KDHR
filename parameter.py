#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong

class para:
    def __init__(self, lr, rec, drop, batchSize, epoch, ratio):
        self.lr = lr
        self.rec = rec
        self.drop = drop
        self.batchSize = batchSize
        self.epoch = epoch
        self.ratio = ratio