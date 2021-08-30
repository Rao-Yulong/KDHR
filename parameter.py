#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong

class para:
    def __init__(self, lr, rec, drop, batchSize, epoch, dev_ratio, test_ratio):
        self.lr = lr
        self.rec = rec
        self.drop = drop
        self.batchSize = batchSize
        self.epoch = epoch
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio