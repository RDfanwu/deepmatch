#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.backbone import backBone
from models.svd import SVDHead


# Furthest Point Matching
class DeepMatch(nn.Module):
    def __init__(self, args):
        super(DeepMatch, self).__init__()
        self.emb_dims = args.emb_dims
        self.emb_nn = backBone(emb_dims=self.emb_dims)
        self.head = SVDHead()

    def forward(self, *input):
        # batch_size, num_dims, 3
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        rotation, translation = self.head(src_embedding, tgt_embedding, src, tgt)

        return rotation, translation
