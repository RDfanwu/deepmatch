#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn

class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())

            R.append(r)

        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + tgt.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)