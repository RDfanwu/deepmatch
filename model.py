#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_furthest(x):
    # batch_size * 3 * num_points
    inner = 2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx + xx.transpose(2, 1).contiguous() - inner

    dist, idx = pairwise_distance.max(dim=-1)  # batch_size * num_points
    return dist, idx


def get_graph_feature(x):
    batch_size, _, num_points = x.size()
    x = x - x.mean(dim=-1, keepdim=True)
    xx = torch.sum(x ** 2, dim=1, keepdim=False)
    x *= 1.0 / (torch.max(xx) ** 0.5)

    dist, idx = get_furthest(x)
    dist = dist.view(batch_size, num_points, 1, 1)

    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    furthest = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k) * 3
    xx = torch.sum(x ** 2, dim=-1, keepdim=True).view(batch_size, num_points, 1, -1)
    furthest_dist = torch.sum(furthest ** 2, dim=-1, keepdim=True).view(batch_size, num_points, 1, -1)
    feature = torch.cat((xx, furthest_dist, dist), dim=-1)  # batch_size * num_ points * 1 * 3

    return feature.permute(0, 3, 1, 2)


class BackBone(nn.Module):
    def __init__(self, emb_dims=128):
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        return x.view(batch_size, -1, num_points)


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


# Furthest Point Matching
class DeepMatch(nn.Module):
    def __init__(self, args):
        super(DeepMatch, self).__init__()
        self.emb_dims = args.emb_dims
        self.emb_nn = BackBone(emb_dims=self.emb_dims)
        self.head = SVDHead()

    def forward(self, *input):
        # batch_size, num_dims, 3
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        rotation, translation = self.head(src_embedding, tgt_embedding, src, tgt)

        return rotation, translation
