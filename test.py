#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data.dataload import ModelNet40
from init import init
from init import IOStream
from models.net import DeepMatch
from util import transform_point_cloud, npmat2euler

def test_one_epoch(net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    eulers = []

    for src, target, rotation, translation, euler in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation = rotation.cuda()
        translation = translation.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_pred, translation_pred = net(src, target)

        ## save rotation and translation
        rotations.append(rotation.detach().cpu().numpy())
        translations.append(translation.detach().cpu().numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())
        eulers.append(euler.numpy())

        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_pred.transpose(2, 1), rotation), identity) \
               + F.mse_loss(translation_pred, translation)

        total_loss += loss.item() * batch_size

    rotations = np.concatenate(rotations, axis=0)
    translations = np.concatenate(translations, axis=0)
    rotations_pred = np.concatenate(rotations_pred, axis=0)
    translations_pred = np.concatenate(translations_pred, axis=0)

    eulers = np.concatenate(eulers, axis=0)

    return total_loss * 1.0 / num_examples, rotations, translations, rotations_pred, translations_pred, eulers


def test(net, test_loader, textio):
    test_loss, test_rotations, test_translations, test_rotations_pred, \
    test_translations_pred, test_eulers = test_one_epoch(net, test_loader)

    test_rotations_pred_euler = npmat2euler(test_rotations_pred)
    test_r_mse = np.mean((test_rotations_pred_euler - np.degrees(test_eulers)) ** 2)
    test_r_abs = np.abs(test_rotations_pred_euler - np.degrees(test_eulers))
    test_r_mae = np.mean(test_r_abs)
    test_t_mse = np.mean((test_translations - test_translations_pred) ** 2)
    test_t_abs = np.abs(test_translations - test_translations_pred)
    test_t_mae = np.mean(test_t_abs)
    cnt, sm = 0, len(test_r_abs)
    for r, t in zip(test_r_abs, test_t_abs):
        if np.all(r < 1.0) and np.all(t < 0.1):
            cnt += 1
    test_recall = 100.00 * cnt / sm

    textio.cprint('==FINAL TEST==')
    textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_MAE: %f, trans_MSE: %f, trans_MAE: %f, recall: %f%%'
                  % (-1, test_loss, test_r_mse, test_r_mae, test_t_mse, test_t_mae, test_recall))


if __name__ == "__main__":
    args = init()
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    net = DeepMatch(args).cuda()

    test_loader = DataLoader(
        ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                   unseen=args.unseen, factor=args.factor),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    if args.model_path == '':
        model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
    else:
        model_path = args.model_path
        print(model_path)
    if not os.path.exists(model_path):
        print("can't find pretrained model")
    else:
        net.load_state_dict(torch.load(model_path), strict=False)

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        test(net, test_loader, textio)

        print('FINISH')
    boardio.close()