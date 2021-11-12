#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import datetime

import gc
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model import DeepMatch
from data import ModelNet40
from utils import *


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def mkdir(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--emb_dims', type=int, default=128, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    mkdir(args)

    return args


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


def train_one_epoch(net, train_loader, opt):
    net.train()

    total_loss = 0
    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    eulers = []

    for src, target, rotation, translation, euler in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation = rotation.cuda()
        translation = translation.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
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
        loss = F.mse_loss(torch.matmul(rotation_pred.transpose(2, 1), rotation), identity) + \
               F.mse_loss(translation_pred, translation)
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

    rotations = np.concatenate(rotations, axis=0)
    translations = np.concatenate(translations, axis=0)
    rotations_pred = np.concatenate(rotations_pred, axis=0)
    translations_pred = np.concatenate(translations_pred, axis=0)

    eulers = np.concatenate(eulers, axis=0)

    return total_loss * 1.0 / num_examples, rotations, translations, rotations_pred, translations_pred, eulers


def train(args, net, train_loader, test_loader, boardio, textio):
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[5, 10], gamma=0.1)
    if args.unseen:
        scheduler = MultiStepLR(opt, milestones=[10, 20], gamma=0.1)

    best_test_loss = np.inf
    best_test_r_mse = np.inf
    best_test_r_mae = np.inf
    best_test_t_mse = np.inf
    best_test_t_mae = np.inf
    best_test_recall = np.inf

    for epoch in range(args.epochs):
        if not args.unseen:
            opt.step()
        scheduler.step()
        train_loss, train_rotations, train_translations, train_rotations_pred, \
        train_translations_pred, train_eulers = train_one_epoch(net, train_loader, opt)
        test_loss, test_rotations, test_translations, test_rotations_pred, \
        test_translations_pred, test_eulers = test_one_epoch(net, test_loader)

        train_rotations_pred_euler = npmat2euler(train_rotations_pred)
        train_r_mse = np.mean((train_rotations_pred_euler - np.degrees(train_eulers)) ** 2)
        train_r_abs = np.abs(train_rotations_pred_euler - np.degrees(train_eulers))
        train_r_mae = np.mean(train_r_abs)
        train_t_mse = np.mean((train_translations - train_translations_pred) ** 2)
        train_t_abs = np.abs(train_translations - train_translations_pred)
        train_t_mae = np.mean(train_t_abs)
        cnt, sm = 0, len(train_r_abs)
        for r, t in zip(train_r_abs, train_t_abs):
            if np.all(r < 1.0) and np.all(t < 0.1):
                cnt += 1
        train_recall = 100.00 * cnt / sm

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

        if best_test_loss >= test_loss:
            best_test_loss = test_loss

            best_test_r_mse = test_r_mse
            best_test_r_mae = test_r_mae

            best_test_t_mse = test_t_mse
            best_test_t_mae = test_t_mae
            best_test_recall = test_recall

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('EPOCH:: %d, Loss: %f rot_MSE: %f, rot_MAE: %f, trans_MSE: %f, trans_MAE: %f, recall: %f%%'
                      % (epoch, train_loss, train_r_mse, train_r_mae, train_t_mse, train_t_mae, train_recall))

        textio.cprint('==TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f rot_MSE: %f, rot_MAE: %f, trans_MSE: %f, trans_MAE: %f, recall: %f%%'
                      % (epoch, test_loss, test_r_mse, test_r_mae, test_t_mse, test_t_mae, test_recall))

        textio.cprint('==BEST TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f rot_MSE: %f, rot_MAE: %f, trans_MSE: %f, trans_MAE: %f, recall: %f%%'
                      % (epoch, best_test_loss, best_test_r_mse, best_test_r_mae, best_test_t_mse, best_test_t_mae,
                         best_test_recall))

        boardio.add_scalar('train/loss', train_loss, epoch)
        boardio.add_scalar('train/rotation/MSE', train_r_mse, epoch)
        boardio.add_scalar('train/rotation/MAE', train_r_mae, epoch)
        boardio.add_scalar('train/translation/MSE', train_t_mse, epoch)
        boardio.add_scalar('train/translation/MAE', train_t_mae, epoch)
        boardio.add_scalar('train/recall', train_recall, epoch)

        ############TEST
        boardio.add_scalar('test/loss', test_loss, epoch)
        boardio.add_scalar('test/rotation/MSE', test_r_mse, epoch)
        boardio.add_scalar('test/rotation/MAE', test_r_mae, epoch)
        boardio.add_scalar('test/translation/MSE', test_t_mse, epoch)
        boardio.add_scalar('test/translation/MAE', test_t_mae, epoch)
        boardio.add_scalar('test/recall', test_recall, epoch)

        ############BEST TEST
        boardio.add_scalar('best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('best_test/rotation/MSE', best_test_r_mse, epoch)
        boardio.add_scalar('best_test/rotation/MAE', best_test_r_mae, epoch)
        boardio.add_scalar('best_test/translation/MSE', best_test_t_mse, epoch)
        boardio.add_scalar('best_test/translation/MAE', best_test_t_mae, epoch)
        boardio.add_scalar('best_test/recall', best_test_recall, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


if __name__ == "__main__":
    args = get_args()
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    net = DeepMatch(args).cuda()

    test_loader = DataLoader(
        ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                   unseen=args.unseen, factor=args.factor),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    if args.eval:
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
    else:
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        start = datetime.datetime.now()
        train(args, net, train_loader, test_loader, boardio, textio)
        print("training time:", datetime.datetime.now() - start)

    print('FINISH')
    boardio.close()
