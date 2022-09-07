import time
import argparse
import datetime

import torch
import torch.nn as nn
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import relative_depth
import relative_depth_for_figure

from model import create_model
from get_data import getTestingData
from tensorboardX import SummaryWriter
from utils import AverageMeter, DepthNorm, colorize, compute_errors, compute_correlation, print_scores, print_metrics, get_model_summary
from multi_loss import get_loss_weights, compute_multi_loss, compute_multi_metric, get_loss_1batch, get_metric_1batch, log_write_test, get_loss_valid, get_metric_valid
from save_prediction import pred2png

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Arguments
    parser = argparse.ArgumentParser(description='MDE, JVCI 2022')
    parser.add_argument('--backbone', default='DenseNet161', type=str, help='DenseNet161 (bs8) / PNASNet5Large (bs6) / PNASNet5LargeMax (bs5) / PNASNet5LargeMin (bs6) / NASNetALarge (bs6)')
    parser.add_argument('--testing_mode', default='OrdinaryRelativeDepth', type=str, help='Default / OrdinaryRelativeDepth')
    parser.add_argument('--num_neighborhood', default=24, type=int, help='num_neighborhood for OrdinaryRelativeDepth')
    parser.add_argument('--relative_loss_weight', default=10.0, type=float, help='weight for RelativeDepth')
    parser.add_argument('--decoder_scale', default=768, type=int, help='valid when using PNASNet5LargeMin')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--iter_per_epoch', default=7231, type=float)
    parser.add_argument('--input_image_size_height', default=288, type=int)
    parser.add_argument('--input_image_size_width', default=384, type=int)
    parser.add_argument('--save_prediction', default=False, type=bool)
    # all arguments
    args = parser.parse_args()

    # dataset using
    test_dataset_use = {'NYUv2_test': True, 'NYUv2_test_raw': False,
                        'KITTI_Eigen_test': False,
                        'Make3D_test': False}

    # image size
    original_image_size = [480, 640]
    input_image_size = [args.input_image_size_height, args.input_image_size_width]
    # interpolation function / relu
    interpolate_bicubic_fullsize = nn.Upsample(size=original_image_size, mode='bicubic')
    interpolate_bicubic_inputsize = nn.Upsample(size=input_image_size, mode='bicubic')
    relu = nn.ReLU()

    # create model
    testing_mode = args.testing_mode
    backbone = args.backbone
    decoder_scale = args.decoder_scale
    num_neighborhood = args.num_neighborhood
    relative_loss_weight = args.relative_loss_weight
    if testing_mode == 'Default':
        model_name = backbone
    elif testing_mode == 'OrdinaryRelativeDepth':
        model_name = backbone + '_' + testing_mode

    model = create_model(model_name, decoder_scale, num_neighborhood)
    print('Summary: All Network')
    print(get_model_summary(model, torch.rand(1, 3, input_image_size[0], input_image_size[1]).cuda(), verbose=True))
    model = nn.DataParallel(model).half()
    print('Model created.')

    # Training parameters
    batch_size = args.bs
    iter_per_epoch = args.iter_per_epoch

    # loading training/testing data
    test_loader, num_test_data = getTestingData(batch_size, test_dataset_use, data_transform_setting=testing_mode, num_neighborhood=num_neighborhood)

    # model path
    model_path = 'runs\D161_ORD_b08_scale0768_nb24\data-202004241509' # for ur

    # Start testing
    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(test_loader)

    end = time.time()

    for epoch in range(23,24): #range(12,100):
        for iter in [7231]: #[1807, 3615, 5423, 7231]:

            # prediction save path
            predPath = ''
            if args.save_prediction == True:
                predPath = model_path + '/prediction/epoch' + str(epoch + 1).zfill(2) + '_iter' + str(iter).zfill(5)
                if os.path.isdir(predPath) == False:
                    os.mkdir(predPath)

            # current_epoch (ex: 1.25 epoch = 1250 current epoch)
            current_epoch = math.ceil((epoch + iter / iter_per_epoch) * 1000)

            # load
            model_name = model_path + "/epoch"+ str(epoch+1).zfill(2) + '_iter' + str(iter).zfill(5) + ".pth"
            model.load_state_dict(torch.load(model_name))
            model.eval()
            print(model_name)

            for i, sample_batched in enumerate(test_loader):

                if testing_mode == 'Default':
                    # Prepare sample and target
                    image_full = torch.autograd.Variable(sample_batched['image'].cuda()).half()
                    depth_gt_full = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

                    # depth gt
                    depth_gt_input = interpolate_bicubic_inputsize(depth_gt_full)
                    depth_gt_for_metric = depth_gt_full[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24]

                    depth_gt_for_loss = depth_gt_for_loss.cuda(torch.device("cuda:0"))
                    depth_gt_for_metric = depth_gt_for_metric.cuda(torch.device("cuda:0"))

                    # Predict
                    image_input = interpolate_bicubic_inputsize(image_full)
                    depth_pred_for_loss = torch.exp(model(image_input)[5])
                    depth_pred_for_loss = relu(depth_pred_for_loss - 0.0001) + 0.0001
                    tensorboard_prefix = 'test'

                    depth_pred_for_loss = interpolate_bicubic_inputsize(depth_pred_for_loss).float()
                    depth_pred_full = interpolate_bicubic_fullsize(depth_pred_for_loss)
                    depth_pred_for_metric = relu(depth_pred_full[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24] - 0.0001) + 0.0001

                    # current batch size
                    current_batch_size = image_input.size(0)

                    # save prediction
                    if args.save_prediction == True:
                        for index_test in range(i*batch_size + 1, i*batch_size + current_batch_size + 1):
                            pred2png(depth_pred_full[index_test - (i*batch_size + 1), 0, :, :].cpu().detach().numpy(), predPath, index_test, 'depth_indoor')

                elif testing_mode == 'OrdinaryRelativeDepth':
                    # prepare sample and target
                    image = sample_batched['image']
                    depth = sample_batched['depth']
                    depth_ORD = sample_batched['depth_ORD']

                    # depth gt
                    depth_gt_for_metric = torch.exp(depth[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24])

                    # predict
                    image_input = interpolate_bicubic_inputsize(image).cuda().half()
                    time_start = time.time()
                    depth_pred = model(image_input)
                    #print('network: ' + str(time.time() - time_start))


                    depth_pred_detached = []
                    for index_pred in range(len(depth_pred)):
                        depth_pred_detached.append(depth_pred[index_pred].float().cpu().detach())

                    depth_use = [  'valid', 'invalid', 'invalid', 'invalid', 'invalid', 'invalid',     'ALS',    'ALS',    'ALS',    'prop',    'prop',    'prop']
                    #depth_use = ['valid', 'valid', 'valid', 'valid', 'valid', 'valid',
                    #             'ALS', 'ALS', 'ALS', 'ALS', 'ALS', 'prop']
                    #depth_use = ['valid', 'valid', 'valid', 'valid', 'valid', 'valid',
                    #             'invalid', 'invalid','invalid','invalid','invalid','invalid']
                    if i+1 in [55, 88, 132, 175, 231, 275, 384, 392, 410, 4, 16, 26, 164, 227, 303, 328, 501, 624]:
                        a=0
                        relative_depth_for_figure.depth_combination(depth_pred_detached, depth_use=depth_use, name='depth_test' + str(i + 1).zfill(3))
                        #depth_pred_for_metric = torch.exp(relative_depth.depth_combination(depth_pred_detached, depth_use=depth_use))
                        #if i == 0:
                        #    time_check = time.time() - time_start
                        #else:
                        #    time_check += time.time() - time_start
                        #print('image: ', str(i+1), '   time: ', str(time_check))
                        #depth_pred_for_metric = interpolate_bicubic_fullsize(depth_pred_for_metric)[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24]

                    # current batch size
                    current_batch_size = image_input.size(0)

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

                if i == 0 or i % 20 == 19 or i == N-1:
                    # Print to console
                    print('Image: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                          'ETA {eta}'
                          .format(i+1, N, batch_time=batch_time, eta=eta))

            print('------------------------ FINISH -------------------------')
            print('---------------------------------------------------------')


if __name__ == '__main__':
    main()
