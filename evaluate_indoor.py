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
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--iter_per_epoch', default=7231, type=float)
    parser.add_argument('--input_image_size_height', default=288, type=int)
    parser.add_argument('--input_image_size_width', default=384, type=int)
    parser.add_argument('--save_prediction', default=True, type=bool)
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

    # log write
    writer = SummaryWriter(model_path, flush_secs=30)

    # prediction path
    if os.path.isdir(model_path + '/prediction') == False:
        os.mkdir(model_path + '/prediction')
    # loss save path
    if os.path.isdir(model_path + '/loss') == False:
        os.mkdir(model_path + '/loss')

    # prediction path
    if os.path.isdir(model_path + '/prediction') == False:
        os.mkdir(model_path + '/prediction')

    # Start testing
    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(test_loader)

    end = time.time()

    # depth scores
    test_scores = np.zeros((num_test_data, 150))  # 135 losses
    test_metrics = np.zeros((num_test_data, 25)) # 25 metrics

    for epoch in range(23,24): #range(12,100):
        for iter in [7231]: #[1807, 3615, 5423, 7231]:

            loss_weights = []
            loss_valid = []
            metric_valid = []
            if testing_mode == 'Default':
                # load - loss weights
                index_type = 0
                savePath = model_path + "/train_loss_weights" + str(epoch + 1).zfill(2) + '_iter' + str(iter).zfill(5) + '_type' + str(index_type).zfill(2) + ".csv"
                loss_weights.append(utils.read_loss_weights(savePath=savePath))
                loss_valid.append(get_loss_valid(valid=True))
                metric_valid.append(get_metric_valid(valid=True))
            elif testing_mode == 'OrdinaryRelativeDepth':
                # load - loss weights
                for index_type in range(12):
                    savePath = model_path + "/train_loss_weights" + str(epoch + 1).zfill(2) + '_iter' + str(iter).zfill(5) + '_type' + str(index_type).zfill(2) + ".csv"
                    loss_weights.append(utils.read_loss_weights(savePath=savePath))
                    loss_valid.append(get_loss_valid(valid=True))
                    metric_valid.append(get_metric_valid(valid=True))

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
                    depth_gt_for_loss = depth_gt_input
                    depth_gt_for_metric = depth_gt_full[:, :, 0 + 44:480 - 9, 0 + 40:640 - 39]

                    depth_gt_for_loss = depth_gt_for_loss.cuda(torch.device("cuda:0"))
                    depth_gt_for_metric = depth_gt_for_metric.cuda(torch.device("cuda:0"))

                    # Predict
                    image_input = interpolate_bicubic_inputsize(image_full)
                    depth_pred_for_loss = torch.exp(model(image_input)[5])
                    depth_pred_for_loss = relu(depth_pred_for_loss - 0.0001) + 0.0001
                    tensorboard_prefix = 'test'

                    depth_pred_for_loss = interpolate_bicubic_inputsize(depth_pred_for_loss).float()
                    depth_pred_full = interpolate_bicubic_fullsize(depth_pred_for_loss)
                    depth_pred_for_metric = relu(depth_pred_full[:, :, 0 + 44:480 - 9, 0 + 40:640 - 39] - 0.0001) + 0.0001

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
                    depth_gt_for_metric = torch.exp(depth[:, :, 0 + 44:480 - 9, 0 + 40:640 - 39])
                    depth_gt_for_loss = depth_ORD
                    #for index_type in range(len(loss_weights)):
                    #    depth_gt_for_loss[index_type] = depth_gt_for_loss[index_type].cuda().half()

                    # predict
                    image_input = interpolate_bicubic_inputsize(image).cuda().half()
                    depth_pred = model(image_input)

                    depth_pred_detached = []
                    for index_pred in range(len(depth_pred)):
                        depth_pred_detached.append(depth_pred[index_pred].float().cpu().detach())

                    #depth_use = [  'valid', 'invalid', 'invalid', 'invalid', 'invalid', 'invalid',     'ALS',    'prop',    'prop',    'prop',    'prop',    'prop']
                    depth_use = ['valid', 'invalid', 'invalid', 'invalid', 'invalid', 'valid',
                                 'ALS',    'ALS',    'ALS',    'prop',    'prop',    'prop']
                    depth_pred_for_metric = torch.exp(relative_depth.depth_combination(depth_pred_detached, depth_use=depth_use))

                    # current batch size
                    current_batch_size = image_input.size(0)

                    # save prediction
                    if args.save_prediction == True:
                        for index_test in range(i*batch_size + 1, i*batch_size + current_batch_size + 1):
                            pred2png(depth_pred_for_metric[index_test - (i*batch_size + 1), 0, :, :].cpu().detach().numpy(), predPath, index_test, 'depth_indoor')

                    depth_pred_for_metric = interpolate_bicubic_fullsize(depth_pred_for_metric)[:, :, 0 + 44:480 - 9, 0 + 40:640 - 39]

                # compute metric
                rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8, \
                si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8, \
                corr_pearson, corr_spearman, corr_kendal \
                    = compute_multi_metric(depth_pred_for_metric.cuda(), depth_gt_for_metric.cuda(), metric_valid[5])

                """
                # compute loss
                l_rmse, l_rmse_log, l_abs_rel, l_sqr_rel, l_log10, l_delta1, l_delta2, l_delta3, l_metric3, l_metric8, \
                l_si_rmse, l_si_rmse_log, l_si_abs_rel, l_si_sqr_rel, l_si_log10, l_si_delta1, l_si_delta2, l_si_delta3, l_si_metric3, l_si_metric8, l_depth, \
                l_depth_dx, l_depth_dy, l_depth_norm, l_depth_dx2, l_depth_dxy, l_depth_dy2, l_depth_dx_norm, l_depth_dy_norm, l_ssim, l_ndepth, l_ndepth_win5, l_ndepth_win17, l_ndepth_win65, l_geo, \
                l_log_depth, l_log_depth_dx, l_log_depth_dy, l_log_depth_norm, l_log_depth_dx2, l_log_depth_dxy, l_log_depth_dy2, l_log_depth_dx_norm, l_log_depth_dy_norm, l_log_ssim, l_log_ndepth, l_log_ndepth_win5, l_log_ndepth_win17, l_log_ndepth_win65, l_log_geo, \
                l_inv_depth, l_inv_depth_dx, l_inv_depth_dy, l_inv_depth_norm, l_inv_depth_dx2, l_inv_depth_dxy, l_inv_depth_dy2, l_inv_depth_dx_norm, l_inv_depth_dy_norm, l_inv_ssim, l_inv_ndepth, l_inv_ndepth_win5, l_inv_ndepth_win17, l_inv_ndepth_win65, l_inv_geo, l_all_geo, \
                l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm, l_down1_ssim, l_down1_ndepth, l_down1_ndepth_win5, l_down1_ndepth_win17, l_down1_ndepth_win65, l_down1_geo, \
                l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm, l_down2_ssim, l_down2_ndepth, l_down2_ndepth_win5, l_down2_ndepth_win17, l_down2_ndepth_win65, l_down2_geo, \
                l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm, l_down3_ssim, l_down3_ndepth, l_down3_ndepth_win5, l_down3_ndepth_win17, l_down3_ndepth_win65, l_down3_geo, \
                l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm, l_down4_ssim, l_down4_ndepth, l_down4_ndepth_win5, l_down4_ndepth_win17, l_down4_ndepth_win65, l_down4_geo, \
                l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm, l_down5_ssim, l_down5_ndepth, l_down5_ndepth_win5, l_down5_ndepth_win17, l_down5_ndepth_win65, l_down5_geo \
                    = compute_multi_loss(depth_pred_for_loss.cuda(), depth_gt_for_loss[5].cuda(), loss_weights[5], loss_valid[5])
                """

                """
                with torch.no_grad():
                    show_image = np.squeeze(np.transpose(image.cpu(), (2, 3, 1, 0)))
                    show_gt_depth = np.squeeze(np.transpose(depth_gt_for_metric.cpu(), (2, 3, 1, 0)))
                    show_pred_depth = np.squeeze(np.transpose(depth_pred_for_metric.cpu(), (2, 3, 1, 0)))
                    plt.subplot(1, 3, 1)
                    plt.imshow(show_image)
                    plt.subplot(1, 3, 2)
                    plt.imshow(show_gt_depth)
                    plt.subplot(1, 3, 3)
                    plt.imshow(show_pred_depth)
                """

                """
                # compute iter loss & test_scores
                loss, l_custom, test_scores = get_loss_1batch(batch_size, current_batch_size, i, num_test_data, loss_weights[5], test_scores,
                        l_rmse, l_rmse_log, l_abs_rel, l_sqr_rel, l_log10, l_delta1, l_delta2, l_delta3, l_metric3, l_metric8,
                        l_si_rmse, l_si_rmse_log, l_si_abs_rel, l_si_sqr_rel, l_si_log10, l_si_delta1, l_si_delta2, l_si_delta3, l_si_metric3, l_si_metric8, l_depth,
                        l_depth_dx, l_depth_dy, l_depth_norm, l_depth_dx2, l_depth_dxy, l_depth_dy2, l_depth_dx_norm, l_depth_dy_norm, l_ssim, l_ndepth, l_ndepth_win5, l_ndepth_win17, l_ndepth_win65, l_geo,
                        l_log_depth, l_log_depth_dx, l_log_depth_dy, l_log_depth_norm, l_log_depth_dx2, l_log_depth_dxy, l_log_depth_dy2, l_log_depth_dx_norm, l_log_depth_dy_norm, l_log_ssim, l_log_ndepth, l_log_ndepth_win5, l_log_ndepth_win17, l_log_ndepth_win65, l_log_geo,
                        l_inv_depth, l_inv_depth_dx, l_inv_depth_dy, l_inv_depth_norm, l_inv_depth_dx2, l_inv_depth_dxy, l_inv_depth_dy2, l_inv_depth_dx_norm, l_inv_depth_dy_norm, l_inv_ssim, l_inv_ndepth, l_inv_ndepth_win5, l_inv_ndepth_win17, l_inv_ndepth_win65, l_inv_geo, l_all_geo,
                        l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm, l_down1_ssim, l_down1_ndepth, l_down1_ndepth_win5, l_down1_ndepth_win17, l_down1_ndepth_win65, l_down1_geo,
                        l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm, l_down2_ssim, l_down2_ndepth, l_down2_ndepth_win5, l_down2_ndepth_win17, l_down2_ndepth_win65, l_down2_geo,
                        l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm, l_down3_ssim, l_down3_ndepth, l_down3_ndepth_win5, l_down3_ndepth_win17, l_down3_ndepth_win65, l_down3_geo,
                        l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm, l_down4_ssim, l_down4_ndepth, l_down4_ndepth_win5, l_down4_ndepth_win17, l_down4_ndepth_win65, l_down4_geo,
                        l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm, l_down5_ssim, l_down5_ndepth, l_down5_ndepth_win5, l_down5_ndepth_win17, l_down5_ndepth_win65, l_down5_geo)
                """

                test_metrics = get_metric_1batch(batch_size, current_batch_size, i, num_test_data, test_metrics,
                                                 rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8,
                                                 si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8,
                                                 corr_pearson, corr_spearman, corr_kendal)

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

            #test_scores_mean = test_scores.mean(axis=0)
            test_metrics_mean = test_metrics.mean(axis=0)
            #loss = np.sum(np.array(loss_weights[5].cpu()) * np.array(test_scores_mean))

            #print_scores(test_scores_mean)
            print('   ')
            print_metrics(test_metrics_mean)

            # save - each image score
            #savePath = model_path + "/" + tensorboard_prefix + "_epoch" + str(0 + epoch + 1).zfill(2) + '_iter' + str(iter).zfill(5) + ".csv"
            #dataframe = pd.DataFrame(test_scores)
            #dataframe.to_csv(savePath, header=False, index=False)
            # save - mean score
            #savePath = model_path + "/" + tensorboard_prefix + "_mean_epoch" + str(0 + epoch + 1).zfill(2) + '_iter' + str(iter).zfill(5) + ".csv"
            #dataframe = pd.DataFrame(test_scores_mean)
            #dataframe.to_csv(savePath, header=False, index=False)

            # save - each image metrics
            #savePath = model_path + "/" + tensorboard_prefix + "_metrics_epoch" + str(0 + epoch + 1).zfill(2) + '_iter' + str(iter).zfill(5) + ".csv"
            #dataframe = pd.DataFrame(test_metrics)
            #dataframe.to_csv(savePath, header=False, index=False)
            # save - mean metrics
            #savePath = model_path + "/" + tensorboard_prefix + "_metrics_mean_epoch" + str(0 + epoch + 1).zfill(2) + '_iter' + str(iter).zfill(5) + ".csv"
            #dataframe = pd.DataFrame(test_metrics_mean)
            #dataframe.to_csv(savePath, header=False, index=False)

            print('------------------------ FINISH -------------------------')
            print('---------------------------------------------------------')


if __name__ == '__main__':
    main()
