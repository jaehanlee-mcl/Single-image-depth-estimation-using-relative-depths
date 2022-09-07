import time
import argparse
import datetime
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from model import create_model
from tensorboardX import SummaryWriter
from get_data import getTrainingData
from utils import AverageMeter, print_scores, print_scores_one, get_model_summary
from multi_loss import get_loss_weights, compute_multi_loss, compute_multi_metric, get_loss_1batch, log_write_train, get_loss_valid, get_metric_valid, get_loss_initialize_scale, compute_1loss_with_record
from results_record import make_model_path
try:
    try:
        from apex import amp
        APEX_AVAILABLE = True
    except ModuleNotFoundError:
        APEX_AVAILABLE = False
except:
    APEX_AVAILABLE = False

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Arguments
    parser = argparse.ArgumentParser(description='depth map estimation (CVPR2020 submission)')
    parser.add_argument('--backbone', default='DenseNet161', type=str,
                        help='DenseNet161 (bs12) / PNASNet5LargeMin (bs6)')
    parser.add_argument('--training_mode', default='OrdinaryRelativeDepth', type=str, help='Default / OrdinaryRelativeDepth')
    parser.add_argument('--num_neighborhood', default=24, type=int, help='num_neighborhood for OrdinaryRelativeDepth')
    parser.add_argument('--relative_loss_weight', default=10.0, type=float, help='weight for RelativeDepth')
    parser.add_argument('--decoder_scale', default=768, type=int, help='valid when using PNASNet5LargeMin')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--evaluation', default=False, type=bool)
    parser.add_argument('--num_save', default=4, type=int)
    parser.add_argument('--weight_flattening', default=False, type=bool)
    parser.add_argument('--weight_adjustment', default=False, type=bool)
    parser.add_argument('--num_weight_arrangement', default=4, type=int)
    parser.add_argument('--lambda_for_adjust_start', default=0, type=float)
    parser.add_argument('--lambda_for_adjust_slope', default=0, type=float)
    parser.add_argument('--lambda_for_adjust_min', default=0, type=float)
    parser.add_argument('--input_image_size_height', default=288, type=int)
    parser.add_argument('--input_image_size_width', default=384, type=int)
    parser.add_argument('--loss_function_space', default='1loss', type=str,
                        help='78loss / 13loss / 5loss / 1loss / 78loss-same / 13loss-same / 5loss-same')
    parser.add_argument('--loss_initialize_type', default='1loss', type=str,
                        help='78loss-same / 78loss-cluster / 13loss-same / 13loss-cluster / 5loss-same / 5loss-cluster / 1loss')
    # all arguments
    args = parser.parse_args()

    # dataset using
    train_dataset_use = {'NYUv2_train_reduced01': False, 'NYUv2_train_reduced05': True, 'NYUv2_train_reduced06': False,
                         'NYUv2_train_reduced10': False,
                         'NYUv2_train_reduced15': False, 'NYUv2_train_reduced20': False, 'NYUv2_train_reduced30': False,
                         'SUNRGB_D_train_reduced01': False, 'SUNRGB_D_train_reduced02': False,
                         'Matterport3D_train_reduced01': False, 'Matterport3D_train_reduced05': False,
                         'Matterport3D_train_reduced10': False,
                         'KITTI_Eigen_train_reduced01': False, 'KITTI_Eigen_train_reduced03': False,
                         'Make3D_train_reduced01': False,
                         'ReDWeb_V1_train_reduced01': False}

    # image size
    original_image_size = [480, 640]
    input_image_size = [args.input_image_size_height, args.input_image_size_width]
    # interpolation function / relu
    interpolate_bicubic_fullsize = nn.Upsample(size=original_image_size, mode='bicubic')
    interpolate_bicubic_inputsize = nn.Upsample(size=input_image_size, mode='bicubic')
    relu = nn.ReLU()

    # create model
    training_mode = args.training_mode
    backbone = args.backbone
    decoder_scale = args.decoder_scale
    num_neighborhood = args.num_neighborhood
    relative_loss_weight = args.relative_loss_weight
    if training_mode == 'Default':
        model_name = backbone
    elif training_mode == 'OrdinaryRelativeDepth':
        model_name = backbone + '_' + training_mode

    model = create_model(model_name, decoder_scale, num_neighborhood)
    print('Summary: All Network')
    print(get_model_summary(model, torch.rand(1, 3, input_image_size[0], input_image_size[1]).cuda(), verbose=True))
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = args.bs

    # loading training/testing data
    num_test_data = 654
    train_loader, num_train_data = getTrainingData(batch_size, train_dataset_use, data_transform_setting=training_mode, num_neighborhood=num_neighborhood)
    # iter/epoch
    iter_per_epoch = len(train_loader)
    # save iteration
    num_save = args.num_save
    num_weight_arrangement = args.num_weight_arrangement
    iter_list_save = np.zeros((num_save, 1))
    for i in range(num_save):
        iter_list_save[i] = math.ceil((i+1) * iter_per_epoch/num_save) - 1
    iter_list_arrangement = np.zeros((num_weight_arrangement, 1))
    for i in range(num_weight_arrangement):
        iter_list_arrangement[i] = math.ceil((i+1) * iter_per_epoch/num_weight_arrangement) - 1

    # weight arrangement argument
    weight_flattening = args.weight_flattening
    weight_adjustment = args.weight_adjustment
    TF_weight_flattening = False
    last_arrangement_iter = 0
    previous_total_loss = []
    previous_loss = []

    # Model path
    model_path = make_model_path(model_name, decoder_scale, batch_size)

    train_scores = []
    loss_weights = []
    loss_initialize_scale = []
    loss_valid = []
    if training_mode == 'Default':
        index_type = 0
        train_scores.append(np.zeros((num_train_data, 150)))  # 150 metrics
        loss_weights.append(get_loss_weights(args.loss_function_space))
        loss_initialize_scale.append(get_loss_initialize_scale(loss_initialize_type=args.loss_initialize_type))
        loss_valid.append(np.array(loss_weights[index_type]) > 0)
        previous_total_loss.append(0)
        previous_loss.append(0)
        # save path
        savePath = model_path + '/loss_weights_type' + str(index_type).zfill(2) + '.csv'
        dataframe = pd.DataFrame(loss_weights[index_type])
        dataframe.to_csv(savePath, header=False, index=False)
    elif training_mode == 'OrdinaryRelativeDepth':
        for index_type in range(12):
            train_scores.append(np.zeros((num_train_data, 150)))  # 150 metrics
            loss_weights.append(get_loss_weights(args.loss_function_space))
            loss_initialize_scale.append(get_loss_initialize_scale(loss_initialize_type=args.loss_initialize_type))
            loss_valid.append(np.array(loss_weights[index_type]) > 0)
            previous_total_loss.append(0)
            previous_loss.append(0)
            # save path
            savePath = model_path + '/loss_weights_type' + str(index_type).zfill(2) + '.csv'
            dataframe = pd.DataFrame(loss_weights[index_type])
            dataframe.to_csv(savePath, header=False, index=False)

    # mixed precision + Dataparallel
    if APEX_AVAILABLE == True:
        use_amp = True
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2",
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )
    else:
        use_amp = False
    model = nn.DataParallel(model)

    try:
        # try to load epoch1_iter00000
        model_name = "epoch01_iter00000.pth"
        model.load_state_dict(torch.load(model_name))
        print('LOAD MODEL ', model_name)
    except:
        # save model
        print('THERE IS NO MODEL TO LOAD')
        model_name = model_path + "/epoch" + str(0 + 1).zfill(2) + '_iter' + str(0).zfill(5) + ".pth"
        print('SAVE MODEL:' + model_path)
        torch.save(model.state_dict(), model_name)

    # Start training...
    for epoch in range(args.epochs):
        print('---------------------------------------------------------')
        print('-------------- TRAINING OF EPOCH ' + str(0 + epoch + 1).zfill(2) + 'START ----------------')
        batch_time = AverageMeter()

        end = time.time()

        # Switch to train mode
        model.train()

        # train parameter
        current_lambda_for_adjust = max(args.lambda_for_adjust_start + epoch * args.lambda_for_adjust_slope, args.lambda_for_adjust_min)

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = sample_batched['image']
            depth = sample_batched['depth']
            current_batch_size = image.size(0)

            loss_decoder = []
            l_custom = []
            if training_mode == 'Default':
                image = torch.autograd.Variable(image.cuda())
                image_full = interpolate_bicubic_fullsize(image)
                depth = torch.autograd.Variable(depth.cuda(non_blocking=True))
                depth_pred = model(image)

                # compute iter loss & train_scores
                index_type = 0
                temp_loss_decoder, temp_l_custom, train_scores[index_type] = compute_1loss_with_record(depth_pred, depth, batch_size, current_batch_size, i, num_train_data, train_scores[index_type])
                loss_decoder.append(temp_loss_decoder)
                l_custom.append(temp_l_custom)
                if index_type == 0:
                    loss = loss_decoder[index_type]
                else:
                    if index_type < 6:
                        loss = loss + loss_decoder[index_type]
                    elif index_type < 12:
                        loss = loss + relative_loss_weight * loss_decoder[index_type]

            elif training_mode == 'OrdinaryRelativeDepth':
                image = torch.autograd.Variable(image.cuda())
                image_full = interpolate_bicubic_fullsize(image)
                depth = torch.autograd.Variable(depth.cuda(non_blocking=True))
                #for index_type in range(12):
                #    depth[index_type] = torch.autograd.Variable(depth[index_type].cuda(non_blocking=True))
                depth_pred = model(image)

                # compute iter loss & train_scores
                for index_type in range(12):
                    interpolate_pred_size = nn.Upsample(size=[depth_pred[index_type].shape[2], depth_pred[index_type].shape[3]], mode='bicubic')
                    depth_temp = interpolate_pred_size(depth)
                    temp_loss_decoder, temp_l_custom, train_scores[index_type] = compute_1loss_with_record(depth_pred[index_type], depth_temp, batch_size, current_batch_size, i, num_train_data, train_scores[index_type])
                    loss_decoder.append(temp_loss_decoder)
                    l_custom.append(temp_l_custom)
                    if index_type == 0:
                        loss = loss_decoder[index_type]
                    else:
                        if index_type < 6:
                            loss = loss + loss_decoder[index_type]
                        elif index_type < 12:
                            loss = loss + relative_loss_weight * loss_decoder[index_type]

            # Update step
            if use_amp == True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (iter_per_epoch - i))))

            # Log progress
            niter = epoch * iter_per_epoch + i + 1
            if (i+1) % 100 == 0 or (i+1) == iter_per_epoch:

                # Print to console
                print('  ')
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t    '
                      .format(epoch, i+1, iter_per_epoch, batch_time=batch_time, eta=eta), end='')

                train_scores_mean = []
                print("l_depth: ", end='')
                for index_type in range(len(loss_decoder)):
                    if epoch == 0:
                        train_scores_mean.append(train_scores[index_type][0:(i+1)*batch_size].mean(axis=0))
                    else:
                        train_scores_mean.append(train_scores[index_type].mean(axis=0))
                    print_scores_one(train_scores_mean[index_type])

                if (i+1) == 1 or (i+1) % 1000 == 0 or (i+1) == iter_per_epoch:
                    savePath = model_path + "/current_type" + str(index_type).zfill(2) + ".csv"
                    dataframe = pd.DataFrame(train_scores[index_type])
                    dataframe.to_csv(savePath, header=False, index=False)

            if i in iter_list_save:
                model_name = model_path + "/epoch" + str(epoch+1).zfill(2) + '_iter' + str(i).zfill(5) + ".pth"
                # save model
                print('SAVE MODEL:' + model_path)
                torch.save(model.state_dict(), model_name)
                last_save_iter = (i+1) % iter_per_epoch

            if i in iter_list_arrangement:
                for index_type in range(len(loss_decoder)):
                    temp_train_scores_mean = train_scores[index_type][last_arrangement_iter*batch_size:(i+1)*batch_size, :].mean(axis=0)
                    total_loss = np.sum(temp_train_scores_mean * loss_weights[index_type])
                    if weight_flattening == True and TF_weight_flattening == False:
                        num_loss_valid = sum(loss_valid[index_type])
                        for index_loss in range(len(loss_valid[index_type])):
                            if loss_valid[index_type][index_loss] == 1:
                                loss_weights[index_type][index_loss] = (total_loss * loss_initialize_scale[index_type][index_loss]) / temp_train_scores_mean[index_loss]
                            else:
                                loss_weights[index_type][index_loss] = 0

                        # save previous record
                        TF_weight_flattening = True
                        previous_total_loss[index_type] = np.sum(temp_train_scores_mean * loss_weights[index_type])
                        previous_loss[index_type] = temp_train_scores_mean

                    elif weight_adjustment == True and (TF_weight_flattening == True or weight_flattening == False):
                        temp_train_scores_mean = train_scores[index_type][last_arrangement_iter*batch_size:(i+1)*batch_size, :].mean(axis=0)
                        total_loss = np.sum(temp_train_scores_mean * loss_weights[index_type])
                        previous_loss_weights = np.array(loss_weights[index_type])
                        if previous_total_loss > 0:
                            for index_loss in range(len(loss_valid[index_type])):
                                if loss_valid[index_type][index_loss] == 1:
                                    adjust_term = 1 + current_lambda_for_adjust * ((total_loss/previous_total_loss[index_type]) * (previous_loss[index_type][index_loss]/temp_train_scores_mean[index_loss]) - 1)
                                    adjust_term = min(max(adjust_term, 1.0/2.0), 2.0/1.0)
                                    loss_weights[index_type][index_loss] = previous_loss_weights[index_type][index_loss] * adjust_term
                                else:
                                    loss_weights[index_type][index_loss] = 0

                        # save previous record
                        previous_total_loss[index_type] = np.sum(temp_train_scores_mean * loss_weights[index_type])
                        previous_loss[index_type] = temp_train_scores_mean

                    # save - loss weights
                    savePath = model_path + "/train_loss_weights" + str(epoch + 1).zfill(2) + '_iter' + str(i).zfill(5) + '_type' + str(index_type).zfill(2) + ".csv"
                    dataframe = pd.DataFrame(loss_weights[index_type])
                    dataframe.to_csv(savePath, header=False, index=False)

                last_arrangement_iter = (i+1) % iter_per_epoch

        for index_type in range(len(loss_decoder)):
            # save - each image train score
            savePath = model_path + "/train_epoch" + str(0 + epoch + 1).zfill(2) + '_type' + str(index_type).zfill(2) + ".csv"
            dataframe = pd.DataFrame(train_scores[index_type])
            dataframe.to_csv(savePath, header=False, index=False)
            # save - train mean score
            savePath = model_path + "/train_mean_epoch" + str(0 + epoch + 1).zfill(2) + '_type' + str(index_type).zfill(2) + ".csv"
            dataframe = pd.DataFrame(train_scores_mean[index_type])
            dataframe.to_csv(savePath, header=False, index=False)

        print('-------------- TRAINING OF EPOCH ' + str(0+epoch+1).zfill(2) + 'FINISH ---------------')
        print('---------------------------------------------------------')
        print('   ')
        print('   ')
        print('   ')

if __name__ == '__main__':
    main()
