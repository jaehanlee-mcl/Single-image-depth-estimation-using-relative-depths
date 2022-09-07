from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib
import matplotlib.cm
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

def read_loss_weights(savePath):
    try:
        loss_weights = pd.read_csv(savePath, header=None).values
    except:
        loss_weights = pd.read_csv('./default/loss_weights.csv', header=None).values
    loss_weights = torch.tensor(loss_weights).cuda()
    return loss_weights

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    gt_mean = gt.mean()
    pred_mean = pred.mean()

    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sqr_rel = np.mean(np.square(gt - pred) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    metric3 = (rmse * abs_rel * a1) ** (1/3)
    metric8 = (rmse * rmse_log * abs_rel * sqr_rel * log_10 * a1 * a2 * a3) ** (1/8)

    # scale-invariant metric
    si_gt = np.exp(np.log(gt) - (np.log(gt)).mean())
    si_pred = np.exp(np.log(pred) - (np.log(pred)).mean())

    si_thresh = np.maximum((si_gt / si_pred), (si_pred / si_gt))

    si_a1 = (si_thresh < 1.25).mean()
    si_a2 = (si_thresh < 1.25 ** 2).mean()
    si_a3 = (si_thresh < 1.25 ** 3).mean()

    si_abs_rel = np.mean(np.abs(si_gt - si_pred) / si_gt)
    si_sqr_rel = np.mean(np.square(si_gt - si_pred) / si_gt)

    si_rmse = (si_gt - si_pred) ** 2
    si_rmse = np.sqrt(si_rmse.mean())

    si_rmse_log = (np.log(si_gt) - np.log(si_pred)) ** 2
    si_rmse_log = np.sqrt(si_rmse_log.mean())

    si_log_10 = (np.abs(np.log10(si_gt) - np.log10(si_pred))).mean()

    si_metric3 = (si_rmse * si_abs_rel * (1-si_a1)) ** (1/3)
    si_metric8 = (si_rmse * si_rmse_log * si_abs_rel * si_sqr_rel * si_log_10 * (1-si_a1) * (1-si_a2) * (1-si_a3)) ** (1/8)

    return rmse, rmse_log, abs_rel, sqr_rel, log_10, a1, a2, a3, metric3, metric8, si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log_10, si_a1, si_a2, si_a3, si_metric3, si_metric8

def compute_correlation(gt, pred):

    gt = np.ravel(gt)
    pred = np.ravel(pred)

    pearson_rho, pearson_p = pearsonr(gt, pred)
    spearman_rho, spearman_p = spearmanr(gt, pred, axis=None)
    kendal_tau, kendal_p = kendalltau(gt, pred)

    return pearson_rho, spearman_rho, kendal_tau

def print_scores_one(scores):
    print("{:10.7f}, ".format(scores[25]), end='')

def print_scores(scores):
    ############
    ## line 1 ##
    ############
    if abs(scores[0]) + abs(scores[1]) + abs(scores[2]) + abs(scores[3]) + abs(scores[4]) \
            + abs(scores[5]) + abs(scores[6]) + abs(scores[7]) + abs(scores[8]) + abs(scores[9]) \
            + abs(scores[70]) + abs(scores[71]) + abs(scores[72]) + abs(scores[73]) + abs(scores[74]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format('l_rmse', 'l_rmse_log',
                                                                                                    'l_abs_rel',
                                                                                                    'l_sqr_rel', 'l_log10',
                                                                                                    'l_delta1', 'l_delta2',
                                                                                                    'l_delta3', 'l_metric3',
                                                                                                    'l_metric8', '-----', '-----', '-----', 'l_all_geo', 'l_custom'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[0],
                scores[1],
                scores[2],
                scores[3],
                scores[4],
                scores[5],
                scores[6],
                scores[7],
                scores[8],
                scores[9],
                scores[70],
                scores[71],
                scores[72],
                scores[73],
                scores[74]
            ))

    ############
    ## line 2 ##
    ############
    if abs(scores[10]) + abs(scores[11]) + abs(scores[12]) + abs(scores[13]) + abs(scores[14]) \
            + abs(scores[15]) + abs(scores[16]) + abs(scores[17]) + abs(scores[18]) + abs(scores[19]) \
            + abs(scores[20]) + abs(scores[21]) + abs(scores[22]) + abs(scores[23]) + abs(scores[24]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format('l_si_rmse', 'l_si_rmse_log',
                                                                                                    'l_si_abs_rel',
                                                                                                    'l_si_sqr_rel', 'l_si_log10',
                                                                                                    'l_si_delta1', 'l_si_delta2',
                                                                                                    'l_si_delta3', 'l_si_metric3',
                                                                                                    'l_si_metric8', 'corr_pearson', 'corr_spearman', 'corr_kendal', '-----', '-----'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[10],
                scores[11],
                scores[12],
                scores[13],
                scores[14],
                scores[17],
                scores[16],
                scores[17],
                scores[18],
                scores[19],
                scores[20],
                scores[21],
                scores[22],
                scores[23],
                scores[24]
            ))

    ############
    ## line 3 ##
    ############
    if abs(scores[25]) + abs(scores[26]) + abs(scores[27]) + abs(scores[28]) + abs(scores[29]) \
            + abs(scores[30]) + abs(scores[31]) + abs(scores[32]) + abs(scores[33]) + abs(scores[34]) \
            + abs(scores[35]) + abs(scores[36]) + abs(scores[37]) + abs(scores[38]) + abs(scores[39]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format('l_depth', 'l_depth_dx',
                                                                                                    'l_depth_dy',
                                                                                                    'l_depth_norm', 'l_depth_dx2',
                                                                                                    'l_depth_dxy', 'l_depth_dy2',
                                                                                                    'l_depth_dx_norm', 'l_depth_dy_norm',
                                                                                                    'l_ssim', 'l_ndepth', 'l_ndepth_w5', 'l_ndepth_w17', 'l_ndepth_w65', 'l_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[25],
                scores[26],
                scores[27],
                scores[28],
                scores[29],
                scores[30],
                scores[31],
                scores[32],
                scores[33],
                scores[34],
                scores[35],
                scores[36],
                scores[37],
                scores[38],
                scores[39]))

    ############
    ## line 4 ##
    ############
    if abs(scores[40]) + abs(scores[41]) + abs(scores[42]) + abs(scores[43]) + abs(scores[44]) \
            + abs(scores[45]) + abs(scores[46]) + abs(scores[47]) + abs(scores[48]) + abs(scores[49]) \
            + abs(scores[50]) + abs(scores[51]) + abs(scores[52]) + abs(scores[53]) + abs(scores[54]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format('l_log_depth', 'l_log_depth_dx',
                                                                                                    'l_log_depth_dy',
                                                                                                    'l_log_depth_norm', 'l_log_depth_dx2',
                                                                                                    'l_log_depth_dxy', 'l_log_depth_dy2',
                                                                                                    'l_log_depth_dx_norm', 'l_log_depth_dy_norm',
                                                                                                    'l_log_ssim', 'l_log_ndepth', 'l_log_ndepth_w5', 'l_log_ndepth_w17', 'l_log_ndepth_w65', 'l_log_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[40],
                scores[41],
                scores[42],
                scores[43],
                scores[44],
                scores[45],
                scores[46],
                scores[47],
                scores[48],
                scores[49],
                scores[50],
                scores[51],
                scores[52],
                scores[53],
                scores[54]))

    ############
    ## line 5 ##
    ############
    if abs(scores[55]) + abs(scores[56]) + abs(scores[57]) + abs(scores[58]) + abs(scores[59]) \
            + abs(scores[60]) + abs(scores[61]) + abs(scores[62]) + abs(scores[63]) + abs(scores[64]) \
            + abs(scores[65]) + abs(scores[66]) + abs(scores[67]) + abs(scores[68]) + abs(scores[69]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format('l_inv_depth', 'l_inv_depth_dx',
                                                                                                    'l_inv_depth_dy',
                                                                                                    'l_inv_depth_norm', 'l_inv_depth_dx2',
                                                                                                    'l_inv_depth_dxy', 'l_inv_depth_dy2',
                                                                                                    'l_inv_depth_dx_norm', 'l_inv_depth_dy_norm',
                                                                                                    'l_inv_ssim', 'l_inv_ndepth', 'l_inv_ndepth_w5', 'l_inv_ndepth_w17', 'l_inv_ndepth_w65', 'l_inv_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[55],
                scores[56],
                scores[57],
                scores[58],
                scores[59],
                scores[60],
                scores[61],
                scores[62],
                scores[63],
                scores[64],
                scores[65],
                scores[66],
                scores[67],
                scores[68],
                scores[69]))

    ############
    ## line 6 ##
    ############
    if abs(scores[75]) + abs(scores[76]) + abs(scores[77]) + abs(scores[78]) + abs(scores[79]) \
            + abs(scores[80]) + abs(scores[81]) + abs(scores[82]) + abs(scores[83]) + abs(scores[84]) \
            + abs(scores[85]) + abs(scores[86]) + abs(scores[87]) + abs(scores[88]) + abs(scores[89]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
                'l_d1_depth', 'l_d1_depth_dx',
                'l_d1_depth_dy',
                'l_d1_depth_norm', 'l_d1_depth_dx2',
                'l_d1_depth_dxy', 'l_d1_depth_dy2',
                'l_d1_depth_dx_norm', 'l_d1_depth_dy_norm',
                'l_d1_ssim', 'l_d1_ndepth', 'l_d1_ndepth_w5', 'l_d1_ndepth_w17', 'l_d1_ndepth_w65', 'l_d1_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[75],
                scores[76],
                scores[77],
                scores[78],
                scores[79],
                scores[80],
                scores[81],
                scores[82],
                scores[83],
                scores[84],
                scores[85],
                scores[86],
                scores[87],
                scores[88],
                scores[89]))

    ############
    ## line 7 ##
    ############
    if abs(scores[90]) + abs(scores[91]) + abs(scores[92]) + abs(scores[93]) + abs(scores[94]) \
            + abs(scores[95]) + abs(scores[96]) + abs(scores[97]) + abs(scores[98]) + abs(scores[99]) \
            + abs(scores[100]) + abs(scores[101]) + abs(scores[102]) + abs(scores[103]) + abs(scores[104]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
                'l_d2_depth', 'l_d2_depth_dx',
                'l_d2_depth_dy',
                'l_d2_depth_norm', 'l_d2_depth_dx2',
                'l_d2_depth_dxy', 'l_d2_depth_dy2',
                'l_d2_depth_dx_norm', 'l_d2_depth_dy_norm',
                'l_d2_ssim', 'l_d2_ndepth', 'l_d2_ndepth_w5', 'l_d2_ndepth_w17', 'l_d2_ndepth_w65', 'l_d2_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[90],
                scores[91],
                scores[92],
                scores[93],
                scores[94],
                scores[95],
                scores[96],
                scores[97],
                scores[98],
                scores[99],
                scores[100],
                scores[101],
                scores[102],
                scores[103],
                scores[104]))

    ############
    ## line 8 ##
    ############
    if abs(scores[105]) + abs(scores[106]) + abs(scores[107]) + abs(scores[108]) + abs(scores[109]) \
            + abs(scores[110]) + abs(scores[111]) + abs(scores[112]) + abs(scores[113]) + abs(scores[114]) \
            + abs(scores[115]) + abs(scores[116]) + abs(scores[117]) + abs(scores[118]) + abs(scores[119]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
                'l_d3_depth', 'l_d3_depth_dx',
                'l_d3_depth_dy',
                'l_d3_depth_norm', 'l_d3_depth_dx2',
                'l_d3_depth_dxy', 'l_d3_depth_dy2',
                'l_d3_depth_dx_norm', 'l_d3_depth_dy_norm',
                'l_d3_ssim', 'l_d3_ndepth', 'l_d3_ndepth_w5', 'l_d3_ndepth_w17', 'l_d3_ndepth_w65', 'l_d3_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[105],
                scores[106],
                scores[107],
                scores[108],
                scores[109],
                scores[110],
                scores[111],
                scores[112],
                scores[113],
                scores[114],
                scores[115],
                scores[116],
                scores[117],
                scores[118],
                scores[119]))

    ############
    ## line 9 ##
    ############
    if abs(scores[120]) + abs(scores[121]) + abs(scores[122]) + abs(scores[123]) + abs(scores[124]) \
            + abs(scores[125]) + abs(scores[126]) + abs(scores[127]) + abs(scores[128]) + abs(scores[129]) \
            + abs(scores[130]) + abs(scores[131]) + abs(scores[132]) + abs(scores[133]) + abs(scores[134]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
                'l_d4_depth', 'l_d4_depth_dx',
                'l_d4_depth_dy',
                'l_d4_depth_norm', 'l_d4_depth_dx2',
                'l_d4_depth_dxy', 'l_d4_depth_dy2',
                'l_d4_depth_dx_norm', 'l_d4_depth_dy_norm',
                'l_d4_ssim', 'l_d4_ndepth', 'l_d4_ndepth_w5', 'l_d4_ndepth_w17', 'l_d4_ndepth_w65', 'l_d4_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[120],
                scores[121],
                scores[122],
                scores[123],
                scores[124],
                scores[125],
                scores[126],
                scores[127],
                scores[128],
                scores[129],
                scores[130],
                scores[131],
                scores[132],
                scores[133],
                scores[134]))

    ############
    ## line 10##
    ############
    if abs(scores[135]) + abs(scores[136]) + abs(scores[137]) + abs(scores[138]) + abs(scores[139]) \
            + abs(scores[140]) + abs(scores[141]) + abs(scores[142]) + abs(scores[143]) + abs(scores[144]) \
            + abs(scores[145]) + abs(scores[146]) + abs(scores[147]) + abs(scores[148]) + abs(scores[149]) > 0:
        print(
            "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
                'l_d5_depth', 'l_d5_depth_dx',
                'l_d5_depth_dy',
                'l_d5_depth_norm', 'l_d5_depth_dx2',
                'l_d5_depth_dxy', 'l_d5_depth_dy2',
                'l_d5_depth_dx_norm', 'l_d5_depth_dy_norm',
                'l_d5_ssim', 'l_d5_ndepth', 'l_d5_ndepth_w5', 'l_d5_ndepth_w17', 'l_d5_ndepth_w65', 'l_d5_geo'))
        print(
            "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
                scores[135],
                scores[136],
                scores[137],
                scores[138],
                scores[139],
                scores[130],
                scores[141],
                scores[142],
                scores[143],
                scores[144],
                scores[145],
                scores[146],
                scores[147],
                scores[148],
                scores[149]))



def print_metrics(metrics):
    ############
    ## line 1 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'rmse', 'rmse_log',
            'abs_rel',
            'sqr_rel', 'log10',
            'delta1', 'delta2',
            'delta3', 'metric3',
            'metric8'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            metrics[0],
            metrics[1],
            metrics[2],
            metrics[3],
            metrics[4],
            metrics[5],
            metrics[6],
            metrics[7],
            metrics[8],
            metrics[9]
        ))

    ############
    ## line 2 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'si_rmse', 'si_rmse_log',
            'si_abs_rel',
            'si_sqr_rel', 'si_log10',
            'si_delta1', 'si_delta2',
            'si_delta3', 'si_metric3',
            'si_metric8', 'corr_pearson', 'corr_spearman', 'corr_kendal', '-----', '-----'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            metrics[10],
            metrics[11],
            metrics[12],
            metrics[13],
            metrics[14],
            metrics[17],
            metrics[16],
            metrics[17],
            metrics[18],
            metrics[19],
            metrics[20],
            metrics[21],
            metrics[22],
            metrics[23],
            metrics[24]
        ))

def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details