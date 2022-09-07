import torch
from math import exp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tensorboardX import SummaryWriter
from loss import ssim, loss_for_metric8, loss_for_derivative, loss_for_normalized_depth, loss_for_depth
from utils import AverageMeter, DepthNorm, colorize, compute_errors, compute_correlation, print_scores

def get_loss_initialize_scale(loss_initialize_type = '78loss-same'):
    if loss_initialize_type == '78loss-same':
        loss_initialize_scale = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        return loss_initialize_scale
    if loss_initialize_type == '78loss-cluster':
        loss_initialize_scale = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.01197025,         0.02401964,         0.01165018,         0.01210747,         0.03130118, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.03686666,         0.02684559,         0.01735569,         0.01735569,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.01150434,         0.02684559,         0.01040926,         0.01259069,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.01197025,         0.01791125,         0.00945813,         0.01143913,         0.01646388, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.01380721,         0.00945813,         0.00945978,         0.01107364,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.01150434,         0.01165018,         0.00926585,         0.01600933,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.01197025,         0.01165018,         0.01091854,         0.00958399,         0.01015522, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.01110596,         0.01056334,         0.01051631,         0.00957361,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.01150434,         0.00926585,         0.00916838,         0.01900492,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.01197025,         0.00916838,         0.01056334,         0.00937624,         0.01087883, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00982492,         0.00973674,         0.00991826,         0.00945978,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.01150434,         0.01096031,         0.00916838,         0.01667717,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.01197025,         0.01096317,         0.01166525,         0.00937624,         0.01112222, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00973674,         0.00973674,         0.00991826,         0.01029893,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.01150434,         0.01087883,         0.01350487,         0.01150434,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.01197025,         0.01141122,         0.01248226,         0.01011307,         0.01379265, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.01071140,         0.01071140,         0.01029893,         0.01007689,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.01461877,         0.01505326,         0.01461877,         0.01748228,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        return loss_initialize_scale
    if loss_initialize_type == '13loss-same':
        loss_initialize_scale = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.07692307,         0.07692307,         0.07692307,         0.07692307,         0.07692307, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.07692307,         0.07692307,         0.07692307,         0.07692307,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.07692307,         0.07692307,         0.07692307,         0.07692307,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        return loss_initialize_scale
    if loss_initialize_type == '13loss-cluster':
        loss_initialize_scale = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.04772404,         0.09576358,         0.04644794,         0.04827112,         0.12479426, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.14698321,         0.10703035,         0.06919517,         0.06919517,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.04586651,         0.10703035,         0.04150056,         0.05019765,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        return loss_initialize_scale
    if loss_initialize_type == '5loss-same':
        loss_initialize_scale = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.20000000,         0.20000000,         0.20000000,         0.20000000,         0.00000000, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.20000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        return loss_initialize_scale
    if loss_initialize_type == '5loss-cluster':
        loss_initialize_scale = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.16799908,         0.33710881,         0.16350695,         0.16992495,         0.00000000, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.16146019,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        return loss_initialize_scale

    if loss_initialize_type == '1loss':
        loss_initialize_scale = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            1.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        return loss_initialize_scale

def get_loss_weights(loss_type = '78loss'):
    if loss_type == '78loss' or loss_type == 'relative':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            1.00000000,         0.00000001,         0.00000001,         0.00000001,         0.00000001, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000001, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000001, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000001, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000001, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000001, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000
    if loss_type == '13loss':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            1.00000000,         0.00000001,         0.00000001,         0.00000001,         0.00000001, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.00000001,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000
    if loss_type == '5loss':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            1.00000000,         0.00000001,         0.00000001,         0.00000001,         0.00000000, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.00000001,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000
    if loss_type == '78loss-same':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000
    if loss_type == '13loss-same':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.07692308,         0.07692308,         0.07692308,         0.07692308,         0.07692308, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.07692308,         0.07692308,         0.07692308,         0.07692308,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.07692308,         0.07692308,         0.07692308,         0.07692308,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000
    if loss_type == '5loss-same':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.20000000,         0.20000000,         0.20000000,         0.20000000,         0.00000000, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.20000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000
    if loss_type == '18loss-same':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            0.05555556,         0.05555556,         0.05555556,         0.00000000,         0.00000000, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.05555556,         0.05555556,         0.05555556,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.05555556,         0.05555556,         0.05555556,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.05555556,         0.05555556,         0.05555556,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.05555556,         0.05555556,         0.05555556,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.05555556,         0.05555556,         0.05555556,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000

    if loss_type == '1loss':
        loss_weights = [
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_rmse,            l_rmse_log,         l_abs_rel,              l_sqr_rel,              l_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_delta1,          l_delta2,           l_delta3,               l_metric3,              l_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_rmse,         l_si_rmse_log,      l_si_abs_rel,           l_si_sqr_rel,           l_si_log10,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_si_delta1,       l_si_delta2,        l_si_delta3,            l_si_metric3,           l_si_metric8,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # corr_pearson,      corr_spearman,      corr_kendal,            0,                      0,
            1.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth,           l_depth_dx,         l_depth_dy,             l_depth_norm,           l_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_depth_dxy,       l_depth_dy2,        l_depth_dx_norm,        l_depth_dy_norm,        l_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_ndepth,          l_ndepth_win5,      l_ndepth_win17,         l_ndepth_win65,         l_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth,       l_log_depth_dx,     l_log_depth_dy,         l_log_depth_norm,       l_log_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_depth_dxy,   l_log_depth_dy2,    l_log_depth_dx_norm,    l_log_depth_dy_norm,    l_log_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_log_ndepth,      l_log_ndepth_win5,  l_log_ndepth_win17,     l_log_ndepth_win65,     l_log_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth,       l_inv_depth_dx,     l_inv_depth_dy,         l_inv_depth_norm,       l_inv_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_depth_dxy,   l_inv_depth_dy2,    l_inv_depth_dx_norm,    l_inv_depth_dy_norm,    l_inv_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_inv_ndepth,      l_inv_ndepth_win5,  l_inv_ndepth_win17,     l_inv_ndepth_win65,     l_inv_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # 0,                 0,                  0,                      l_all_geo,              l_custom
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d1_ndepth,       l_d1_ndepth_win5,   l_d1_ndepth_win17,      l_d1_ndepth_win65,      l_d1_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d2_ndepth,       l_d2_ndepth_win5,   l_d2_ndepth_win17,      l_d2_ndepth_win65,      l_d2_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d3_ndepth,       l_d3_ndepth_win5,   l_d3_ndepth_win17,      l_d3_ndepth_win65,      l_d3_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d4_ndepth,       l_d4_ndepth_win5,   l_d4_ndepth_win17,      l_d4_ndepth_win65,      l_d4_geo,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ssim,
            0.00000000,         0.00000000,         0.00000000,         0.00000000,         0.00000000, # l_d5_ndepth,       l_d5_ndepth_win5,   l_d5_ndepth_win17,      l_d5_ndepth_win65,      l_d5_geo,
        ]
        for index in range(0, len(loss_weights)):
            loss_weights[index] = loss_weights[index] * 1.0000
    if loss_type == 'relative':
        loss_weights = loss_weights + loss_weights + loss_weights + loss_weights + loss_weights + loss_weights
        return loss_weights
    else:
        return loss_weights

def get_loss_valid(valid = True):
    if valid == True:
        return np.ones(150)
    else:
        return np.zeros(150)

def get_metric_valid(valid = True):
    if valid == True:
        return np.ones(25)
    else:
        return np.zeros(25)

def compute_multi_metric_with_record(depth_pred_for_metric, depth_gt_for_metric, metric_valid, batch_size, current_batch_size, i, num_test_data, test_metrics):
    rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8, \
    si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8, \
    corr_pearson, corr_spearman, corr_kendal = compute_multi_metric(depth_pred_for_metric, depth_gt_for_metric, metric_valid)

    test_metrics = get_metric_1batch(batch_size, current_batch_size, i, num_test_data, test_metrics,
                                     rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8,
                                     si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8,
                                     corr_pearson, corr_spearman, corr_kendal)
    return test_metrics

def compute_multi_metric(depth_pred_for_metric, depth_gt_for_metric, loss_valid):
    current_batch_size = depth_gt_for_metric.size(0)
    invalid_input = torch.zeros(current_batch_size).cuda(torch.device("cuda:0"))
    ## METRIC LIST
    if \
            loss_valid[0] != 0 or loss_valid[1] != 0 or loss_valid[2] != 0 or loss_valid[3] != 0 or \
                    loss_valid[4] != 0 or \
                    loss_valid[5] != 0 or loss_valid[6] != 0 or loss_valid[7] != 0 or loss_valid[
                8] != 0 or loss_valid[9] != 0:
        rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8 \
            = loss_for_metric8(depth_pred_for_metric, depth_gt_for_metric)
        delta1 = 1 - delta1
        delta2 = 1 - delta2
        delta3 = 1 - delta3
    else:
        rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8 \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if \
            loss_valid[10] != 0 or loss_valid[11] != 0 or loss_valid[12] != 0 or loss_valid[13] != 0 or \
                    loss_valid[14] != 0 or \
                    loss_valid[15] != 0 or loss_valid[16] != 0 or loss_valid[17] != 0 or loss_valid[
                18] != 0 or loss_valid[19] != 0:
        si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8 \
            = loss_for_metric8(depth_pred_for_metric / depth_pred_for_metric.mean(),
                               depth_gt_for_metric / depth_gt_for_metric.mean())
        si_delta1 = 1 - si_delta1
        si_delta2 = 1 - si_delta2
        si_delta3 = 1 - si_delta3
    else:
        si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8 \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if \
            loss_valid[20] != 0 or loss_valid[21] != 0 or loss_valid[22] != 0:
        corr_pearson = np.zeros(current_batch_size)
        corr_spearman = np.zeros(current_batch_size)
        corr_kendal = np.zeros(current_batch_size)
        for index_batch in range(current_batch_size):
            depth_pred_for_metric_one = depth_pred_for_metric[index_batch:index_batch + 1, :, :,
                                        :].cpu().detach().numpy()
            depth_gt_for_metric_one = depth_gt_for_metric[index_batch:index_batch + 1, :, :, :].cpu().detach().numpy()

            depth_pred_for_metric_one = depth_pred_for_metric_one[np.nonzero(depth_pred_for_metric_one)]
            depth_gt_for_metric_one = depth_gt_for_metric_one[np.nonzero(depth_gt_for_metric_one)]

            corr_pearson[index_batch], corr_spearman[index_batch], corr_kendal[index_batch] = compute_correlation(
                depth_pred_for_metric_one, depth_gt_for_metric_one)
    else:
        corr_pearson, corr_spearman, corr_kendal \
            = invalid_input, invalid_input, invalid_input

    return rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8, \
           si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8, \
           corr_pearson, corr_spearman, corr_kendal

def compute_multi_loss_with_record(depth_pred_for_loss, depth_gt_for_loss, loss_weights, loss_valid, batch_size, current_batch_size, i, num_train_data, train_scores):
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
		= compute_multi_loss(depth_pred_for_loss, depth_gt_for_loss, loss_weights, loss_valid)

    # compute iter loss & train_scores
    loss, l_custom, train_scores = get_loss_1batch(batch_size, current_batch_size, i, num_train_data, loss_weights, train_scores,
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

    return loss, l_custom, train_scores

def compute_multi_loss(depth_pred_for_loss, depth_gt_for_loss, loss_weights, loss_valid):
    _, channel, height, width = depth_pred_for_loss.size()

    # Loss
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    relu = nn.ReLU()

    # interpolation function
    interpolate_bicubic_div02 = nn.Upsample(scale_factor=1 / 2, mode='bicubic')
    interpolate_bicubic_div04 = nn.Upsample(scale_factor=1 / 4, mode='bicubic')
    interpolate_bicubic_div08 = nn.Upsample(scale_factor=1 / 8, mode='bicubic')
    interpolate_bicubic_div16 = nn.Upsample(scale_factor=1 / 16, mode='bicubic')
    interpolate_bicubic_div32 = nn.Upsample(scale_factor=1 / 32, mode='bicubic')
    interpolate_bicubic_1by1 = nn.Upsample(size=[1, 1], mode='bicubic')

    # resize map
    # size 144x192
    if min(height, width) >= pow(2,1):
        depth_pred_for_loss_down1 = interpolate_bicubic_div02(depth_pred_for_loss)
        depth_gt_for_loss_down1 = interpolate_bicubic_div02(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down1 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down1 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 72x96
    if min(height, width) >= pow(2, 2):
        depth_pred_for_loss_down2 = interpolate_bicubic_div04(depth_pred_for_loss)
        depth_gt_for_loss_down2 = interpolate_bicubic_div04(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down2 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down2 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 36x48
    if min(height, width) >= pow(2, 3):
        depth_pred_for_loss_down3 = interpolate_bicubic_div08(depth_pred_for_loss)
        depth_gt_for_loss_down3 = interpolate_bicubic_div08(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down3 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down3 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 18x24
    if min(height, width) >= pow(2, 4):
        depth_pred_for_loss_down4 = interpolate_bicubic_div16(depth_pred_for_loss)
        depth_gt_for_loss_down4 = interpolate_bicubic_div16(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down4 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down4 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 9x12
    if min(height, width) >= pow(2, 5):
        depth_pred_for_loss_down5 = interpolate_bicubic_div32(depth_pred_for_loss)
        depth_gt_for_loss_down5 = interpolate_bicubic_div32(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down5 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down5 = interpolate_bicubic_1by1(depth_gt_for_loss)

    current_batch_size = depth_gt_for_loss.size(0)
    invalid_input = torch.zeros(current_batch_size).cuda(torch.device("cuda:0"))

    ## LOSS LIST
    if \
            loss_valid[0] != 0 or loss_valid[1] != 0 or loss_valid[2] != 0 or loss_valid[3] != 0 or \
                    loss_valid[4] != 0 or \
                    loss_valid[5] != 0 or loss_valid[6] != 0 or loss_valid[7] != 0 or loss_valid[
                8] != 0 or loss_valid[9] != 0:
        l_rmse, l_rmse_log, l_abs_rel, l_sqr_rel, l_log10, l_delta1, l_delta2, l_delta3, l_metric3, l_metric8 \
            = loss_for_metric8(depth_pred_for_loss, depth_gt_for_loss)
    else:
        l_rmse, l_rmse_log, l_abs_rel, l_sqr_rel, l_log10, l_delta1, l_delta2, l_delta3, l_metric3, l_metric8 \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if \
            loss_valid[10] != 0 or loss_valid[11] != 0 or loss_valid[12] != 0 or loss_valid[13] != 0 or \
                    loss_valid[14] != 0 or \
                    loss_valid[15] != 0 or loss_valid[16] != 0 or loss_valid[17] != 0 or loss_valid[
                18] != 0 or loss_valid[19] != 0:
        l_si_rmse, l_si_rmse_log, l_si_abs_rel, l_si_sqr_rel, l_si_log10, l_si_delta1, l_si_delta2, l_si_delta3, l_si_metric3, l_si_metric8 \
            = loss_for_metric8(depth_pred_for_loss / depth_pred_for_loss.mean(),
                               depth_gt_for_loss / depth_gt_for_loss.mean())
    else:
        l_si_rmse, l_si_rmse_log, l_si_abs_rel, l_si_sqr_rel, l_si_log10, l_si_delta1, l_si_delta2, l_si_delta3, l_si_metric3, l_si_metric8 \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if \
            loss_valid[25] != 0 or loss_valid[26] != 0 or loss_valid[27] != 0 or loss_valid[28] != 0 or \
                    loss_valid[29] != 0 or \
                    loss_valid[30] != 0 or loss_valid[31] != 0 or loss_valid[32] != 0 or loss_valid[
                33] != 0:
        l_depth, l_depth_dx, l_depth_dy, l_depth_norm, l_depth_dx2, l_depth_dxy, l_depth_dy2, l_depth_dx_norm, l_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss, depth_gt_for_loss)
    else:
        l_depth, l_depth_dx, l_depth_dy, l_depth_norm, l_depth_dx2, l_depth_dxy, l_depth_dy2, l_depth_dx_norm, l_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[34] != 0:
        l_ssim = torch.clamp(
            (1 - ssim(depth_pred_for_loss, depth_gt_for_loss, val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_ssim = invalid_input

    if loss_valid[35] != 0 or loss_valid[36] != 0 or loss_valid[37] != 0 or loss_valid[38] != 0:
        l_ndepth = loss_for_normalized_depth(depth_pred_for_loss, depth_gt_for_loss, window_size=0)
        l_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss, depth_gt_for_loss, window_size=2)
        l_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss, depth_gt_for_loss,
                                                   window_size=8)
        l_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss, depth_gt_for_loss,
                                                   window_size=32)
    else:
        l_ndepth, l_ndepth_win5, l_ndepth_win17, l_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[39] != 0:
        l_geo = (
                        l_depth * l_depth_dx * l_depth_dy * l_depth_norm * l_depth_dx2 * l_depth_dxy * l_depth_dy2 * l_depth_dx_norm * l_depth_dy_norm * l_ssim * l_ndepth * l_ndepth_win5 * l_ndepth_win17 * l_ndepth_win65) ** (
                        1 / 14)
    else:
        l_geo = (
                        l_depth * l_depth_dx * l_depth_dy * l_depth_norm * l_depth_dx2 * l_depth_dxy * l_depth_dy2 * l_depth_dx_norm * l_depth_dy_norm * l_ssim * l_ndepth * l_ndepth_win5 * l_ndepth_win17 * l_ndepth_win65) ** (
                        1 / 14)

    if \
            loss_valid[40] != 0 or loss_valid[41] != 0 or loss_valid[42] != 0 or loss_valid[43] != 0 or \
                    loss_valid[44] != 0 or \
                    loss_valid[45] != 0 or loss_valid[46] != 0 or loss_valid[47] != 0 or loss_valid[
                48] != 0:
        l_log_depth, l_log_depth_dx, l_log_depth_dy, l_log_depth_norm, l_log_depth_dx2, l_log_depth_dxy, l_log_depth_dy2, l_log_depth_dx_norm, l_log_depth_dy_norm \
            = loss_for_derivative(torch.log(depth_pred_for_loss), torch.log(depth_gt_for_loss))
    else:
        l_log_depth, l_log_depth_dx, l_log_depth_dy, l_log_depth_norm, l_log_depth_dx2, l_log_depth_dxy, l_log_depth_dy2, l_log_depth_dx_norm, l_log_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[49] != 0:
        l_log_ssim = torch.clamp(
            (1 - ssim(torch.log(depth_pred_for_loss), torch.log(depth_gt_for_loss),
                      val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_log_ssim = invalid_input

    if loss_valid[50] != 0 or loss_valid[51] != 0 or loss_valid[52] != 0 or loss_valid[53] != 0:
        l_log_ndepth = loss_for_normalized_depth(torch.log(depth_pred_for_loss),
                                                 torch.log(depth_gt_for_loss), window_size=0)
        l_log_ndepth_win5 = loss_for_normalized_depth(torch.log(depth_pred_for_loss),
                                                      torch.log(depth_gt_for_loss), window_size=2)
        l_log_ndepth_win17 = loss_for_normalized_depth(torch.log(depth_pred_for_loss),
                                                       torch.log(depth_gt_for_loss), window_size=8)
        l_log_ndepth_win65 = loss_for_normalized_depth(torch.log(depth_pred_for_loss),
                                                       torch.log(depth_gt_for_loss), window_size=32)
    else:
        l_log_ndepth, l_log_ndepth_win5, l_log_ndepth_win17, l_log_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[54] != 0:
        l_log_geo = (
                            l_log_depth * l_log_depth_dx * l_log_depth_dy * l_log_depth_norm * l_log_depth_dx2 * l_log_depth_dxy * l_log_depth_dy2 * l_log_depth_dx_norm * l_log_depth_dy_norm * l_log_ssim * l_log_ndepth * l_log_ndepth_win5 * l_log_ndepth_win17 * l_log_ndepth_win65) ** (
                            1 / 14)
    else:
        l_log_geo = (
                            l_log_depth * l_log_depth_dx * l_log_depth_dy * l_log_depth_norm * l_log_depth_dx2 * l_log_depth_dxy * l_log_depth_dy2 * l_log_depth_dx_norm * l_log_depth_dy_norm * l_log_ssim * l_log_ndepth * l_log_ndepth_win5 * l_log_ndepth_win17 * l_log_ndepth_win65) ** (
                            1 / 14)

    if \
            loss_valid[55] != 0 or loss_valid[56] != 0 or loss_valid[57] != 0 or loss_valid[58] != 0 or \
                    loss_valid[59] != 0 or \
                    loss_valid[60] != 0 or loss_valid[61] != 0 or loss_valid[62] != 0 or loss_valid[
                63] != 0:
        l_inv_depth, l_inv_depth_dx, l_inv_depth_dy, l_inv_depth_norm, l_inv_depth_dx2, l_inv_depth_dxy, l_inv_depth_dy2, l_inv_depth_dx_norm, l_inv_depth_dy_norm \
            = loss_for_derivative(torch.pow(depth_pred_for_loss, -1),
                                  torch.pow(depth_gt_for_loss, -1))
    else:
        l_inv_depth, l_inv_depth_dx, l_inv_depth_dy, l_inv_depth_norm, l_inv_depth_dx2, l_inv_depth_dxy, l_inv_depth_dy2, l_inv_depth_dx_norm, l_inv_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[64] != 0:
        l_inv_ssim = torch.clamp(
            (1 - ssim(torch.pow(depth_pred_for_loss, -1), torch.pow(depth_gt_for_loss, -1),
                      val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_inv_ssim = invalid_input

    if loss_valid[65] != 0 or loss_valid[66] != 0 or loss_valid[67] != 0 or loss_valid[68] != 0:
        l_inv_ndepth = loss_for_normalized_depth(torch.pow(depth_pred_for_loss, -1),
                                                 torch.pow(depth_gt_for_loss, -1), window_size=0)
        l_inv_ndepth_win5 = loss_for_normalized_depth(torch.pow(depth_pred_for_loss, -1),
                                                      torch.pow(depth_gt_for_loss, -1), window_size=2)
        l_inv_ndepth_win17 = loss_for_normalized_depth(torch.pow(depth_pred_for_loss, -1),
                                                       torch.pow(depth_gt_for_loss, -1), window_size=8)
        l_inv_ndepth_win65 = loss_for_normalized_depth(torch.pow(depth_pred_for_loss, -1),
                                                       torch.pow(depth_gt_for_loss, -1), window_size=32)
    else:
        l_inv_ndepth, l_inv_ndepth_win5, l_inv_ndepth_win17, l_inv_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[69] != 0:
        l_inv_geo = (
                            l_inv_depth * l_inv_depth_dx * l_inv_depth_dy * l_inv_depth_norm * l_inv_depth_dx2 * l_inv_depth_dxy * l_inv_depth_dy2 * l_inv_depth_dx_norm * l_inv_depth_dy_norm * l_inv_ssim * l_inv_ndepth * l_inv_ndepth_win5 * l_inv_ndepth_win17 * l_inv_ndepth_win65) ** (
                            1 / 14)
    else:
        l_inv_geo = (
                            l_inv_depth * l_inv_depth_dx * l_inv_depth_dy * l_inv_depth_norm * l_inv_depth_dx2 * l_inv_depth_dxy * l_inv_depth_dy2 * l_inv_depth_dx_norm * l_inv_depth_dy_norm * l_inv_ssim * l_inv_ndepth * l_inv_ndepth_win5 * l_inv_ndepth_win17 * l_inv_ndepth_win65) ** (
                            1 / 14)

    if loss_valid[73] != 0:
        l_all_geo = (l_geo * l_log_geo * l_inv_geo) ** (1 / 3)
    else:
        l_all_geo = (l_geo * l_log_geo * l_inv_geo) ** (1 / 3)

    if \
            loss_valid[75] != 0 or loss_valid[76] != 0 or loss_valid[77] != 0 or loss_valid[78] != 0 or \
                    loss_valid[79] != 0 or \
                    loss_valid[80] != 0 or loss_valid[81] != 0 or loss_valid[82] != 0 or loss_valid[
                83] != 0:
        l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down1, depth_gt_for_loss_down1)
    else:
        l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[84] != 0:
        l_down1_ssim = torch.clamp(
            (1 - ssim(depth_pred_for_loss_down1, depth_gt_for_loss_down1,
                      val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_down1_ssim = invalid_input

    if loss_valid[85] != 0 or loss_valid[86] != 0 or loss_valid[87] != 0 or loss_valid[88] != 0:
        l_down1_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down1, depth_gt_for_loss_down1,
                                                   window_size=0)
        l_down1_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down1,
                                                        depth_gt_for_loss_down1, window_size=2)
        l_down1_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down1,
                                                        depth_gt_for_loss_down1, window_size=8)
        l_down1_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down1,
                                                         depth_gt_for_loss_down1, window_size=32)
    else:
        l_down1_ndepth, l_down1_ndepth_win5, l_down1_ndepth_win17, l_down1_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[89] != 0:
        l_down1_geo = (
                              l_down1_depth * l_down1_depth_dx * l_down1_depth_dy * l_down1_depth_norm * l_down1_depth_dx2 * l_down1_depth_dxy * l_down1_depth_dy2 * l_down1_depth_dx_norm * l_down1_depth_dy_norm
                              * l_down1_ssim * l_down1_ndepth * l_down1_ndepth_win5 * l_down1_ndepth_win17 * l_down1_ndepth_win65) ** (
                              1 / 14)
    else:
        l_down1_geo = (
                              l_down1_depth * l_down1_depth_dx * l_down1_depth_dy * l_down1_depth_norm * l_down1_depth_dx2 * l_down1_depth_dxy * l_down1_depth_dy2 * l_down1_depth_dx_norm * l_down1_depth_dy_norm
                              * l_down1_ssim * l_down1_ndepth * l_down1_ndepth_win5 * l_down1_ndepth_win17 * l_down1_ndepth_win65) ** (
                              1 / 14)

    if \
            loss_valid[90] != 0 or loss_valid[91] != 0 or loss_valid[92] != 0 or loss_valid[93] != 0 or \
                    loss_valid[94] != 0 or \
                    loss_valid[95] != 0 or loss_valid[96] != 0 or loss_valid[97] != 0 or loss_valid[
                98] != 0:
        l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down2, depth_gt_for_loss_down2)
    else:
        l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[99] != 0:
        l_down2_ssim = torch.clamp(
            (1 - ssim(depth_pred_for_loss_down2, depth_gt_for_loss_down2,
                      val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_down2_ssim = invalid_input

    if loss_valid[100] != 0 or loss_valid[101] != 0 or loss_valid[102] != 0 or loss_valid[103] != 0:
        l_down2_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down2, depth_gt_for_loss_down2,
                                                   window_size=0)
        l_down2_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down2,
                                                        depth_gt_for_loss_down2, window_size=2)
        l_down2_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down2,
                                                         depth_gt_for_loss_down2, window_size=8)
        l_down2_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down2,
                                                         depth_gt_for_loss_down2, window_size=32)
    else:
        l_down2_ndepth, l_down2_ndepth_win5, l_down2_ndepth_win17, l_down2_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[104] != 0:
        l_down2_geo = (
                              l_down2_depth * l_down2_depth_dx * l_down2_depth_dy * l_down2_depth_norm * l_down2_depth_dx2 * l_down2_depth_dxy * l_down2_depth_dy2 * l_down2_depth_dx_norm * l_down2_depth_dy_norm
                              * l_down2_ssim * l_down2_ndepth * l_down2_ndepth_win5 * l_down2_ndepth_win17 * l_down2_ndepth_win65) ** (
                              1 / 13)
    else:
        l_down2_geo = (
                              l_down2_depth * l_down2_depth_dx * l_down2_depth_dy * l_down2_depth_norm * l_down2_depth_dx2 * l_down2_depth_dxy * l_down2_depth_dy2 * l_down2_depth_dx_norm * l_down2_depth_dy_norm
                              * l_down2_ssim * l_down2_ndepth * l_down2_ndepth_win5 * l_down2_ndepth_win17 * l_down2_ndepth_win65) ** (
                              1 / 13)

    if \
            loss_valid[105] != 0 or loss_valid[106] != 0 or loss_valid[107] != 0 or loss_valid[
                108] != 0 or loss_valid[109] != 0 or \
                    loss_valid[110] != 0 or loss_valid[111] != 0 or loss_valid[112] != 0 or loss_valid[
                113] != 0:
        l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down3, depth_gt_for_loss_down3)
    else:
        l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[114] != 0:
        l_down3_ssim = torch.clamp(
            (1 - ssim(depth_pred_for_loss_down3, depth_gt_for_loss_down3,
                      val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_down3_ssim = invalid_input

    if loss_valid[115] != 0 or loss_valid[116] != 0 or loss_valid[117] != 0 or loss_valid[118] != 0:
        l_down3_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down3, depth_gt_for_loss_down3,
                                                   window_size=0)
        l_down3_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down3,
                                                        depth_gt_for_loss_down3, window_size=2)
        l_down3_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down3,
                                                        depth_gt_for_loss_down3, window_size=8)
        l_down3_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down3,
                                                        depth_gt_for_loss_down3, window_size=32)
    else:
        l_down3_ndepth, l_down3_ndepth_win5, l_down3_ndepth_win17, l_down3_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[119] != 0:
        l_down3_geo = (
                              l_down3_depth * l_down3_depth_dx * l_down3_depth_dy * l_down3_depth_norm * l_down3_depth_dx2 * l_down3_depth_dxy * l_down3_depth_dy2 * l_down3_depth_dx_norm * l_down3_depth_dy_norm
                              * l_down3_ssim * l_down3_ndepth * l_down3_ndepth_win5 * l_down3_ndepth_win17 * l_down3_ndepth_win65) ** (
                              1 / 13)
    else:
        l_down3_geo = (
                              l_down3_depth * l_down3_depth_dx * l_down3_depth_dy * l_down3_depth_norm * l_down3_depth_dx2 * l_down3_depth_dxy * l_down3_depth_dy2 * l_down3_depth_dx_norm * l_down3_depth_dy_norm
                              * l_down3_ssim * l_down3_ndepth * l_down3_ndepth_win5 * l_down3_ndepth_win17 * l_down3_ndepth_win65) ** (
                              1 / 13)

    if \
            loss_valid[120] != 0 or loss_valid[121] != 0 or loss_valid[122] != 0 or loss_valid[
                123] != 0 or loss_valid[124] != 0 or \
                    loss_valid[125] != 0 or loss_valid[126] != 0 or loss_valid[127] != 0 or loss_valid[
                128] != 0:
        l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down4, depth_gt_for_loss_down4)
    else:
        l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[129] != 0:
        l_down4_ssim = torch.clamp(
            (1 - ssim(depth_pred_for_loss_down4, depth_gt_for_loss_down4,
                      val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_down4_ssim = invalid_input

    if loss_valid[130] != 0 or loss_valid[131] != 0 or loss_valid[132] != 0 or loss_valid[133] != 0:
        l_down4_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down4, depth_gt_for_loss_down4,
                                                   window_size=0)
        l_down4_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down4,
                                                        depth_gt_for_loss_down4, window_size=2)
        l_down4_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down4,
                                                        depth_gt_for_loss_down4, window_size=8)
        l_down4_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down4,
                                                        depth_gt_for_loss_down4, window_size=32)
    else:
        l_down4_ndepth, l_down4_ndepth_win5, l_down4_ndepth_win17, l_down4_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[134] != 0:
        l_down4_geo = (
                              l_down4_depth * l_down4_depth_dx * l_down4_depth_dy * l_down4_depth_norm * l_down4_depth_dx2 * l_down4_depth_dxy * l_down4_depth_dy2 * l_down4_depth_dx_norm * l_down4_depth_dy_norm
                              * l_down4_ssim * l_down4_ndepth * l_down4_ndepth_win5 * l_down4_ndepth_win17 * l_down4_ndepth_win65) ** (
                              1 / 12)
    else:
        l_down4_geo = (
                              l_down4_depth * l_down4_depth_dx * l_down4_depth_dy * l_down4_depth_norm * l_down4_depth_dx2 * l_down4_depth_dxy * l_down4_depth_dy2 * l_down4_depth_dx_norm * l_down4_depth_dy_norm
                              * l_down4_ssim * l_down4_ndepth * l_down4_ndepth_win5 * l_down4_ndepth_win17 * l_down4_ndepth_win65) ** (
                              1 / 12)

    if \
            loss_valid[135] != 0 or loss_valid[136] != 0 or loss_valid[137] != 0 or loss_valid[
                138] != 0 or loss_valid[139] != 0 or \
                    loss_valid[140] != 0 or loss_valid[141] != 0 or loss_valid[142] != 0 or loss_valid[
                143] != 0:
        l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down5, depth_gt_for_loss_down5)
    else:
        l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[144] != 0:
        l_down5_ssim = torch.clamp(
            (1 - ssim(depth_pred_for_loss_down5, depth_gt_for_loss_down5,
                      val_range=1000.0 / 10.0)) * 0.5, 0,
            1)
    else:
        l_down5_ssim = invalid_input

    if loss_valid[145] != 0 or loss_valid[146] != 0 or loss_valid[147] != 0 or loss_valid[148] != 0:
        l_down5_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down5, depth_gt_for_loss_down5,
                                                   window_size=0)
        l_down5_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down5,
                                                        depth_gt_for_loss_down5, window_size=2)
        l_down5_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down5,
                                                        depth_gt_for_loss_down5, window_size=8)
        l_down5_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down5,
                                                        depth_gt_for_loss_down5, window_size=32)
    else:
        l_down5_ndepth, l_down5_ndepth_win5, l_down5_ndepth_win17, l_down5_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if loss_valid[149] != 0:
        l_down5_geo = (
                              l_down5_depth * l_down5_depth_dx * l_down5_depth_dy * l_down5_depth_norm * l_down5_depth_dx2 * l_down5_depth_dxy * l_down5_depth_dy2 * l_down5_depth_dx_norm * l_down5_depth_dy_norm
                              * l_down5_ssim * l_down5_ndepth * l_down5_ndepth_win5 * l_down5_ndepth_win17 * l_down5_ndepth_win65) ** (
                              1 / 12)
    else:
        l_down5_geo = (
                              l_down5_depth * l_down5_depth_dx * l_down5_depth_dy * l_down5_depth_norm * l_down5_depth_dx2 * l_down5_depth_dxy * l_down5_depth_dy2 * l_down5_depth_dx_norm * l_down5_depth_dy_norm
                              * l_down5_ssim * l_down5_ndepth * l_down5_ndepth_win5 * l_down5_ndepth_win17 * l_down5_ndepth_win65) ** (
                              1 / 12)

    return l_rmse, l_rmse_log, l_abs_rel, l_sqr_rel, l_log10, l_delta1, l_delta2, l_delta3, l_metric3, l_metric8, \
           l_si_rmse, l_si_rmse_log, l_si_abs_rel, l_si_sqr_rel, l_si_log10, l_si_delta1, l_si_delta2, l_si_delta3, l_si_metric3, l_si_metric8, l_depth, \
           l_depth_dx, l_depth_dy, l_depth_norm, l_depth_dx2, l_depth_dxy, l_depth_dy2, l_depth_dx_norm, l_depth_dy_norm, l_ssim, l_ndepth, l_ndepth_win5, l_ndepth_win17, l_ndepth_win65, l_geo, \
           l_log_depth, l_log_depth_dx, l_log_depth_dy, l_log_depth_norm, l_log_depth_dx2, l_log_depth_dxy, l_log_depth_dy2, l_log_depth_dx_norm, l_log_depth_dy_norm, l_log_ssim, l_log_ndepth, l_log_ndepth_win5, l_log_ndepth_win17, l_log_ndepth_win65, l_log_geo, \
           l_inv_depth, l_inv_depth_dx, l_inv_depth_dy, l_inv_depth_norm, l_inv_depth_dx2, l_inv_depth_dxy, l_inv_depth_dy2, l_inv_depth_dx_norm, l_inv_depth_dy_norm, l_inv_ssim, l_inv_ndepth, l_inv_ndepth_win5, l_inv_ndepth_win17, l_inv_ndepth_win65, l_inv_geo, l_all_geo, \
           l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm, l_down1_ssim, l_down1_ndepth, l_down1_ndepth_win5, l_down1_ndepth_win17, l_down1_ndepth_win65, l_down1_geo, \
           l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm, l_down2_ssim, l_down2_ndepth, l_down2_ndepth_win5, l_down2_ndepth_win17, l_down2_ndepth_win65, l_down2_geo, \
           l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm, l_down3_ssim, l_down3_ndepth, l_down3_ndepth_win5, l_down3_ndepth_win17, l_down3_ndepth_win65, l_down3_geo, \
           l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm, l_down4_ssim, l_down4_ndepth, l_down4_ndepth_win5, l_down4_ndepth_win17, l_down4_ndepth_win65, l_down4_geo, \
           l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm, l_down5_ssim, l_down5_ndepth, l_down5_ndepth_win5, l_down5_ndepth_win17, l_down5_ndepth_win65, l_down5_geo

def get_loss_1batch(batch_size, current_batch_size, index_iter, num_data, loss_weights, scores,
                    l_rmse, l_rmse_log, l_abs_rel, l_sqr_rel, l_log10, l_delta1, l_delta2, l_delta3, l_metric3, l_metric8,
                    l_si_rmse, l_si_rmse_log, l_si_abs_rel, l_si_sqr_rel, l_si_log10, l_si_delta1, l_si_delta2, l_si_delta3, l_si_metric3, l_si_metric8, l_depth,
                    l_depth_dx, l_depth_dy, l_depth_norm, l_depth_dx2, l_depth_dxy, l_depth_dy2, l_depth_dx_norm, l_depth_dy_norm, l_ssim, l_ndepth, l_ndepth_win5, l_ndepth_win17, l_ndepth_win65, l_geo,
                    l_log_depth, l_log_depth_dx, l_log_depth_dy, l_log_depth_norm, l_log_depth_dx2, l_log_depth_dxy, l_log_depth_dy2, l_log_depth_dx_norm, l_log_depth_dy_norm, l_log_ssim, l_log_ndepth, l_log_ndepth_win5, l_log_ndepth_win17, l_log_ndepth_win65, l_log_geo,
                    l_inv_depth, l_inv_depth_dx, l_inv_depth_dy, l_inv_depth_norm, l_inv_depth_dx2, l_inv_depth_dxy, l_inv_depth_dy2, l_inv_depth_dx_norm, l_inv_depth_dy_norm, l_inv_ssim, l_inv_ndepth, l_inv_ndepth_win5, l_inv_ndepth_win17, l_inv_ndepth_win65, l_inv_geo, l_all_geo,
                    l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm, l_down1_ssim, l_down1_ndepth, l_down1_ndepth_win5, l_down1_ndepth_win17, l_down1_ndepth_win65, l_down1_geo,
                    l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm, l_down2_ssim, l_down2_ndepth, l_down2_ndepth_win5, l_down2_ndepth_win17, l_down2_ndepth_win65, l_down2_geo,
                    l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm, l_down3_ssim, l_down3_ndepth, l_down3_ndepth_win5, l_down3_ndepth_win17, l_down3_ndepth_win65, l_down3_geo,
                    l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm, l_down4_ssim, l_down4_ndepth, l_down4_ndepth_win5, l_down4_ndepth_win17, l_down4_ndepth_win65, l_down4_geo,
                    l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm, l_down5_ssim, l_down5_ndepth, l_down5_ndepth_win5, l_down5_ndepth_win17, l_down5_ndepth_win65, l_down5_geo):

    l_custom = torch.zeros(current_batch_size).cuda(torch.device("cuda:0"))
    for index_batch in range(current_batch_size):
        index_record = batch_size * index_iter + index_batch

        if index_batch == 0:
            loss = 0

        if index_record < num_data:
            loss_1batch = 0
            if loss_weights[0] != 0:
                loss_1batch = loss_1batch + loss_weights[0] * l_rmse[index_batch]
            if loss_weights[1] != 0:
                loss_1batch = loss_1batch + loss_weights[1] * l_rmse_log[index_batch]
            if loss_weights[2] != 0:
                loss_1batch = loss_1batch + loss_weights[2] * l_abs_rel[index_batch]
            if loss_weights[3] != 0:
                loss_1batch = loss_1batch + loss_weights[3] * l_sqr_rel[index_batch]
            if loss_weights[4] != 0:
                loss_1batch = loss_1batch + loss_weights[4] * l_log10[index_batch]
            if loss_weights[5] != 0:
                loss_1batch = loss_1batch + loss_weights[5] * l_delta1[index_batch]
            if loss_weights[6] != 0:
                loss_1batch = loss_1batch + loss_weights[6] * l_delta2[index_batch]
            if loss_weights[7] != 0:
                loss_1batch = loss_1batch + loss_weights[7] * l_delta3[index_batch]
            if loss_weights[8] != 0:
                loss_1batch = loss_1batch + loss_weights[8] * l_metric3[index_batch]
            if loss_weights[9] != 0:
                loss_1batch = loss_1batch + loss_weights[9] * l_metric8[index_batch]
            if loss_weights[10] != 0:
                loss_1batch = loss_1batch + loss_weights[10] * l_si_rmse[index_batch]
            if loss_weights[11] != 0:
                loss_1batch = loss_1batch + loss_weights[11] * l_si_rmse_log[index_batch]
            if loss_weights[12] != 0:
                loss_1batch = loss_1batch + loss_weights[12] * l_si_abs_rel[index_batch]
            if loss_weights[13] != 0:
                loss_1batch = loss_1batch + loss_weights[13] * l_si_sqr_rel[index_batch]
            if loss_weights[14] != 0:
                loss_1batch = loss_1batch + loss_weights[14] * l_si_log10[index_batch]
            if loss_weights[15] != 0:
                loss_1batch = loss_1batch + loss_weights[15] * l_si_delta1[index_batch]
            if loss_weights[16] != 0:
                loss_1batch = loss_1batch + loss_weights[16] * l_si_delta2[index_batch]
            if loss_weights[17] != 0:
                loss_1batch = loss_1batch + loss_weights[17] * l_si_delta3[index_batch]
            if loss_weights[18] != 0:
                loss_1batch = loss_1batch + loss_weights[18] * l_si_metric3[index_batch]
            if loss_weights[19] != 0:
                loss_1batch = loss_1batch + loss_weights[19] * l_si_metric8[index_batch]
            if loss_weights[23] != 0:
                loss_1batch = loss_1batch + loss_weights[23] * 0
            if loss_weights[24] != 0:
                loss_1batch = loss_1batch + loss_weights[24] * 0
            if loss_weights[25] != 0:
                loss_1batch = loss_1batch + loss_weights[25] * l_depth[index_batch]
            if loss_weights[26] != 0:
                loss_1batch = loss_1batch + loss_weights[26] * l_depth_dx[index_batch]
            if loss_weights[27] != 0:
                loss_1batch = loss_1batch + loss_weights[27] * l_depth_dy[index_batch]
            if loss_weights[28] != 0:
                loss_1batch = loss_1batch + loss_weights[28] * l_depth_norm[index_batch]
            if loss_weights[29] != 0:
                loss_1batch = loss_1batch + loss_weights[29] * l_depth_dx2[index_batch]
            if loss_weights[30] != 0:
                loss_1batch = loss_1batch + loss_weights[30] * l_depth_dxy[index_batch]
            if loss_weights[31] != 0:
                loss_1batch = loss_1batch + loss_weights[31] * l_depth_dy2[index_batch]
            if loss_weights[32] != 0:
                loss_1batch = loss_1batch + loss_weights[32] * l_depth_dx_norm[index_batch]
            if loss_weights[33] != 0:
                loss_1batch = loss_1batch + loss_weights[33] * l_depth_dy_norm[index_batch]
            if loss_weights[34] != 0:
                loss_1batch = loss_1batch + loss_weights[34] * l_ssim[index_batch]
            if loss_weights[35] != 0:
                loss_1batch = loss_1batch + loss_weights[35] * l_ndepth[index_batch]
            if loss_weights[36] != 0:
                loss_1batch = loss_1batch + loss_weights[36] * l_ndepth_win5[index_batch]
            if loss_weights[37] != 0:
                loss_1batch = loss_1batch + loss_weights[37] * l_ndepth_win17[index_batch]
            if loss_weights[38] != 0:
                loss_1batch = loss_1batch + loss_weights[38] * l_ndepth_win65[index_batch]
            if loss_weights[39] != 0:
                loss_1batch = loss_1batch + loss_weights[39] * l_geo[index_batch]
            if loss_weights[40] != 0:
                loss_1batch = loss_1batch + loss_weights[40] * l_log_depth[index_batch]
            if loss_weights[41] != 0:
                loss_1batch = loss_1batch + loss_weights[41] * l_log_depth_dx[index_batch]
            if loss_weights[42] != 0:
                loss_1batch = loss_1batch + loss_weights[42] * l_log_depth_dy[index_batch]
            if loss_weights[43] != 0:
                loss_1batch = loss_1batch + loss_weights[43] * l_log_depth_norm[index_batch]
            if loss_weights[44] != 0:
                loss_1batch = loss_1batch + loss_weights[44] * l_log_depth_dx2[index_batch]
            if loss_weights[45] != 0:
                loss_1batch = loss_1batch + loss_weights[45] * l_log_depth_dxy[index_batch]
            if loss_weights[46] != 0:
                loss_1batch = loss_1batch + loss_weights[46] * l_log_depth_dy2[index_batch]
            if loss_weights[47] != 0:
                loss_1batch = loss_1batch + loss_weights[47] * l_log_depth_dx_norm[index_batch]
            if loss_weights[48] != 0:
                loss_1batch = loss_1batch + loss_weights[48] * l_log_depth_dy_norm[index_batch]
            if loss_weights[49] != 0:
                loss_1batch = loss_1batch + loss_weights[49] * l_log_ssim[index_batch]
            if loss_weights[50] != 0:
                loss_1batch = loss_1batch + loss_weights[50] * l_log_ndepth[index_batch]
            if loss_weights[51] != 0:
                loss_1batch = loss_1batch + loss_weights[51] * l_log_ndepth_win5[index_batch]
            if loss_weights[52] != 0:
                loss_1batch = loss_1batch + loss_weights[52] * l_log_ndepth_win17[index_batch]
            if loss_weights[53] != 0:
                loss_1batch = loss_1batch + loss_weights[53] * l_log_ndepth_win65[index_batch]
            if loss_weights[54] != 0:
                loss_1batch = loss_1batch + loss_weights[54] * l_log_geo[index_batch]
            if loss_weights[55] != 0:
                loss_1batch = loss_1batch + loss_weights[55] * l_inv_depth[index_batch]
            if loss_weights[56] != 0:
                loss_1batch = loss_1batch + loss_weights[56] * l_inv_depth_dx[index_batch]
            if loss_weights[57] != 0:
                loss_1batch = loss_1batch + loss_weights[57] * l_inv_depth_dy[index_batch]
            if loss_weights[58] != 0:
                loss_1batch = loss_1batch + loss_weights[58] * l_inv_depth_norm[index_batch]
            if loss_weights[59] != 0:
                loss_1batch = loss_1batch + loss_weights[59] * l_inv_depth_dx2[index_batch]
            if loss_weights[60] != 0:
                loss_1batch = loss_1batch + loss_weights[60] * l_inv_depth_dxy[index_batch]
            if loss_weights[61] != 0:
                loss_1batch = loss_1batch + loss_weights[61] * l_inv_depth_dy2[index_batch]
            if loss_weights[62] != 0:
                loss_1batch = loss_1batch + loss_weights[62] * l_inv_depth_dx_norm[index_batch]
            if loss_weights[63] != 0:
                loss_1batch = loss_1batch + loss_weights[63] * l_inv_depth_dy_norm[index_batch]
            if loss_weights[64] != 0:
                loss_1batch = loss_1batch + loss_weights[64] * l_inv_ssim[index_batch]
            if loss_weights[65] != 0:
                loss_1batch = loss_1batch + loss_weights[65] * l_inv_ndepth[index_batch]
            if loss_weights[66] != 0:
                loss_1batch = loss_1batch + loss_weights[66] * l_inv_ndepth_win5[index_batch]
            if loss_weights[67] != 0:
                loss_1batch = loss_1batch + loss_weights[67] * l_inv_ndepth_win17[index_batch]
            if loss_weights[68] != 0:
                loss_1batch = loss_1batch + loss_weights[68] * l_inv_ndepth_win65[index_batch]
            if loss_weights[69] != 0:
                loss_1batch = loss_1batch + loss_weights[69] * l_inv_geo[index_batch]
            if loss_weights[70] != 0:
                loss_1batch = loss_1batch + loss_weights[70] * 0
            if loss_weights[71] != 0:
                loss_1batch = loss_1batch + loss_weights[71] * 0
            if loss_weights[72] != 0:
                loss_1batch = loss_1batch + loss_weights[72] * 0
            if loss_weights[73] != 0:
                loss_1batch = loss_1batch + loss_weights[73] * l_all_geo[index_batch]
            if loss_weights[75] != 0:
                loss_1batch = loss_1batch + loss_weights[75] * l_down1_depth[index_batch]
            if loss_weights[76] != 0:
                loss_1batch = loss_1batch + loss_weights[76] * l_down1_depth_dx[index_batch]
            if loss_weights[77] != 0:
                loss_1batch = loss_1batch + loss_weights[77] * l_down1_depth_dy[index_batch]
            if loss_weights[78] != 0:
                loss_1batch = loss_1batch + loss_weights[78] * l_down1_depth_norm[index_batch]
            if loss_weights[79] != 0:
                loss_1batch = loss_1batch + loss_weights[79] * l_down1_depth_dx2[index_batch]
            if loss_weights[80] != 0:
                loss_1batch = loss_1batch + loss_weights[80] * l_down1_depth_dxy[index_batch]
            if loss_weights[81] != 0:
                loss_1batch = loss_1batch + loss_weights[81] * l_down1_depth_dy2[index_batch]
            if loss_weights[82] != 0:
                loss_1batch = loss_1batch + loss_weights[82] * l_down1_depth_dx_norm[index_batch]
            if loss_weights[83] != 0:
                loss_1batch = loss_1batch + loss_weights[83] * l_down1_depth_dy_norm[index_batch]
            if loss_weights[84] != 0:
                loss_1batch = loss_1batch + loss_weights[84] * l_down1_ssim[index_batch]
            if loss_weights[85] != 0:
                loss_1batch = loss_1batch + loss_weights[85] * l_down1_ndepth[index_batch]
            if loss_weights[86] != 0:
                loss_1batch = loss_1batch + loss_weights[86] * l_down1_ndepth_win5[index_batch]
            if loss_weights[87] != 0:
                loss_1batch = loss_1batch + loss_weights[87] * l_down1_ndepth_win17[index_batch]
            if loss_weights[88] != 0:
                loss_1batch = loss_1batch + loss_weights[88] * l_down1_ndepth_win65[index_batch]
            if loss_weights[89] != 0:
                loss_1batch = loss_1batch + loss_weights[89] * l_down1_geo[index_batch]
            if loss_weights[90] != 0:
                loss_1batch = loss_1batch + loss_weights[90] * l_down2_depth[index_batch]
            if loss_weights[91] != 0:
                loss_1batch = loss_1batch + loss_weights[91] * l_down2_depth_dx[index_batch]
            if loss_weights[92] != 0:
                loss_1batch = loss_1batch + loss_weights[92] * l_down2_depth_dy[index_batch]
            if loss_weights[93] != 0:
                loss_1batch = loss_1batch + loss_weights[93] * l_down2_depth_norm[index_batch]
            if loss_weights[94] != 0:
                loss_1batch = loss_1batch + loss_weights[94] * l_down2_depth_dx2[index_batch]
            if loss_weights[95] != 0:
                loss_1batch = loss_1batch + loss_weights[95] * l_down2_depth_dxy[index_batch]
            if loss_weights[96] != 0:
                loss_1batch = loss_1batch + loss_weights[96] * l_down2_depth_dy2[index_batch]
            if loss_weights[97] != 0:
                loss_1batch = loss_1batch + loss_weights[97] * l_down2_depth_dx_norm[index_batch]
            if loss_weights[98] != 0:
                loss_1batch = loss_1batch + loss_weights[98] * l_down2_depth_dy_norm[index_batch]
            if loss_weights[99] != 0:
                loss_1batch = loss_1batch + loss_weights[99] * l_down2_ssim[index_batch]
            if loss_weights[100] != 0:
                loss_1batch = loss_1batch + loss_weights[100] * l_down2_ndepth[index_batch]
            if loss_weights[101] != 0:
                loss_1batch = loss_1batch + loss_weights[101] * l_down2_ndepth_win5[index_batch]
            if loss_weights[102] != 0:
                loss_1batch = loss_1batch + loss_weights[102] * l_down2_ndepth_win17[index_batch]
            if loss_weights[103] != 0:
                loss_1batch = loss_1batch + loss_weights[103] * l_down2_ndepth_win65[index_batch]
            if loss_weights[104] != 0:
                loss_1batch = loss_1batch + loss_weights[104] * l_down2_geo[index_batch]
            if loss_weights[105] != 0:
                loss_1batch = loss_1batch + loss_weights[105] * l_down3_depth[index_batch]
            if loss_weights[106] != 0:
                loss_1batch = loss_1batch + loss_weights[106] * l_down3_depth_dx[index_batch]
            if loss_weights[107] != 0:
                loss_1batch = loss_1batch + loss_weights[107] * l_down3_depth_dy[index_batch]
            if loss_weights[108] != 0:
                loss_1batch = loss_1batch + loss_weights[108] * l_down3_depth_norm[index_batch]
            if loss_weights[109] != 0:
                loss_1batch = loss_1batch + loss_weights[109] * l_down3_depth_dx2[index_batch]
            if loss_weights[110] != 0:
                loss_1batch = loss_1batch + loss_weights[110] * l_down3_depth_dxy[index_batch]
            if loss_weights[111] != 0:
                loss_1batch = loss_1batch + loss_weights[111] * l_down3_depth_dy2[index_batch]
            if loss_weights[112] != 0:
                loss_1batch = loss_1batch + loss_weights[112] * l_down3_depth_dx_norm[index_batch]
            if loss_weights[113] != 0:
                loss_1batch = loss_1batch + loss_weights[113] * l_down3_depth_dy_norm[index_batch]
            if loss_weights[114] != 0:
                loss_1batch = loss_1batch + loss_weights[114] * l_down3_ssim[index_batch]
            if loss_weights[115] != 0:
                loss_1batch = loss_1batch + loss_weights[115] * l_down3_ndepth[index_batch]
            if loss_weights[116] != 0:
                loss_1batch = loss_1batch + loss_weights[116] * l_down3_ndepth_win5[index_batch]
            if loss_weights[117] != 0:
                loss_1batch = loss_1batch + loss_weights[117] * l_down3_ndepth_win17[index_batch]
            if loss_weights[118] != 0:
                loss_1batch = loss_1batch + loss_weights[118] * l_down3_ndepth_win65[index_batch]
            if loss_weights[119] != 0:
                loss_1batch = loss_1batch + loss_weights[119] * l_down3_geo[index_batch]
            if loss_weights[120] != 0:
                loss_1batch = loss_1batch + loss_weights[120] * l_down4_depth[index_batch]
            if loss_weights[121] != 0:
                loss_1batch = loss_1batch + loss_weights[121] * l_down4_depth_dx[index_batch]
            if loss_weights[122] != 0:
                loss_1batch = loss_1batch + loss_weights[122] * l_down4_depth_dy[index_batch]
            if loss_weights[123] != 0:
                loss_1batch = loss_1batch + loss_weights[123] * l_down4_depth_norm[index_batch]
            if loss_weights[124] != 0:
                loss_1batch = loss_1batch + loss_weights[124] * l_down4_depth_dx2[index_batch]
            if loss_weights[125] != 0:
                loss_1batch = loss_1batch + loss_weights[125] * l_down4_depth_dxy[index_batch]
            if loss_weights[126] != 0:
                loss_1batch = loss_1batch + loss_weights[126] * l_down4_depth_dy2[index_batch]
            if loss_weights[127] != 0:
                loss_1batch = loss_1batch + loss_weights[127] * l_down4_depth_dx_norm[index_batch]
            if loss_weights[128] != 0:
                loss_1batch = loss_1batch + loss_weights[128] * l_down4_depth_dy_norm[index_batch]
            if loss_weights[129] != 0:
                loss_1batch = loss_1batch + loss_weights[129] * l_down4_ssim[index_batch]
            if loss_weights[130] != 0:
                loss_1batch = loss_1batch + loss_weights[130] * l_down4_ndepth[index_batch]
            if loss_weights[131] != 0:
                loss_1batch = loss_1batch + loss_weights[131] * l_down4_ndepth_win5[index_batch]
            if loss_weights[132] != 0:
                loss_1batch = loss_1batch + loss_weights[132] * l_down4_ndepth_win17[index_batch]
            if loss_weights[133] != 0:
                loss_1batch = loss_1batch + loss_weights[133] * l_down4_ndepth_win65[index_batch]
            if loss_weights[134] != 0:
                loss_1batch = loss_1batch + loss_weights[134] * l_down4_geo[index_batch]
            if loss_weights[135] != 0:
                loss_1batch = loss_1batch + loss_weights[135] * l_down5_depth[index_batch]
            if loss_weights[136] != 0:
                loss_1batch = loss_1batch + loss_weights[136] * l_down5_depth_dx[index_batch]
            if loss_weights[137] != 0:
                loss_1batch = loss_1batch + loss_weights[137] * l_down5_depth_dy[index_batch]
            if loss_weights[138] != 0:
                loss_1batch = loss_1batch + loss_weights[138] * l_down5_depth_norm[index_batch]
            if loss_weights[139] != 0:
                loss_1batch = loss_1batch + loss_weights[139] * l_down5_depth_dx2[index_batch]
            if loss_weights[140] != 0:
                loss_1batch = loss_1batch + loss_weights[140] * l_down5_depth_dxy[index_batch]
            if loss_weights[141] != 0:
                loss_1batch = loss_1batch + loss_weights[141] * l_down5_depth_dy2[index_batch]
            if loss_weights[142] != 0:
                loss_1batch = loss_1batch + loss_weights[142] * l_down5_depth_dx_norm[index_batch]
            if loss_weights[143] != 0:
                loss_1batch = loss_1batch + loss_weights[143] * l_down5_depth_dy_norm[index_batch]
            if loss_weights[144] != 0:
                loss_1batch = loss_1batch + loss_weights[144] * l_down5_ssim[index_batch]
            if loss_weights[145] != 0:
                loss_1batch = loss_1batch + loss_weights[145] * l_down5_ndepth[index_batch]
            if loss_weights[146] != 0:
                loss_1batch = loss_1batch + loss_weights[146] * l_down5_ndepth_win5[index_batch]
            if loss_weights[147] != 0:
                loss_1batch = loss_1batch + loss_weights[147] * l_down5_ndepth_win17[index_batch]
            if loss_weights[148] != 0:
                loss_1batch = loss_1batch + loss_weights[148] * l_down5_ndepth_win65[index_batch]
            if loss_weights[149] != 0:
                loss_1batch = loss_1batch + loss_weights[149] * l_down5_geo[index_batch]

            l_custom[index_batch] = loss_1batch
            loss = loss + loss_1batch / current_batch_size

            scores[index_record, :] = [
                l_rmse[index_batch],                l_rmse_log[index_batch],          l_abs_rel[index_batch],            l_sqr_rel[index_batch],            l_log10[index_batch],
                l_delta1[index_batch],              l_delta2[index_batch],            l_delta3[index_batch],             l_metric3[index_batch],            l_metric8[index_batch],
                l_si_rmse[index_batch],             l_si_rmse_log[index_batch],       l_si_abs_rel[index_batch],         l_si_sqr_rel[index_batch],         l_si_log10[index_batch],
                l_si_delta1[index_batch],           l_si_delta2[index_batch],         l_si_delta3[index_batch],          l_si_metric3[index_batch],         l_si_metric8[index_batch],
                0,                                  0,                    			  0,                                 0,                    				0,
                l_depth[index_batch],               l_depth_dx[index_batch],          l_depth_dy[index_batch],           l_depth_norm[index_batch],         l_depth_dx2[index_batch],
                l_depth_dxy[index_batch],           l_depth_dy2[index_batch],         l_depth_dx_norm[index_batch],      l_depth_dy_norm[index_batch],      l_ssim[index_batch],
                l_ndepth[index_batch],              l_ndepth_win5[index_batch],       l_ndepth_win17[index_batch],       l_ndepth_win65[index_batch],       l_geo[index_batch],
                l_log_depth[index_batch],           l_log_depth_dx[index_batch],      l_log_depth_dy[index_batch],       l_log_depth_norm[index_batch],     l_log_depth_dx2[index_batch],
                l_log_depth_dxy[index_batch],       l_log_depth_dy2[index_batch],     l_log_depth_dx_norm[index_batch],  l_log_depth_dy_norm[index_batch],  l_log_ssim[index_batch],
                l_log_ndepth[index_batch],          l_log_ndepth_win5[index_batch],   l_log_ndepth_win17[index_batch],   l_log_ndepth_win65[index_batch],   l_log_geo[index_batch],
                l_inv_depth[index_batch],           l_inv_depth_dx[index_batch],      l_inv_depth_dy[index_batch],       l_inv_depth_norm[index_batch],     l_inv_depth_dx2[index_batch],
                l_inv_depth_dxy[index_batch],       l_inv_depth_dy2[index_batch],     l_inv_depth_dx_norm[index_batch],  l_inv_depth_dy_norm[index_batch],  l_inv_ssim[index_batch],
                l_inv_ndepth[index_batch],          l_inv_ndepth_win5[index_batch],   l_inv_ndepth_win17[index_batch],   l_inv_ndepth_win65[index_batch],   l_inv_geo[index_batch],
                0,								    0,        				          0,      				             l_all_geo[index_batch],            l_custom[index_batch],
                l_down1_depth[index_batch],         l_down1_depth_dx[index_batch],    l_down1_depth_dy[index_batch],     l_down1_depth_norm[index_batch],   l_down1_depth_dx2[index_batch],
                l_down1_depth_dxy[index_batch],     l_down1_depth_dy2[index_batch],   l_down1_depth_dx_norm[index_batch],l_down1_depth_dy_norm[index_batch],l_down1_ssim[index_batch],
                l_down1_ndepth[index_batch],        l_down1_ndepth_win5[index_batch], l_down1_ndepth_win17[index_batch], l_down1_ndepth_win65[index_batch], l_down1_geo[index_batch],
                l_down2_depth[index_batch],         l_down2_depth_dx[index_batch],    l_down2_depth_dy[index_batch],     l_down2_depth_norm[index_batch],   l_down2_depth_dx2[index_batch],
                l_down2_depth_dxy[index_batch],     l_down2_depth_dy2[index_batch],   l_down2_depth_dx_norm[index_batch],l_down2_depth_dy_norm[index_batch],l_down2_ssim[index_batch],
                l_down2_ndepth[index_batch],        l_down2_ndepth_win5[index_batch], l_down2_ndepth_win17[index_batch], l_down2_ndepth_win65[index_batch], l_down2_geo[index_batch],
                l_down3_depth[index_batch],         l_down3_depth_dx[index_batch],    l_down3_depth_dy[index_batch],     l_down3_depth_norm[index_batch],   l_down3_depth_dx2[index_batch],
                l_down3_depth_dxy[index_batch],     l_down3_depth_dy2[index_batch],   l_down3_depth_dx_norm[index_batch],l_down3_depth_dy_norm[index_batch],l_down3_ssim[index_batch],
                l_down3_ndepth[index_batch],        l_down3_ndepth_win5[index_batch], l_down3_ndepth_win17[index_batch], l_down3_ndepth_win65[index_batch], l_down3_geo[index_batch],
                l_down4_depth[index_batch],         l_down4_depth_dx[index_batch],    l_down4_depth_dy[index_batch],     l_down4_depth_norm[index_batch],   l_down4_depth_dx2[index_batch],
                l_down4_depth_dxy[index_batch],     l_down4_depth_dy2[index_batch],   l_down4_depth_dx_norm[index_batch],l_down4_depth_dy_norm[index_batch],l_down4_ssim[index_batch],
                l_down4_ndepth[index_batch],        l_down4_ndepth_win5[index_batch], l_down4_ndepth_win17[index_batch], l_down4_ndepth_win65[index_batch], l_down4_geo[index_batch],
                l_down5_depth[index_batch],         l_down5_depth_dx[index_batch],    l_down5_depth_dy[index_batch],     l_down5_depth_norm[index_batch],   l_down5_depth_dx2[index_batch],
                l_down5_depth_dxy[index_batch],     l_down5_depth_dy2[index_batch],   l_down5_depth_dx_norm[index_batch],l_down5_depth_dy_norm[index_batch],l_down5_ssim[index_batch],
                l_down5_ndepth[index_batch],        l_down5_ndepth_win5[index_batch], l_down5_ndepth_win17[index_batch], l_down5_ndepth_win65[index_batch], l_down5_geo[index_batch]
            ]

    return loss, l_custom, scores

def get_metric_1batch(batch_size, current_batch_size, index_iter, num_data, metrics,
                      rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8,
                      si_rmse, si_rmse_log, si_abs_rel, si_sqr_rel, si_log10, si_delta1, si_delta2, si_delta3, si_metric3, si_metric8,
                      corr_pearson, corr_spearman, corr_kendal):

    for index_batch in range(current_batch_size):
        index_record = batch_size * index_iter + index_batch

        if index_record < num_data:
            # Compute the metrics & loss
            metrics[index_record, :] = [
                rmse.cpu().detach().numpy()[index_batch],               rmse_log.cpu().detach().numpy()[index_batch],           abs_rel.cpu().detach().numpy()[index_batch],            sqr_rel.cpu().detach().numpy()[index_batch],            log10.cpu().detach().numpy()[index_batch],
                delta1.cpu().detach().numpy()[index_batch],             delta2.cpu().detach().numpy()[index_batch],             delta3.cpu().detach().numpy()[index_batch],             metric3.cpu().detach().numpy()[index_batch],            metric8.cpu().detach().numpy()[index_batch],
                si_rmse.cpu().detach().numpy()[index_batch],            si_rmse_log.cpu().detach().numpy()[index_batch],        si_abs_rel.cpu().detach().numpy()[index_batch],         si_sqr_rel.cpu().detach().numpy()[index_batch],         si_log10.cpu().detach().numpy()[index_batch],
                si_delta1.cpu().detach().numpy()[index_batch],          si_delta2.cpu().detach().numpy()[index_batch],          si_delta3.cpu().detach().numpy()[index_batch],          si_metric3.cpu().detach().numpy()[index_batch],         si_metric8.cpu().detach().numpy()[index_batch],
                corr_pearson[index_batch],       corr_spearman[index_batch],      corr_kendal[index_batch],        0,                               0,
            ]

    return metrics

def log_write_train(writer, train_scores, batch_size, index_iter, niter, tensorboard_prefix='d0'):
    i = index_iter
    # Log to tensorboard
    writer.add_scalar(tensorboard_prefix + '_metric8/l_rmse', train_scores[(i-99)*batch_size:(i+1)*batch_size, 0].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_rmse_log', train_scores[(i-99)*batch_size:(i+1)*batch_size, 1].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_abs_rel', train_scores[(i-99)*batch_size:(i+1)*batch_size, 2].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_sqr_rel', train_scores[(i-99)*batch_size:(i+1)*batch_size, 3].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_log10', train_scores[(i-99)*batch_size:(i+1)*batch_size, 4].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_delta1', train_scores[(i-99)*batch_size:(i+1)*batch_size, 5].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_delta2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 6].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_delta3', train_scores[(i-99)*batch_size:(i+1)*batch_size, 7].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_metric3', train_scores[(i-99)*batch_size:(i+1)*batch_size, 8].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_metric8/l_metric8', train_scores[(i-99)*batch_size:(i+1)*batch_size, 9].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_rmse', train_scores[(i-99)*batch_size:(i+1)*batch_size, 10].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_rmse_log', train_scores[(i-99)*batch_size:(i+1)*batch_size, 11].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_abs_rel', train_scores[(i-99)*batch_size:(i+1)*batch_size, 12].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_sqr_rel', train_scores[(i-99)*batch_size:(i+1)*batch_size, 13].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_log10', train_scores[(i-99)*batch_size:(i+1)*batch_size, 14].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_delta1', train_scores[(i-99)*batch_size:(i+1)*batch_size, 15].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_delta2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 16].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_delta3', train_scores[(i-99)*batch_size:(i+1)*batch_size, 17].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_metric3', train_scores[(i-99)*batch_size:(i+1)*batch_size, 18].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_si_metric8/l_si_metric8', train_scores[(i-99)*batch_size:(i+1)*batch_size, 19].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_corr/corr_pearson', train_scores[(i-99)*batch_size:(i+1)*batch_size, 20].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_corr/corr_spearman', train_scores[(i-99)*batch_size:(i+1)*batch_size, 21].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_corr/corr_kendal', train_scores[(i-99)*batch_size:(i+1)*batch_size, 22].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 25].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 26].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 27].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 28].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 29].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 30].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 31].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 32].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 33].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 34].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 35].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 36].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 37].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 38].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_orig/l_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 39].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 40].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 41].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 42].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 43].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 44].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 45].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 46].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 47].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 48].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 49].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 50].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 51].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 52].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 53].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_log/l_log_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 54].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 55].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 56].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 57].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 58].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 59].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 60].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 61].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 62].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 63].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 64].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 65].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 66].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 67].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 68].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_inv/l_inv_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 69].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '/l_all_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 73].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '/l_custom', train_scores[(i-99)*batch_size:(i+1)*batch_size, 74].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 75].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 76].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 77].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 78].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 79].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 80].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 81].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 82].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 83].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 84].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 85].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 86].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 87].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 88].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down1/l_down1_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 89].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 90].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 91].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 92].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 93].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 94].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 95].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 96].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 97].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 98].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 99].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 100].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 101].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 102].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 103].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down2/l_down2_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 104].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 105].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 106].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 107].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 108].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 109].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 110].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 111].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 112].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 113].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 114].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 115].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 116].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 117].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 118].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down3/l_down3_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 119].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 120].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 121].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 122].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 123].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 124].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 125].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 126].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 127].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 128].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 129].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 130].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 131].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 132].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 133].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down4/l_down4_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 134].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 135].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_dx', train_scores[(i-99)*batch_size:(i+1)*batch_size, 136].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_dy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 137].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 138].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_dx2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 139].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_dxy', train_scores[(i-99)*batch_size:(i+1)*batch_size, 140].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_dy2', train_scores[(i-99)*batch_size:(i+1)*batch_size, 141].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_dx_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 142].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_dy_norm', train_scores[(i-99)*batch_size:(i+1)*batch_size, 143].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_depth_ssim', train_scores[(i-99)*batch_size:(i+1)*batch_size, 144].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_ndepth', train_scores[(i-99)*batch_size:(i+1)*batch_size, 145].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_ndepth_win5', train_scores[(i-99)*batch_size:(i+1)*batch_size, 146].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_ndepth_win17', train_scores[(i-99)*batch_size:(i+1)*batch_size, 147].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_ndepth_win65', train_scores[(i-99)*batch_size:(i+1)*batch_size, 148].mean(), niter)
    writer.add_scalar(tensorboard_prefix + '_down5/l_down5_geo', train_scores[(i-99)*batch_size:(i+1)*batch_size, 149].mean(), niter)

def log_write_test(writer, loss, test_scores_mean, test_metrics_mean, current_epoch, tensorboard_prefix):
    # Log to tensorboard
    writer.add_scalar(tensorboard_prefix + '/scores/Loss', loss, current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_rmse', test_scores_mean[0], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_rmse_log', test_scores_mean[1], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_abs_rel', test_scores_mean[2], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_sqr_rel', test_scores_mean[3], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_log10', test_scores_mean[4], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_delta1', test_scores_mean[5], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_delta2', test_scores_mean[6], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_delta3', test_scores_mean[7], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_metric3', test_scores_mean[8], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/matric8/l_metric8', test_scores_mean[9], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_rmse', test_scores_mean[10], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_rmse_log', test_scores_mean[11], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_abs_rel', test_scores_mean[12], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_sqr_rel', test_scores_mean[13], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_log10', test_scores_mean[14], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_delta1', test_scores_mean[15], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_delta2', test_scores_mean[16], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_delta3', test_scores_mean[17], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_metric3', test_scores_mean[18], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/si_matric8/l_si_metric8', test_scores_mean[19], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/corr/corr_pearson', test_scores_mean[20], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/corr/corr_spearman', test_scores_mean[21], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/corr/corr_kendal', test_scores_mean[22], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth', test_scores_mean[25], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_dx', test_scores_mean[26], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_dy', test_scores_mean[27], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_norm', test_scores_mean[28], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_dx2', test_scores_mean[29], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_dxy', test_scores_mean[30], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_dy2', test_scores_mean[31], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_dx_norm', test_scores_mean[32], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_dy_norm', test_scores_mean[33], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_depth_ssim', test_scores_mean[34], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_ndepth', test_scores_mean[35], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_ndepth_win5', test_scores_mean[36], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_ndepth_win17', test_scores_mean[37], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_ndepth_win65', test_scores_mean[38], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/original/l_geo', test_scores_mean[39], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth', test_scores_mean[40], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_dx', test_scores_mean[41], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_dy', test_scores_mean[42], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_norm', test_scores_mean[43], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_dx2', test_scores_mean[44], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_dxy', test_scores_mean[45], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_dy2', test_scores_mean[46], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_dx_norm', test_scores_mean[47], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_dy_norm', test_scores_mean[48], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_depth_ssim', test_scores_mean[49], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_ndepth', test_scores_mean[50], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_ndepth_win5', test_scores_mean[51], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_ndepth_win17', test_scores_mean[52], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_ndepth_win65', test_scores_mean[53], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/log/l_log_geo', test_scores_mean[54], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth', test_scores_mean[55], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_dx', test_scores_mean[56], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_dy', test_scores_mean[57], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_norm', test_scores_mean[58], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_dx2', test_scores_mean[59], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_dxy', test_scores_mean[60], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_dy2', test_scores_mean[61], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_dx_norm', test_scores_mean[62], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_dy_norm', test_scores_mean[63], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_depth_ssim', test_scores_mean[64], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_ndepth', test_scores_mean[65], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_ndepth_win5', test_scores_mean[66], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_ndepth_win17', test_scores_mean[67], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_ndepth_win65', test_scores_mean[68], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/inverse/l_inv_geo', test_scores_mean[69], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/l_all_geo', test_scores_mean[73], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth', test_scores_mean[75], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_dx', test_scores_mean[76], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_dy', test_scores_mean[77], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_norm', test_scores_mean[78], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_dx2', test_scores_mean[79], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_dxy', test_scores_mean[80], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_dy2', test_scores_mean[81], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_dx_norm', test_scores_mean[82], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_dy_norm', test_scores_mean[83], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_depth_ssim', test_scores_mean[84], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_ndepth', test_scores_mean[85], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_ndepth_win5', test_scores_mean[86], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_ndepth_win17', test_scores_mean[87], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_ndepth_win65', test_scores_mean[88], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down1/l_down1_geo', test_scores_mean[89], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth', test_scores_mean[90], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_dx', test_scores_mean[91], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_dy', test_scores_mean[92], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_norm', test_scores_mean[93], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_dx2', test_scores_mean[94], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_dxy', test_scores_mean[95], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_dy2', test_scores_mean[96], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_dx_norm', test_scores_mean[97], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_dy_norm', test_scores_mean[98], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_depth_ssim', test_scores_mean[99], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_ndepth', test_scores_mean[100], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_ndepth_win5', test_scores_mean[101], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_ndepth_win17', test_scores_mean[102], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_ndepth_win65', test_scores_mean[103], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down2/l_down2_geo', test_scores_mean[104], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth', test_scores_mean[105], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_dx', test_scores_mean[106], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_dy', test_scores_mean[107], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_norm', test_scores_mean[108], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_dx2', test_scores_mean[109], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_dxy', test_scores_mean[110], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_dy2', test_scores_mean[111], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_dx_norm', test_scores_mean[112], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_dy_norm', test_scores_mean[113], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_depth_ssim', test_scores_mean[114], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_ndepth', test_scores_mean[115], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_ndepth_win5', test_scores_mean[116], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_ndepth_win17', test_scores_mean[117], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_ndepth_win65', test_scores_mean[118], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down3/l_down3_geo', test_scores_mean[119], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth', test_scores_mean[120], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_dx', test_scores_mean[121], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_dy', test_scores_mean[122], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_norm', test_scores_mean[123], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_dx2', test_scores_mean[124], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_dxy', test_scores_mean[125], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_dy2', test_scores_mean[126], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_dx_norm', test_scores_mean[127], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_dy_norm', test_scores_mean[128], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_depth_ssim', test_scores_mean[129], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_ndepth', test_scores_mean[130], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_ndepth_win5', test_scores_mean[131], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_ndepth_win17', test_scores_mean[132], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_ndepth_win65', test_scores_mean[133], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down4/l_down4_geo', test_scores_mean[134], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth', test_scores_mean[135], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_dx', test_scores_mean[136], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_dy', test_scores_mean[137], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_norm', test_scores_mean[138], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_dx2', test_scores_mean[139], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_dxy', test_scores_mean[140], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_dy2', test_scores_mean[141], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_dx_norm', test_scores_mean[142], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_dy_norm', test_scores_mean[143], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_depth_ssim', test_scores_mean[144], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_ndepth', test_scores_mean[145], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_ndepth_win5', test_scores_mean[146], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_ndepth_win17', test_scores_mean[147], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_ndepth_win65', test_scores_mean[148], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/scores/down5/l_down5_geo', test_scores_mean[149], current_epoch)

    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/rmse', test_metrics_mean[0], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/rmse_log', test_metrics_mean[1], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/abs_rel', test_metrics_mean[2], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/sqr_rel', test_metrics_mean[3], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/log10', test_metrics_mean[4], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/delta1', test_metrics_mean[5], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/delta2', test_metrics_mean[6], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/delta3', test_metrics_mean[7], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/metric3', test_metrics_mean[8], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/matric8/metric8', test_metrics_mean[9], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_rmse', test_metrics_mean[10], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_rmse_log', test_metrics_mean[11], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_abs_rel', test_metrics_mean[12], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_sqr_rel', test_metrics_mean[13], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_log10', test_metrics_mean[14], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_delta1', test_metrics_mean[15], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_delta2', test_metrics_mean[16], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_delta3', test_metrics_mean[17], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_metric3', test_metrics_mean[18], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/si_matric8/si_metric8', test_metrics_mean[19], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/corr/corr_pearson', test_metrics_mean[20], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/corr/corr_spearman', test_metrics_mean[21], current_epoch)
    writer.add_scalar(tensorboard_prefix + '/metrics/corr/corr_kendal', test_metrics_mean[22], current_epoch)


def compute_1loss_with_record(depth_pred_for_loss, depth_gt_for_loss, batch_size, current_batch_size, i, num_train_data, train_scores):
    # compute loss
    l_depth = compute_1loss(depth_pred_for_loss, depth_gt_for_loss)

    # compute iter loss & train_scores
    loss, l_custom, train_scores = get_1loss_1batch(batch_size, current_batch_size, i, num_train_data, train_scores, l_depth)

    return loss, l_custom, train_scores

def compute_1loss(depth_pred_for_loss, depth_gt_for_loss):
    ## LOSS LIST
    l_depth = loss_for_depth(depth_pred_for_loss, depth_gt_for_loss)
    return l_depth

def get_1loss_1batch(batch_size, current_batch_size, index_iter, num_data, scores, l_depth):

    l_custom = torch.zeros(current_batch_size).cuda(torch.device("cuda:0"))
    for index_batch in range(current_batch_size):
        index_record = batch_size * index_iter + index_batch

        if index_batch == 0:
            loss = 0

        if index_record < num_data:
            loss_1batch = 0
            loss_1batch = loss_1batch + l_depth[index_batch]

            l_custom[index_batch] = loss_1batch
            loss = loss + loss_1batch / current_batch_size

            scores[index_record, :] = [
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                l_depth[index_batch].cpu().detach().numpy(),               0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
                0,                                  0,                    			  0,                                 0,                    				0,
            ]

    return loss, l_custom, scores