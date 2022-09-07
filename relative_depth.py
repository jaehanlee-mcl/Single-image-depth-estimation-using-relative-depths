import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

def depth_combination(depth_pred, depth_use=['valid', 'valid', 'valid', 'valid', 'invalid', 'invalid', 'ALS', 'ALS', 'ALS', 'prop', 'prop', 'invalid']):
    depth_components = []
    valid_components = []
    rel_depth = []
    ## 1. depth components setting
    for index_pred in range(len(depth_use)):
        if depth_use[index_pred] == 'valid':
            rel_depth.append([])
            depth_components.append(decompose_depth(depth_pred[index_pred]))
        elif depth_use[index_pred] == 'invalid':
            rel_depth.append([])
            depth_components.append([])
        elif depth_use[index_pred] == 'ALS':
            num_iter = 300 #(index_pred-5) * 100
            rel_depth.append(torch.log(relative_depth_estimation(torch.exp(depth_pred[index_pred]), partition_size=999, num_iter=num_iter)))
            depth_components.append(decompose_depth(rel_depth[index_pred]))
        elif depth_use[index_pred] == 'prop':
            rel_depth.append([])
            depth_components.append([])

        if depth_use[index_pred] == 'valid' or depth_use[index_pred] == 'ALS':
            if index_pred == 0:
                valid_components.append([1/1, 1/1, 0/1, 0/1, 0/1, 0/1, 0/1])
            elif index_pred == 1:
                valid_components.append([1/1, 1/1, 1/1, 0/1, 0/1, 0/1, 0/1])
            elif index_pred == 2:
                valid_components.append([1/1, 1/1, 1/1, 1/1, 0/1, 0/1, 0/1])
            elif index_pred == 3:
                valid_components.append([1/1, 1/1, 1/1, 1/1, 1/1, 0/1, 0/1])
            elif index_pred == 4:
                valid_components.append([1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 0/1])
            elif index_pred == 5:
                valid_components.append([1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1])
            elif index_pred == 6:
                valid_components.append([0/1, 1/1, 0/1, 0/1, 0/1, 0/1, 0/1])
            elif index_pred == 7:
                valid_components.append([0/1, 1/1, 1/1, 0/1, 0/1, 0/1, 0/1])
            elif index_pred == 8:
                valid_components.append([0/1, 1/1, 1/1, 1/1, 0/1, 0/1, 0/1])
            elif index_pred == 9:
                valid_components.append([0/1, 1/1, 1/1, 1/1, 1/1, 0/1, 0/1])
            elif index_pred == 10:
                valid_components.append([0/1, 1/1, 1/1, 1/1, 1/1, 1/1, 0/1])
            elif index_pred == 11:
                valid_components.append([0/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1])
        elif depth_use[index_pred] == 'invalid' or depth_use[index_pred] == 'prop':
            valid_components.append([0/1, 0/1, 0/1, 0/1, 0/1, 0/1, 0/1])
    ## 2. depth combination by scale
    up = []
    depth_combined_components = []
    for index_scale in range(7):
        num_combined = 0
        if index_scale == 0:
            depth_combined_components.append(torch.zeros(torch.mean(depth_pred[index_scale - 1], dim=[2,3], keepdim=True).size()))
        elif index_scale > 0:
            depth_combined_components.append(torch.zeros(depth_pred[index_scale-1].size()))

        for index_pred in range(len(depth_use)):
            if valid_components[index_pred][index_scale] > 0:
                if num_combined == 0:
                    num_combined += valid_components[index_pred][index_scale]
                    depth_combined_components[index_scale] = valid_components[index_pred][index_scale] * depth_components[index_pred][index_scale]
                elif num_combined > 0:
                    num_combined += valid_components[index_pred][index_scale]
                    depth_combined_components[index_scale] += valid_components[index_pred][index_scale] * depth_components[index_pred][index_scale]
        if num_combined > 1:
            depth_combined_components[index_scale] = depth_combined_components[index_scale] / num_combined
        if index_scale == 0:
            up.append([])
        else:
            up.append(torch.nn.Upsample(size=[9*pow(2,index_scale-1), 12*pow(2,index_scale-1)], mode='nearest'))
    ## 3. depth combination across scale
    depth_combined = []
    depth_combined.append(depth_combined_components[0])
    for index_scale in range(1,7):
        depth_combined_temp = up[index_scale](depth_combined[index_scale-1]) + depth_combined_components[index_scale]
        if depth_use[index_scale+5] == 'prop':
            num_iter = 1
            depth_combined.append(relative_depth_propagation(depth_combined_temp, depth_pred[index_scale+5], num_iter=num_iter))
        else:
            depth_combined.append(depth_combined_temp)

    return depth_combined[-1]







def relative_depth_propagation(input_depth_map, input_relative_depth, num_iter = 1, gpu_mode=True):
    batch_size, _, num_input_row, num_input_col = input_depth_map.size()
    batch_size, num_channel, num_output_row, num_output_col = input_relative_depth.size()
    upsample_depth_map_shifted = torch.zeros((batch_size, num_channel, num_output_row, num_output_col))

    if gpu_mode == True:
        input_depth_map = input_depth_map.half().cuda()
        input_relative_depth = input_relative_depth.half().cuda()
        upsample_depth_map_shifted = upsample_depth_map_shifted.half().cuda()

    upscale_ratio = round(num_output_row/num_input_row)
    upsample = torch.nn.Upsample(scale_factor=upscale_ratio, mode='nearest')

    if num_channel == 25:
        pad_size = 2
        index_rel_row = [
            -2, -2, -2, -2, -2,
            -1, -1, -1, -1, -1,
             0,  0,  0,  0,  0,
             1,  1,  1,  1,  1,
             2,  2,  2,  2,  2,
        ]
        index_rel_col = [
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
        ]

    upsample_depth_map = upsample(input_depth_map)
    for index_iter in range(num_iter):
        upsample_depth_map_with_pad = F.pad(upsample_depth_map, pad=[pad_size, pad_size, pad_size, pad_size], mode='replicate')

        for index_rel in range(num_channel):
            upsample_depth_map_shifted[:, index_rel:index_rel+1, :, :] \
                = upsample_depth_map_with_pad[
                    :,
                    :,
                    pad_size+index_rel_row[index_rel]:pad_size+index_rel_row[index_rel]+num_output_row,
                    pad_size+index_rel_col[index_rel]:pad_size+index_rel_col[index_rel]+num_output_col,
                ]

        propagated_depth_map = upsample_depth_map_shifted + input_relative_depth
        upsample_depth_map = torch.mean(propagated_depth_map, dim=1, keepdim=True)

    output_depth_map = upsample_depth_map
    if gpu_mode == True:
        output_depth_map = output_depth_map.cpu().float().detach()

    return output_depth_map

def relative_depth_estimation(input_tensor, partition_size=9, stride_size=3, num_iter=100, num_neighborhood=24, gpu_mode=True):
    batch_size, num_channel, num_row, num_col = input_tensor.size()

    time_start = time.time()
    # 1. partition size 지정
    partition_size_new = [min(num_row, partition_size), min(num_col, partition_size)]
    partition_size = partition_size_new

    # 2. partition 위치 지정
    loc_row, loc_col = pairwise_comparison_matrix_partition_location(num_row=num_row, num_col=num_col, partition_size=partition_size, stride_size=stride_size)
    num_partition = [len(loc_row), len(loc_col)]
    #print(time.time() - time_start)

    # 3. input tensor 분할
    input_tensor_partition = input_tensor2partition(input_tensor=input_tensor, partition_size=partition_size, num_partition=num_partition, loc_row=loc_row, loc_col=loc_col)
    #print(time.time() - time_start)

    # 4. pairwise_comparison_matrix 생성
    pairwise_comparison_matrix = pairwise_comparison_matrix_generation(input_tensor=input_tensor_partition, gpu_mode=gpu_mode)
    #print(time.time() - time_start)

    # 5. matrix completion
    #time_start = time.time()
    completed_matrix = matrix_completion_ALS(pairwise_comparison_matrix, partition_size=partition_size, num_iter=num_iter, gpu_mode=gpu_mode)
    #print('ALS: ' + str(time.time() - time_start))
    #print(time.time() - time_start)

    # 6. marging_partition
    output_tensor = merging_partition(completed_matrix, num_row=num_row, num_col=num_col, partition_size=partition_size, loc_row=loc_row, loc_col=loc_col)
    #print(time.time() - time_start)

    return output_tensor

def input_tensor2partition(input_tensor, partition_size=[9,9], num_partition=[1,2], loc_row=[0], loc_col=[0,3]):
    batch_size, num_channel, num_row, num_col = input_tensor.size()

    output_tensor = torch.zeros((batch_size, num_partition[0], num_partition[1], num_channel, partition_size[0], partition_size[1]))

    for index_row_partition in range(num_partition[0]):
        for index_col_partition in range(num_partition[1]):
            output_tensor[:, index_row_partition, index_col_partition, :, :, :] \
                = input_tensor[
                  :,
                  :,
                  loc_row[index_row_partition]:loc_row[index_row_partition]+partition_size[0],
                  loc_col[index_col_partition]:loc_col[index_col_partition]+partition_size[1]
                  ]
    return output_tensor

def pairwise_comparison_matrix_partition_location(num_row, num_col, partition_size=[9,9], stride_size=3):
    num_partition = [math.ceil((num_row-partition_size[0])/stride_size+1), math.ceil((num_col-partition_size[1])/stride_size+1)]
    loc_row = []
    for index_row in range(num_partition[0]):
        if index_row < num_partition[0]-1:
            loc_row.append(index_row * stride_size)
        elif index_row == num_partition[0]-1:
            loc_row.append(num_row-partition_size[0])
    loc_col = []
    for index_col in range(num_partition[1]):
        if index_col < num_partition[1]-1:
            loc_col.append(index_col * stride_size)
        elif index_col == num_partition[1]-1:
            loc_col.append(num_col-partition_size[1])
    return loc_row, loc_col

def merging_partition(input_tensor, num_row=9, num_col=12, partition_size=[9,9], loc_row=[0], loc_col=[0,3]):
    batch_size, num_row_partition, num_col_partition, num_channel, _, _ = input_tensor.size()
    output_tensor = torch.zeros((batch_size, num_channel, num_row, num_col))
    weight_tensor = torch.zeros((batch_size, num_channel, num_row, num_col))

    input_tensor = torch.log(input_tensor)
    for index_row_partition in range(num_row_partition):
        for index_col_partition in range(num_col_partition):
            output_tensor[
                :,
                :,
                loc_row[index_row_partition]:loc_row[index_row_partition]+partition_size[0],
                loc_col[index_col_partition]:loc_col[index_col_partition]+partition_size[1],
            ] \
                += input_tensor[
                :,
                index_row_partition,
                index_col_partition,
                :,
                :,
            ]
            weight_tensor[
            :,
            :,
            loc_row[index_row_partition]:loc_row[index_row_partition] + partition_size[0],
            loc_col[index_col_partition]:loc_col[index_col_partition] + partition_size[1],
            ] += 1
    output_tensor = output_tensor / weight_tensor
    output_tensor = output_tensor - torch.mean(output_tensor, dim=[2,3], keepdim=True)
    output_tensor = torch.exp(output_tensor)

    return output_tensor

def pairwise_comparison_matrix_generation(input_tensor, gpu_mode=True):
    batch_size, num_row_partition, num_col_partition, num_channel, num_row, num_col = input_tensor.size()
    output_tensor = torch.zeros((batch_size, num_row_partition, num_col_partition, 1, num_row*num_col, num_row*num_col))

    if gpu_mode == True:
        input_tensor = input_tensor.half().cuda()
        output_tensor = output_tensor.half().cuda()

    if num_channel == 4:
        index_rel_row = [0, 0, -1, 1]
        index_rel_col = [-1, 1, 0, 0]
    elif num_channel == 25:
        index_rel_row = [
            -2, -2, -2, -2, -2,
            -1, -1, -1, -1, -1,
             0,  0,  0,  0,  0,
             1,  1,  1,  1,  1,
             2,  2,  2,  2,  2,
        ]
        index_rel_col = [
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
            -2, -1,  0,  1,  2,
        ]

    for input_row in range(num_row):
        for input_col in range(num_col):
            output_row = (input_row) * num_col + (input_col) * 1
            for index_rel in range(len(index_rel_row)):
                if input_row >= -index_rel_row[index_rel] \
                        and input_col >= -index_rel_col[index_rel] \
                        and input_row < num_row-index_rel_row[index_rel] \
                        and input_col < num_col-index_rel_col[index_rel]:
                    output_col = (input_row+index_rel_row[index_rel]) * num_col + (input_col+index_rel_col[index_rel])*1
                    output_tensor[:, :, :, :, output_row, output_col] \
                        = input_tensor[:, :, :, index_rel:index_rel+1, input_row, input_col]
    if gpu_mode == True:
        output_tensor = output_tensor.cpu().float().detach()
    return output_tensor

def matrix_completion_ALS(input_tensor, partition_size=[9,9], num_iter=100, gpu_mode=True):
    batch_size, num_row_partition, num_col_partition, num_channel, num_row, num_col = input_tensor.size()
    output_shape = [batch_size, num_row_partition, num_col_partition, num_channel, partition_size[0], partition_size[1]]

    input_tensor_valid = (input_tensor > 0)
    vec_col = torch.ones((batch_size, num_row_partition, num_col_partition, num_channel, num_row, 1))  # initialize
    vec_row = torch.ones((batch_size, num_row_partition, num_col_partition, num_channel, 1, num_col))  # initialize

    if gpu_mode == True:
        input_tensor = input_tensor.half().cuda()
        input_tensor_valid = input_tensor_valid.half().cuda()
        vec_col = vec_col.half().cuda()  # initialize
        vec_row = vec_row.half().cuda()  # initialize

    for index_iter in range(num_iter):
        # step 1 : vec_col optimize
        expanded_vec_row = vec_row.repeat(1,1,1,1,num_row,1) * input_tensor_valid
        vec_col \
            = torch.sum(input_tensor * expanded_vec_row, dim=5, keepdim=True) \
              / torch.sum(expanded_vec_row * expanded_vec_row, dim=5, keepdim=True)
        # step 2 : vec_row optimize
        expanded_vec_col = vec_col.repeat(1,1,1,1,1,num_col) * input_tensor_valid
        vec_row \
            = torch.sum(input_tensor * expanded_vec_col, dim=4, keepdim=True) \
              / torch.sum(expanded_vec_col * expanded_vec_col, dim=4, keepdim=True)
    depth_from_row = torch.log(vec_row.reshape(output_shape)) - torch.mean(torch.log(vec_row), dim=[4,5], keepdim=True)
    depth_from_col = torch.log(vec_col.reshape(output_shape)) - torch.mean(torch.log(vec_col), dim=[4,5], keepdim=True)
    depth_combined = torch.exp(-0.5 * depth_from_row + 0.5 * depth_from_col)

    if gpu_mode == True:
        output_tensor = depth_combined.cpu().float().detach()
    elif gpu_mode == False:
        output_tensor = depth_combined

    return output_tensor

def decompose_depth(input_tensor):
    batch_size, num_channel, num_row, num_col = input_tensor.size()
    up = []
    down = []
    sampled_tensor = []
    for index_upsample in range(7):
        if index_upsample == 0:
            up.append([])
            down.append([])
            sampled_tensor.append(torch.mean(input_tensor, dim=(2,3), keepdim=True))
        else:
            up.append(torch.nn.Upsample(size=[9*pow(2,index_upsample-1), 12*pow(2,index_upsample-1)], mode='nearest'))
            down.append(torch.nn.Upsample(size=[9*pow(2,index_upsample-1), 12*pow(2,index_upsample-1)], mode='bilinear'))
            if num_row == 9*pow(2,index_upsample-1):
                sampled_tensor.append(input_tensor)
            elif num_row < 9*pow(2,index_upsample-1):
                sampled_tensor.append(up[index_upsample](input_tensor))
            elif num_row > 9*pow(2,index_upsample-1):
                sampled_tensor.append(down[index_upsample](input_tensor))
    output_tensor = []
    output_tensor.append(sampled_tensor[0])
    for index_upsample in range(1,7):
        output_tensor.append(sampled_tensor[index_upsample] - up[index_upsample](sampled_tensor[index_upsample-1]))

    return output_tensor


def combine_depth(depth_components):
    up = []
    up.append([])
    combined_depth = depth_components[0]
    for index_upsample in range(1,8):
        up.append(torch.nn.Upsample(size=[pow(2,index_upsample), pow(2,index_upsample)], mode='nearest'))
        temp_combined_depth = up[index_upsample](combined_depth) + depth_components[index_upsample]
        combined_depth = temp_combined_depth
    return combined_depth

def combine_depth_with_setting(decomposed_depth, setting):
    depth_components = []
    if setting == "D3":
        weights = np.array(\
            [
                [1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
            ])
    if setting == "D4":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
            ])
    if setting == "D5":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
            ])
    if setting == "D6":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
            ])
    if setting == "D7":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1,  1/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
            ])
    if setting == "comb1":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/1,  1/1,  1/1,  1/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  1/1,  1/1,  1/1,  1/1 ],
            ])
    if setting == "comb2":
        weights = np.array(\
            [
                [1/5,  1/7,  1/8,  1/9,  0/1,  0/1,  0/1,  0/1 ],
                [1/5,  1/7,  1/8,  1/9,  1/8,  0/1,  0/1,  0/1 ],
                [1/5,  1/7,  1/8,  1/9,  1/8,  1/6,  0/1,  0/1 ],
                [1/5,  1/7,  1/8,  1/9,  1/8,  1/6,  1/4,  0/1 ],
                [1/5,  1/7,  1/8,  1/9,  1/8,  1/6,  1/4,  1/2 ],
                [0/1,  1/7,  1/8,  1/9,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  1/7,  1/8,  1/9,  1/8,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  1/8,  1/9,  1/8,  1/6,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  1/9,  1/8,  1/6,  1/4,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  1/8,  1/6,  1/4,  1/2 ],
            ])
    if setting == "comb3":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/3,  1/3,  1/4,  1/5,  1/6,  1/6,  0/1,  0/1 ],
                [1/3,  1/3,  1/4,  1/5,  1/6,  1/6,  1/4,  0/1 ],
                [1/3,  1/3,  1/4,  1/5,  1/6,  1/6,  1/4,  1/2 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  1/4,  1/5,  1/6,  1/6,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  1/5,  1/6,  1/6,  1/4,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  1/6,  1/6,  1/4,  1/2 ],
            ])
    if setting == "comb4":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/2,  1/2,  1/2,  1/3,  1/4,  1/4,  1/4,  0/1 ],
                [1/2,  1/2,  1/2,  1/3,  1/4,  1/4,  1/4,  1/2 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  1/3,  1/4,  1/4,  1/4,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  1/4,  1/4,  1/2 ],
            ])
    if setting == "comb5":
        weights = np.array(\
            [
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [1/1,  1/1,  1/1,  1/1,  1/2,  1/2,  1/2,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1,  0/1 ],
                [0/1,  0/1,  0/1,  0/1,  1/2,  1/2,  1/2,  1/1 ],
            ])
    for index_scale in range(np.shape(weights)[1]):
        for index_decoder in range(np.shape(weights)[0]):
            if index_decoder == 0:
                temp_component = decomposed_depth[index_decoder][index_scale] * weights[index_decoder][index_scale]
            else:
                temp_component = temp_component + decomposed_depth[index_decoder][index_scale] * weights[index_decoder][index_scale]
        depth_components.append(temp_component)
    combined_depth = combine_depth(depth_components)
    return combined_depth