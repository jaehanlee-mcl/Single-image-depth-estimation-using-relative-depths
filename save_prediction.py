from PIL import Image
import numpy as np

def pred2png(data, data_path, index_data, data_type):
    if data_type == 'depth_indoor':
        data = (data / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'di'
    if data_type == 'depth_outdoor':
        data = (data / 100 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'do'
    if data_type == 'd0_depth_dx_left':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd0xl'
    if data_type == 'd0_depth_dx_right':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd0xr'
    if data_type == 'd0_depth_dy_top':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd0yt'
    if data_type == 'd0_depth_dy_down':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd0yd'
    if data_type == 'd0_ndepth_w5':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd0n5'
    if data_type == 'd0_ndepth_w17':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd0n17'
    if data_type == 'd1_depth_dx_left':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd1xl'
    if data_type == 'd1_depth_dx_right':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd1xr'
    if data_type == 'd1_depth_dy_top':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd1yt'
    if data_type == 'd1_depth_dy_down':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd1yd'
    if data_type == 'd1_ndepth_w5':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd1n5'
    if data_type == 'd1_ndepth_w17':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd1n17'
    if data_type == 'd2_depth_dx_left':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd2xl'
    if data_type == 'd2_depth_dx_right':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd2xr'
    if data_type == 'd2_depth_dy_top':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd2yt'
    if data_type == 'd2_depth_dy_down':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd2yd'
    if data_type == 'd2_ndepth_w5':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd2n5'
    if data_type == 'd2_ndepth_w17':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd2n17'
    if data_type == 'd3_depth_dx_left':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd3xl'
    if data_type == 'd3_depth_dx_right':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd3xr'
    if data_type == 'd3_depth_dy_top':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd3yt'
    if data_type == 'd3_depth_dy_down':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd3yd'
    if data_type == 'd3_ndepth_w5':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd3n5'
    if data_type == 'd3_ndepth_w17':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd3n17'
    if data_type == 'd4_depth_dx_left':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd4xl'
    if data_type == 'd4_depth_dx_right':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd4xr'
    if data_type == 'd4_depth_dy_top':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd4yt'
    if data_type == 'd4_depth_dy_down':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd4yd'
    if data_type == 'd4_ndepth_w5':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd4n5'
    if data_type == 'd4_ndepth_w17':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd4n17'
    if data_type == 'd5_depth_dx_left':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd5xl'
    if data_type == 'd5_depth_dx_right':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd5xr'
    if data_type == 'd5_depth_dy_top':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd5yt'
    if data_type == 'd5_depth_dy_down':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd5yd'
    if data_type == 'd5_ndepth_w5':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd5n5'
    if data_type == 'd5_ndepth_w17':
        data = ((data+5) / 10 * (pow(2,16)-1)).astype(np.uint16)
        data_suffix = 'd5n17'

    data_name = data_path + '/' + str(index_data).zfill(4) + '_' + data_suffix + '.png'
    data_image = Image.fromarray(data)
    data_image.save(data_name)