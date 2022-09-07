import datetime
import os

def make_model_path(model_name, decoder_scale, batch_size, decoder_resolution=0, num_neighborhood=24):
    now = datetime.datetime.now()
    model_path_backbone = ''
    if model_name == 'DenseNet161':
        model_path_backbone = 'D161_b' + str(batch_size).zfill(2) + '_scale' + str(decoder_scale).zfill(4)
    if model_name == 'DenseNet161_OrdinaryRelativeDepth':
        model_path_backbone = 'D161_ORD_b' + str(batch_size).zfill(2) + '_scale' + str(decoder_scale).zfill(4) + '_nb' + str(num_neighborhood).zfill(2)
    if model_name == 'PNASNet5LargeMin':
        model_path_backbone = 'P5L_MIN_b' + str(batch_size).zfill(2) + '_scale' + str(decoder_scale).zfill(4)
    if model_name == 'PNASNet5LargeInteg':
        model_path_backbone = 'P5L_Integ_b' + str(batch_size).zfill(2) + '_scale' + str(decoder_scale).zfill(4) + '_resol' + str(decoder_resolution)

    model_path_data = 'data-' + str(now.year).zfill(4) + str(now.month).zfill(2) + str(now.day).zfill(2) + str(
        now.hour).zfill(2) + str(now.minute).zfill(2)
    model_path = 'runs/' + model_path_backbone + '/' + model_path_data
    if not os.path.isdir('runs'):
        os.mkdir('runs')
    if not os.path.isdir('runs/' + model_path_backbone):
        os.mkdir('runs/' + model_path_backbone)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    return model_path