import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import networks.pnasnet, networks.nasnet
from collections import OrderedDict

def create_model(model_name, decoder_scale=1024, decoder_resolution=0, num_neighborhood=24):
    # Create model
    if model_name == 'DenseNet161':
        model = Model_DenseNet161(decoder_scale).cuda()
    if model_name == 'DenseNet161_OrdinaryRelativeDepth':
        model = Model_DenseNet161_OrdinaryRelativeDepth(decoder_scale, num_neighborhood=num_neighborhood).cuda()
    if model_name == 'PNASNet5LargeMin':
        model = Model_PNASNet5Large_min(decoder_scale).cuda()
    return model

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.reluA = nn.ReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.reluB = nn.ReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.reluB( self.convB( self.reluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )

class UpSample_simple(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample_simple, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=1, stride=1, padding=0)
        self.reluA = nn.ReLU()
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.reluB = nn.ReLU()

    def forward(self, x):
        up_x = F.interpolate(x, size=[x.size(2)*2, x.size(3)*2], mode='bilinear', align_corners=True)
        return self.reluB( self.convB( self.reluA(self.convA( up_x ) ) )  )

class SameSample_simple(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(SameSample_simple, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=1, stride=1, padding=0)
        self.reluA = nn.ReLU()
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.reluB = nn.ReLU()

    def forward(self, x):
        return self.reluB( self.convB( self.reluA(self.convA( x ) ) )  )

class Decoder_DenseNet161(nn.Module):
    def __init__(self, decoder_scale = 512, num_upsample = 5, num_channels_final = 1):
        super(Decoder_DenseNet161, self).__init__()

        num_channels_in = 2208
        num_channels_out = decoder_scale
        self.features = nn.Sequential(OrderedDict([
            ('decoder_conv0', nn.Conv2d(num_channels_in, num_channels_out, kernel_size=1, stride=1)),
        ]))

        for index_upsample in range(5):
            if num_upsample < index_upsample+1:
                num_channels_in = num_channels_out
                num_channels_out = num_channels_in
                block = SameSample_simple(skip_input=num_channels_in, output_features=num_channels_out)
                self.features.add_module('decoder_same%d' % (index_upsample+1), block)
            else:
                num_channels_in = num_channels_out
                num_channels_out = num_channels_in // 2
                block = UpSample_simple(skip_input=num_channels_in, output_features=num_channels_out)
                self.features.add_module('decoder_up%d' % (index_upsample+1), block)

        num_channels_in = num_channels_out
        num_channels_out = num_channels_final
        block = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=1, stride=1)
        self.features.add_module('decoder_conv_final', block)

    def forward(self, features_encoder):
        features_decoder = self.features(features_encoder)
        return features_decoder

class Encoder_DenseNet161(nn.Module):
    def __init__(self):
        super(Encoder_DenseNet161, self).__init__()
        import torchvision.models as models
        self.original_model = models.densenet161( pretrained=True )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class Model_DenseNet161(nn.Module):
    def __init__(self, decoder_scale = 512):
        super(Model_DenseNet161, self).__init__()
        self.encoder = Encoder_DenseNet161()
        self.decoder = Decoder_DenseNet161(decoder_scale=decoder_scale)

    def forward(self, x):
        features_encoder = self.encoder(x)
        return self.decoder(features_encoder[11])

class Model_DenseNet161_OrdinaryRelativeDepth(nn.Module):
    def __init__(self, decoder_scale = 512, num_neighborhood = 24):
        super(Model_DenseNet161_OrdinaryRelativeDepth, self).__init__()
        self.encoder = Encoder_DenseNet161()
        self.decoder_D3 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=0, num_channels_final=1)
        self.decoder_D4 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=1, num_channels_final=1)
        self.decoder_D5 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=2, num_channels_final=1)
        self.decoder_D6 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=3, num_channels_final=1)
        self.decoder_D7 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=4, num_channels_final=1)
        self.decoder_D8 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=5, num_channels_final=1)
        self.decoder_R3 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=0, num_channels_final=num_neighborhood+1)
        self.decoder_R4 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=1, num_channels_final=num_neighborhood+1)
        self.decoder_R5 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=2, num_channels_final=num_neighborhood+1)
        self.decoder_R6 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=3, num_channels_final=num_neighborhood+1)
        self.decoder_R7 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=4, num_channels_final=num_neighborhood+1)
        self.decoder_R8 = Decoder_DenseNet161(decoder_scale=decoder_scale, num_upsample=5, num_channels_final=num_neighborhood+1)

    def forward(self, x, get_pred_name=False):
        features_encoder = self.encoder(x)
        pred = []
        pred.append(self.decoder_D3(features_encoder[11]))
        pred.append(self.decoder_D4(features_encoder[11]))
        pred.append(self.decoder_D5(features_encoder[11]))
        pred.append(self.decoder_D6(features_encoder[11]))
        pred.append(self.decoder_D7(features_encoder[11]))
        pred.append(self.decoder_D8(features_encoder[11]))
        pred.append(self.decoder_R3(features_encoder[11]))
        pred.append(self.decoder_R4(features_encoder[11]))
        pred.append(self.decoder_R5(features_encoder[11]))
        pred.append(self.decoder_R6(features_encoder[11]))
        pred.append(self.decoder_R7(features_encoder[11]))
        pred.append(self.decoder_R8(features_encoder[11]))

        if get_pred_name == False:
            return pred
        elif get_pred_name == True:
            pred_name = ['D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
            return pred, pred_name

class Model_PNASNet5Large_min(nn.Module):
    def __init__(self, decoder_scale = 1024):
        super(Model_PNASNet5Large_min, self).__init__()
        self.encoder = networks.pnasnet.pnasnet5large(num_classes=1000, pretrained='imagenet')

        num_channels_d32_in = 4320

        num_channels_d32_out = decoder_scale

        self.conv_d32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1 = UpSample_simple(skip_input=num_channels_d32_out // 1, output_features=num_channels_d32_out // 2)
        self.up2 = UpSample_simple(skip_input=num_channels_d32_out // 2, output_features=num_channels_d32_out // 4)
        self.up3 = UpSample_simple(skip_input=num_channels_d32_out // 4, output_features=num_channels_d32_out // 8)
        self.up4 = UpSample_simple(skip_input=num_channels_d32_out // 8, output_features=num_channels_d32_out // 16)
        self.up5 = UpSample_simple(skip_input=num_channels_d32_out // 16, output_features=num_channels_d32_out // 32)

        self.conv3 = nn.Conv2d(num_channels_d32_out // 32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_d32 = self.encoder.get_features_min(x)

        reduced_d32 = self.conv_d32(x_d32)

        decoder_scale_d16 = self.up1(reduced_d32)
        decoder_scale_d08 = self.up2(decoder_scale_d16)
        decoder_scale_d04 = self.up3(decoder_scale_d08)
        decoder_scale_d02 = self.up4(decoder_scale_d04)
        decoder_scale_d01 = self.up5(decoder_scale_d02)
        output = self.conv3(decoder_scale_d01)
        return output

class Model_PNASNet5Large_integ(nn.Module):
    def __init__(self, decoder_scale = 1024, decoder_resolution = 0):
        super(Model_PNASNet5Large_integ, self).__init__()
        self.encoder = networks.pnasnet.pnasnet5large(num_classes=1000, pretrained='imagenet')

        self.decoder_resolution = decoder_resolution # 0: origianl input, n: original input / (2^n)

        if self.decoder_resolution <= 0:
            self.decoder_depth_dx = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=5)
            self.decoder_depth_dy = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=5)
            self.decoder_ndepth_w5 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=5)
            self.decoder_ndepth_w17 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=5)

        if self.decoder_resolution <= 1:
            self.decoder_d1_depth_dx = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=4)
            self.decoder_d1_depth_dy = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=4)
            self.decoder_d1_ndepth_w5 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=4)
            self.decoder_d1_ndepth_w17 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=4)

        if self.decoder_resolution <= 2:
            self.decoder_d2_depth_dx = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=3)
            self.decoder_d2_depth_dy = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=3)
            self.decoder_d2_ndepth_w5 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=3)
            self.decoder_d2_ndepth_w17 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=3)

        if self.decoder_resolution <= 3:
            self.decoder_d3_depth_dx = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=2)
            self.decoder_d3_depth_dy = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=2)
            self.decoder_d3_ndepth_w5 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=2)
            self.decoder_d3_ndepth_w17 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=2)

        if self.decoder_resolution <= 4:
            self.decoder_d4_depth_dx = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=1)
            self.decoder_d4_depth_dy = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=1)
            self.decoder_d4_ndepth_w5 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=1)
            self.decoder_d4_ndepth_w17 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=1)

        if self.decoder_resolution <= 5:
            self.decoder_d5_depth_dx = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=0)
            self.decoder_d5_depth_dy = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=2, num_scale=0)
            self.decoder_d5_ndepth_w5 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=0)
            self.decoder_d5_ndepth_w17 = Decoder_PNASNet5Large_integ(num_encoder_out=4320, num_decoder_in=decoder_scale, num_decoder_out=1, num_scale=0)

    def forward(self, x):
        x_d32 = self.encoder.get_features_min(x)

        if self.decoder_resolution <= 0:
            depth_dx = self.decoder_depth_dx(x_d32)
            depth_dy = self.decoder_depth_dy(x_d32)
            ndepth_w5 = self.decoder_ndepth_w5(x_d32)
            ndepth_w17 = self.decoder_ndepth_w17(x_d32)
            d0_depth = torch.cat((depth_dx, depth_dy, ndepth_w5, ndepth_w17), dim=1)

        if self.decoder_resolution <= 1:
            d1_depth_dx = self.decoder_d1_depth_dx(x_d32)
            d1_depth_dy = self.decoder_d1_depth_dy(x_d32)
            d1_ndepth_w5 = self.decoder_d1_ndepth_w5(x_d32)
            d1_ndepth_w17 = self.decoder_d1_ndepth_w17(x_d32)
            d1_depth = torch.cat((d1_depth_dx, d1_depth_dy, d1_ndepth_w5, d1_ndepth_w17), dim=1)

        if self.decoder_resolution <= 2:
            d2_depth_dx = self.decoder_d2_depth_dx(x_d32)
            d2_depth_dy = self.decoder_d2_depth_dy(x_d32)
            d2_ndepth_w5 = self.decoder_d2_ndepth_w5(x_d32)
            d2_ndepth_w17 = self.decoder_d2_ndepth_w17(x_d32)
            d2_depth = torch.cat((d2_depth_dx, d2_depth_dy, d2_ndepth_w5, d2_ndepth_w17), dim=1)

        if self.decoder_resolution <= 3:
            d3_depth_dx = self.decoder_d3_depth_dx(x_d32)
            d3_depth_dy = self.decoder_d3_depth_dy(x_d32)
            d3_ndepth_w5 = self.decoder_d3_ndepth_w5(x_d32)
            d3_ndepth_w17 = self.decoder_d3_ndepth_w17(x_d32)
            d3_depth = torch.cat((d3_depth_dx, d3_depth_dy, d3_ndepth_w5, d3_ndepth_w17), dim=1)

        if self.decoder_resolution <= 4:
            d4_depth_dx = self.decoder_d4_depth_dx(x_d32)
            d4_depth_dy = self.decoder_d4_depth_dy(x_d32)
            d4_ndepth_w5 = self.decoder_d4_ndepth_w5(x_d32)
            d4_ndepth_w17 = self.decoder_d4_ndepth_w17(x_d32)
            d4_depth = torch.cat((d4_depth_dx, d4_depth_dy, d4_ndepth_w5, d4_ndepth_w17), dim=1)

        if self.decoder_resolution <= 5:
            d5_depth_dx = self.decoder_d5_depth_dx(x_d32)
            d5_depth_dy = self.decoder_d5_depth_dy(x_d32)
            d5_ndepth_w5 = self.decoder_d5_ndepth_w5(x_d32)
            d5_ndepth_w17 = self.decoder_d5_ndepth_w17(x_d32)
            d5_depth = torch.cat((d5_depth_dx, d5_depth_dy, d5_ndepth_w5, d5_ndepth_w17), dim=1)

        if self.decoder_resolution <= 0:
            return d0_depth, d1_depth, d2_depth, d3_depth, d4_depth, d5_depth
        if self.decoder_resolution <= 1:
            return False, d1_depth, d2_depth, d3_depth, d4_depth, d5_depth
        if self.decoder_resolution <= 2:
            return False, False, d2_depth, d3_depth, d4_depth, d5_depth
        if self.decoder_resolution <= 3:
            return False, False, False, d3_depth, d4_depth, d5_depth
        if self.decoder_resolution <= 4:
            return False, False, False, False, d4_depth, d5_depth
        if self.decoder_resolution <= 5:
            return False, False, False, False, False, d5_depth

class Decoder_PNASNet5Large_integ(nn.Module):
    def __init__(self, num_encoder_out=4320, num_decoder_in=1024, num_decoder_out=1, num_scale=5):
        super(Decoder_PNASNet5Large_integ, self).__init__()

        num_channels_d32_in = num_encoder_out
        num_channels_d32_out = num_decoder_in

        self.conv_d32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        if num_scale == 5:
            self.up1 = UpSample_simple(skip_input=num_channels_d32_out // 1, output_features=num_channels_d32_out // 2)
            self.up2 = UpSample_simple(skip_input=num_channels_d32_out // 2, output_features=num_channels_d32_out // 4)
            self.up3 = UpSample_simple(skip_input=num_channels_d32_out // 4, output_features=num_channels_d32_out // 8)
            self.up4 = UpSample_simple(skip_input=num_channels_d32_out // 8, output_features=num_channels_d32_out // 16)
            self.up5 = UpSample_simple(skip_input=num_channels_d32_out // 16, output_features=num_channels_d32_out // 32)
        if num_scale == 4:
            self.up1 = UpSample_simple(skip_input=num_channels_d32_out // 1, output_features=num_channels_d32_out // 2)
            self.up2 = UpSample_simple(skip_input=num_channels_d32_out // 2, output_features=num_channels_d32_out // 4)
            self.up3 = UpSample_simple(skip_input=num_channels_d32_out // 4, output_features=num_channels_d32_out // 8)
            self.up4 = UpSample_simple(skip_input=num_channels_d32_out // 8, output_features=num_channels_d32_out // 16)
            self.up5 = SameSample_simple(skip_input=num_channels_d32_out // 16, output_features=num_channels_d32_out // 32)
        if num_scale == 3:
            self.up1 = UpSample_simple(skip_input=num_channels_d32_out // 1, output_features=num_channels_d32_out // 2)
            self.up2 = UpSample_simple(skip_input=num_channels_d32_out // 2, output_features=num_channels_d32_out // 4)
            self.up3 = UpSample_simple(skip_input=num_channels_d32_out // 4, output_features=num_channels_d32_out // 8)
            self.up4 = SameSample_simple(skip_input=num_channels_d32_out // 8, output_features=num_channels_d32_out // 16)
            self.up5 = SameSample_simple(skip_input=num_channels_d32_out // 16, output_features=num_channels_d32_out // 32)
        if num_scale == 2:
            self.up1 = UpSample_simple(skip_input=num_channels_d32_out // 1, output_features=num_channels_d32_out // 2)
            self.up2 = UpSample_simple(skip_input=num_channels_d32_out // 2, output_features=num_channels_d32_out // 4)
            self.up3 = SameSample_simple(skip_input=num_channels_d32_out // 4, output_features=num_channels_d32_out // 8)
            self.up4 = SameSample_simple(skip_input=num_channels_d32_out // 8, output_features=num_channels_d32_out // 16)
            self.up5 = SameSample_simple(skip_input=num_channels_d32_out // 16, output_features=num_channels_d32_out // 32)
        if num_scale == 1:
            self.up1 = UpSample_simple(skip_input=num_channels_d32_out // 1, output_features=num_channels_d32_out // 2)
            self.up2 = SameSample_simple(skip_input=num_channels_d32_out // 2, output_features=num_channels_d32_out // 4)
            self.up3 = SameSample_simple(skip_input=num_channels_d32_out // 4, output_features=num_channels_d32_out // 8)
            self.up4 = SameSample_simple(skip_input=num_channels_d32_out // 8, output_features=num_channels_d32_out // 16)
            self.up5 = SameSample_simple(skip_input=num_channels_d32_out // 16, output_features=num_channels_d32_out // 32)
        if num_scale == 0:
            self.up1 = SameSample_simple(skip_input=num_channels_d32_out // 1, output_features=num_channels_d32_out // 2)
            self.up2 = SameSample_simple(skip_input=num_channels_d32_out // 2, output_features=num_channels_d32_out // 4)
            self.up3 = SameSample_simple(skip_input=num_channels_d32_out // 4, output_features=num_channels_d32_out // 8)
            self.up4 = SameSample_simple(skip_input=num_channels_d32_out // 8, output_features=num_channels_d32_out // 16)
            self.up5 = SameSample_simple(skip_input=num_channels_d32_out // 16, output_features=num_channels_d32_out // 32)

        self.conv3 = nn.Conv2d(num_channels_d32_out // 32, num_decoder_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        reduced_d32 = self.conv_d32(x)

        decoder_scale_d16 = self.up1(reduced_d32)
        decoder_scale_d08 = self.up2(decoder_scale_d16)
        decoder_scale_d04 = self.up3(decoder_scale_d08)
        decoder_scale_d02 = self.up4(decoder_scale_d04)
        decoder_scale_d01 = self.up5(decoder_scale_d02)
        output = self.conv3(decoder_scale_d01)
        return output