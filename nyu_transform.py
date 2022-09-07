import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageStat
import matplotlib.pyplot as plt
import collections
import torch.nn.functional as F
import torch.nn as nn

try:
    import accimage
except ImportError:
    accimage = None
import random
import scipy.ndimage as ndimage

import pdb


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        mean_depth = round(ImageStat.Stat(depth).mean[0] * 257)

        applied_angle = random.uniform(-self.angle, self.angle)

        image = image.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(255,255,255))
        depth = depth.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(mean_depth))

        return {'image': image, 'depth': depth}


class RandomZoom(object):
    """Random zoom of the image from ratio (in zoom-ratio)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    ratio: max zoom-ratio of zoom
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, ratio=1):
        self.ratio = ratio

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        w1, h1 = image.size

        applied_zoom = random.uniform(1, self.ratio)
        w2 = round(w1 * applied_zoom)
        h2 = round(h1 * applied_zoom)

        image = image.resize((w2, h2), Image.BICUBIC)
        depth = depth.resize((w2, h2), Image.BICUBIC)
        enhancer = ImageEnhance.Brightness(depth)
        depth = enhancer.enhance(1/applied_zoom)

        image, depth = self.randomCrop(image, depth, (w1, h1))

        return {'image': image, 'depth': depth}

    def randomCrop(self, image, depth, size):
        w1, h1 = size
        w2, h2 = image.size

        if w1 == w2 and h1 == h2:
            return image, depth

        x = round(random.uniform(0, w2 - w1) - 0.5)
        y = round(random.uniform(0, h2 - h1) - 0.5)

        image = image.crop((x, y, x + w1, y + h1))
        depth = depth.crop((x, y, x + w1, y + h1))

        return image, depth

class RandomHorizontalFlip(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.changeScale(image, self.size)
        depth = self.changeScale(depth, self.size, Image.NEAREST)

        return {'image': image, 'depth': depth}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)


class CenterCrop(object):
    def __init__(self, size_image, size_depth):
        self.size_image = size_image
        self.size_depth = size_depth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        ### crop image and depth to (304, 228)
        image = self.centerCrop(image, self.size_image)
        depth = self.centerCrop(depth, self.size_image)
        ### resize depth to (152, 114) downsample 2
        ow, oh = self.size_depth
        depth = depth.resize((ow, oh))

        return {'image': image, 'depth': depth}

    def centerCrop(self, image, size):
        w1, h1 = image.size

        tw, th = size

        if w1 == tw and h1 == th:
            return image
        ## (320-304) / 2. = 8
        ## (240-228) / 2. = 8
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image

class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if self.alphastd == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return {'image': image, 'depth': depth}


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)

        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if self.transforms is None:
            return {'image': image, 'depth': depth}
        order = torch.randperm(len(self.transforms))
        for i in order:
            image = self.transforms[i](image)

        return {'image': image, 'depth': depth}


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image, depth = sample['image'], sample['depth']

        image = self.normalize(image, self.mean, self.std)

        return {'image': image, 'depth': depth}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

class LogDepth(object):

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor depth of size (1, H, W) to be converted.
        Returns:
            Tensor: log-converted depth
        """
        image, depth = sample['image'], sample['depth']

        depth = torch.max(depth, torch.tensor(0.0001))
        depth = torch.log(depth)

        return {'image': image, 'depth': depth}

class ExtendedDepth(object):
    def __init__(self, decoder_resolution = 0):
        # interpolation function
        self.decoder_resolution = decoder_resolution
        self.interpolate_bicubic_div02 = nn.Upsample(scale_factor=1 / 2, mode='bicubic')
        self.interpolate_bicubic_div04 = nn.Upsample(scale_factor=1 / 4, mode='bicubic')
        self.interpolate_bicubic_div08 = nn.Upsample(scale_factor=1 / 8, mode='bicubic')
        self.interpolate_bicubic_div16 = nn.Upsample(scale_factor=1 / 16, mode='bicubic')
        self.interpolate_bicubic_div32 = nn.Upsample(scale_factor=1 / 32, mode='bicubic')
        self.kernel_dx_left = torch.tensor(np.array([[-1,  1]])).float().unsqueeze(0).unsqueeze(0)
        self.kernel_dx_right = torch.tensor(np.array([[ 1, -1]])).float().unsqueeze(0).unsqueeze(0)
        self.kernel_dy_top = torch.tensor(np.array([[-1], [ 1]])).float().unsqueeze(0).unsqueeze(0)
        self.kernel_dy_bottom = torch.tensor(np.array([[ 1], [-1]])).float().unsqueeze(0).unsqueeze(0)
        self.kernel_mean_w5 = torch.tensor(np.ones((5,5))/(5*5)).float().unsqueeze(0).unsqueeze(0)
        self.kernel_mean_w17 = torch.tensor(np.ones((17,17))/(17*17)).float().unsqueeze(0).unsqueeze(0)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        d0_depth = depth.unsqueeze(0)
        d1_depth = self.interpolate_bicubic_div02(d0_depth)
        d2_depth = self.interpolate_bicubic_div04(d0_depth)
        d3_depth = self.interpolate_bicubic_div08(d0_depth)
        d4_depth = self.interpolate_bicubic_div16(d0_depth)
        d5_depth = self.interpolate_bicubic_div32(d0_depth)

        if self.decoder_resolution <= 0:
            depth_dx, depth_dy, ndepth_w5, ndepth_w17 = self.get_relative_depth(d0_depth)
        if self.decoder_resolution <= 1:
            d1_depth_dx, d1_depth_dy, d1_ndepth_w5, d1_ndepth_w17 = self.get_relative_depth(d1_depth)
        if self.decoder_resolution <= 2:
            d2_depth_dx, d2_depth_dy, d2_ndepth_w5, d2_ndepth_w17 = self.get_relative_depth(d2_depth)
        if self.decoder_resolution <= 3:
            d3_depth_dx, d3_depth_dy, d3_ndepth_w5, d3_ndepth_w17 = self.get_relative_depth(d3_depth)
        if self.decoder_resolution <= 4:
            d4_depth_dx, d4_depth_dy, d4_ndepth_w5, d4_ndepth_w17 = self.get_relative_depth(d4_depth)
        if self.decoder_resolution <= 5:
            d5_depth_dx, d5_depth_dy, d5_ndepth_w5, d5_ndepth_w17 = self.get_relative_depth(d5_depth)

        if self.decoder_resolution <= 0:
            return {'image': image,
                    'depth': depth, 'depth_dx': depth_dx, 'depth_dy': depth_dy, 'ndepth_w5': ndepth_w5, 'ndepth_w17': ndepth_w17,
                    'd1_depth': d1_depth, 'd1_depth_dx': d1_depth_dx, 'd1_depth_dy': d1_depth_dy, 'd1_ndepth_w5': d1_ndepth_w5, 'd1_ndepth_w17': d1_ndepth_w17,
                    'd2_depth': d2_depth, 'd2_depth_dx': d2_depth_dx, 'd2_depth_dy': d2_depth_dy, 'd2_ndepth_w5': d2_ndepth_w5, 'd2_ndepth_w17': d2_ndepth_w17,
                    'd3_depth': d3_depth, 'd3_depth_dx': d3_depth_dx, 'd3_depth_dy': d3_depth_dy, 'd3_ndepth_w5': d3_ndepth_w5, 'd3_ndepth_w17': d3_ndepth_w17,
                    'd4_depth': d4_depth, 'd4_depth_dx': d4_depth_dx, 'd4_depth_dy': d4_depth_dy, 'd4_ndepth_w5': d4_ndepth_w5, 'd4_ndepth_w17': d4_ndepth_w17,
                    'd5_depth': d5_depth, 'd5_depth_dx': d5_depth_dx, 'd5_depth_dy': d5_depth_dy, 'd5_ndepth_w5': d5_ndepth_w5, 'd5_ndepth_w17': d5_ndepth_w17
                }
        if self.decoder_resolution <= 1:
            return {'image': image,
                    'depth': depth,
                    'd1_depth': d1_depth, 'd1_depth_dx': d1_depth_dx, 'd1_depth_dy': d1_depth_dy, 'd1_ndepth_w5': d1_ndepth_w5, 'd1_ndepth_w17': d1_ndepth_w17,
                    'd2_depth': d2_depth, 'd2_depth_dx': d2_depth_dx, 'd2_depth_dy': d2_depth_dy, 'd2_ndepth_w5': d2_ndepth_w5, 'd2_ndepth_w17': d2_ndepth_w17,
                    'd3_depth': d3_depth, 'd3_depth_dx': d3_depth_dx, 'd3_depth_dy': d3_depth_dy, 'd3_ndepth_w5': d3_ndepth_w5, 'd3_ndepth_w17': d3_ndepth_w17,
                    'd4_depth': d4_depth, 'd4_depth_dx': d4_depth_dx, 'd4_depth_dy': d4_depth_dy, 'd4_ndepth_w5': d4_ndepth_w5, 'd4_ndepth_w17': d4_ndepth_w17,
                    'd5_depth': d5_depth, 'd5_depth_dx': d5_depth_dx, 'd5_depth_dy': d5_depth_dy, 'd5_ndepth_w5': d5_ndepth_w5, 'd5_ndepth_w17': d5_ndepth_w17
                }
        if self.decoder_resolution <= 2:
            return {'image': image,
                    'depth': depth,
                    'd1_depth': d1_depth,
                    'd2_depth': d2_depth, 'd2_depth_dx': d2_depth_dx, 'd2_depth_dy': d2_depth_dy, 'd2_ndepth_w5': d2_ndepth_w5, 'd2_ndepth_w17': d2_ndepth_w17,
                    'd3_depth': d3_depth, 'd3_depth_dx': d3_depth_dx, 'd3_depth_dy': d3_depth_dy, 'd3_ndepth_w5': d3_ndepth_w5, 'd3_ndepth_w17': d3_ndepth_w17,
                    'd4_depth': d4_depth, 'd4_depth_dx': d4_depth_dx, 'd4_depth_dy': d4_depth_dy, 'd4_ndepth_w5': d4_ndepth_w5, 'd4_ndepth_w17': d4_ndepth_w17,
                    'd5_depth': d5_depth, 'd5_depth_dx': d5_depth_dx, 'd5_depth_dy': d5_depth_dy, 'd5_ndepth_w5': d5_ndepth_w5, 'd5_ndepth_w17': d5_ndepth_w17
                }
        if self.decoder_resolution <= 3:
            return {'image': image,
                    'depth': depth,
                    'd1_depth': d1_depth,
                    'd2_depth': d2_depth,
                    'd3_depth': d3_depth, 'd3_depth_dx': d3_depth_dx, 'd3_depth_dy': d3_depth_dy, 'd3_ndepth_w5': d3_ndepth_w5, 'd3_ndepth_w17': d3_ndepth_w17,
                    'd4_depth': d4_depth, 'd4_depth_dx': d4_depth_dx, 'd4_depth_dy': d4_depth_dy, 'd4_ndepth_w5': d4_ndepth_w5, 'd4_ndepth_w17': d4_ndepth_w17,
                    'd5_depth': d5_depth, 'd5_depth_dx': d5_depth_dx, 'd5_depth_dy': d5_depth_dy, 'd5_ndepth_w5': d5_ndepth_w5, 'd5_ndepth_w17': d5_ndepth_w17
                }
        if self.decoder_resolution <= 4:
            return {'image': image,
                    'depth': depth,
                    'd1_depth': d1_depth,
                    'd2_depth': d2_depth,
                    'd3_depth': d3_depth,
                    'd4_depth': d4_depth, 'd4_depth_dx': d4_depth_dx, 'd4_depth_dy': d4_depth_dy, 'd4_ndepth_w5': d4_ndepth_w5, 'd4_ndepth_w17': d4_ndepth_w17,
                    'd5_depth': d5_depth, 'd5_depth_dx': d5_depth_dx, 'd5_depth_dy': d5_depth_dy, 'd5_ndepth_w5': d5_ndepth_w5, 'd5_ndepth_w17': d5_ndepth_w17
                }
        if self.decoder_resolution <= 5:
            return {'image': image,
                    'depth': depth,
                    'd1_depth': d1_depth,
                    'd2_depth': d2_depth,
                    'd3_depth': d3_depth,
                    'd4_depth': d4_depth,
                    'd5_depth': d5_depth, 'd5_depth_dx': d5_depth_dx, 'd5_depth_dy': d5_depth_dy, 'd5_ndepth_w5': d5_ndepth_w5, 'd5_ndepth_w17': d5_ndepth_w17
                }

    def get_relative_depth(self, depth):
        depth_dx_left = self.get_dx_left(depth).squeeze(0)
        depth_dx_right = self.get_dx_right(depth).squeeze(0)
        depth_dy_top = self.get_dy_top(depth).squeeze(0)
        depth_dy_bottom = self.get_dy_bottom(depth).squeeze(0)
        ndepth_w5 = self.get_ndepth_w5(depth).squeeze(0)
        ndepth_w17 = self.get_ndepth_w17(depth).squeeze(0)

        depth_dx = torch.cat((depth_dx_left, depth_dx_right), dim=0)
        depth_dy = torch.cat((depth_dy_top, depth_dy_bottom), dim=0)
        return depth_dx, depth_dy, ndepth_w5, ndepth_w17

    def get_dx_left(self, depth):
        depth_padded = F.pad(depth, pad=(1,0,0,0), mode='replicate')
        depth_dx_left = F.conv2d(depth_padded, self.kernel_dx_left)
        return depth_dx_left

    def get_dx_right(self, depth):
        depth_padded = F.pad(depth, pad=(0,1,0,0), mode='replicate')
        depth_dx_right = F.conv2d(depth_padded, self.kernel_dx_right)
        return depth_dx_right

    def get_dy_top(self, depth):
        depth_padded = F.pad(depth, pad=(0,0,1,0), mode='replicate')
        depth_dy_top = F.conv2d(depth_padded, self.kernel_dy_top)
        return depth_dy_top

    def get_dy_bottom(self, depth):
        depth_padded = F.pad(depth, pad=(0,0,0,1), mode='replicate')
        depth_dy_bottom = F.conv2d(depth_padded, self.kernel_dy_bottom)
        return depth_dy_bottom

    def get_ndepth_w5(self, depth):
        depth_padded = F.pad(depth, pad=(2,2,2,2), mode='replicate')
        depth_mean_w5 = F.conv2d(depth_padded, self.kernel_mean_w5)
        depth_ndepth_w5 = depth - depth_mean_w5
        return depth_ndepth_w5

    def get_ndepth_w17(self, depth):
        _,_,h,w = depth.size()
        if (h<17) and (w<17):
            depth_mean_w17 = depth.mean()
        else:
            depth_padded = F.pad(depth, pad=(8,8,8,8), mode='replicate')
            depth_mean_w17 = F.conv2d(depth_padded, self.kernel_mean_w17)
        depth_ndepth_w17 = depth - depth_mean_w17
        return depth_ndepth_w17

class OrdinaryRelativeDepth(object):
    def __init__(self, num_neighborhood = 24):
        self.num_neighborhood = num_neighborhood
        # interpolation function
        self.interpolate_down = []
        for index_down in range(6):
            self.interpolate_down.append(nn.Upsample(size=(9*pow(2, index_down), 12*pow(2,index_down)), mode='bilinear'))
        # kernel
        self.kernel = self.get_kernel(self.num_neighborhood)
        if self.num_neighborhood < 9:
            self.kernel_size = 3
            self.pad_size = 1
        elif self.num_neighborhood < 25:
            self.kernel_size = 5
            self.pad_size = 2
        elif self.num_neighborhood < 49:
            self.kernel_size = 7
            self.pad_size = 3

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        depth_unsqueeze = depth.unsqueeze(0).float().cuda()
        depth_new = []
        for index_type in range(12):
            depth_new.append([])
        for index_down in range(6):
            ordinary_depth_temp = self.interpolate_down[index_down](depth_unsqueeze)
            ordinary_depth_temp_with_pad = F.pad(ordinary_depth_temp, pad=[self.pad_size, self.pad_size, self.pad_size, self.pad_size], mode='replicate')
            relative_depth_temp = F.conv2d(ordinary_depth_temp_with_pad, self.kernel)

            ordinary_depth_temp = ordinary_depth_temp.squeeze(0).float().cpu().detach()
            relative_depth_temp = relative_depth_temp.squeeze(0).float().cpu().detach()

            depth_new[6*0 + index_down] = ordinary_depth_temp
            depth_new[6*1 + index_down] = relative_depth_temp

        depth_ORD = depth_new
        return {'image': image, 'depth': depth, 'depth_ORD': depth_ORD}

    def get_kernel(self, num_neighborhood = 24):
        if num_neighborhood < 9:
            kernel_size = 3
            kernel_center = 1
        elif num_neighborhood < 25:
            kernel_size = 5
            kernel_center = 2
        elif num_neighborhood < 49:
            kernel_size = 7
            kernel_center = 3

        kernel = np.zeros([kernel_size*kernel_size, 1, kernel_size, kernel_size])
        kernel[:,:,kernel_center,kernel_center] = 1

        for index_kernel_row in range(kernel_size):
            for index_kernel_col in range(kernel_size):
                index_kernel_batch = index_kernel_row * kernel_size + index_kernel_col
                kernel[index_kernel_batch,:,index_kernel_row,index_kernel_col] \
                    = kernel[index_kernel_batch,:,index_kernel_row,index_kernel_col] - 1
        kernel = torch.tensor(kernel).float().cuda()

        return kernel