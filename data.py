from torch.utils.data import Dataset
from io import BytesIO
import matplotlib.pyplot as plt
from nyu_transform import *

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

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

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}

def loadZipToMem(zip_file, csv_name):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')

    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    train = list((row.split(',') for row in (data[csv_name]).decode("utf-8").split('\n') if len(row) > 0))
    #from sklearn.utils import shuffle
    #train = shuffle(train, random_state=0)

    print('Loaded ({0}) data.'.format(len(train)))
    return data, train

class depthDatasetMemoryTrain(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform
        self.maxDepth = 1000.0

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )

        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor_with_RandomZoom(object):
    def __init__(self, ratio=1):
        self.ratio = ratio

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        original_size = image.size
        applied_zoom = random.uniform(1, self.ratio)

        image, depth = self.zoom(image, depth, applied_zoom)
        image, depth = self.randomCrop(image, depth, original_size)
        
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).float()
        depth = depth / (pow(2, 16) - 1) * 10 / applied_zoom
        depth_mean = depth.mean()

        return {'image': image, 'depth': depth}

    def zoom(self, image, depth, applied_zoom):
        w1, h1 = image.size
        w2 = round(w1 * applied_zoom)
        h2 = round(h1 * applied_zoom)

        image = image.resize((w2, h2), Image.BICUBIC)
        depth = depth.resize((w2, h2), Image.BICUBIC)

        return image, depth

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

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getTrainDataPath():
    # path for rr/rh
    train_dataset_path = {'NYUv2_train_reduced01': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced01.zip",
                          'NYUv2_train_reduced05': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced05.zip",
                          'NYUv2_train_reduced06': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced06.zip",
                          'NYUv2_train_reduced10': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced10.zip",
                          'NYUv2_train_reduced15': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced15.zip",
                          'NYUv2_train_reduced20': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced20.zip",
                          'NYUv2_train_reduced30': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced30.zip",
                          'NYUv2_train_reduced795': "C:/moveToD/depth_map_estimation/dataset/NYUv2/train_reduced795.zip",
                          'SUNRGB_D_train_reduced01': "C:/moveToD/depth_map_estimation/dataset/SUNRGB_D/train_reduced01.zip",
                          'SUNRGB_D_train_reduced02': "C:/moveToD/depth_map_estimation/dataset/SUNRGB_D/train_reduced02.zip",
                          'Matterport3D_train_reduced01': "C:/moveToD/depth_map_estimation/dataset/Matterport3D/train_reduced01.zip",
                          'Matterport3D_train_reduced05': "C:/moveToD/depth_map_estimation/dataset/Matterport3D/train_reduced05.zip",
                          'Matterport3D_train_reduced10': "C:/moveToD/depth_map_estimation/dataset/Matterport3D/train_reduced10.zip",
                          'KITTI_Eigen_train_reduced01': "C:/moveToD/depth_map_estimation/dataset/KITTI_Eigen/train_reduced01.zip",
                          'KITTI_Eigen_train_reduced03': "C:/moveToD/depth_map_estimation/dataset/KITTI_Eigen/train_reduced03.zip",
                          'Make3D_train_reduced01': "C:/moveToD/depth_map_estimation/dataset/Make3D/train_reduced01.zip",
                          'ReDWeb_V1_train_reduced01': "C:/moveToD/depth_map_estimation/dataset/ReDWeb_V1/train_reduced01.zip"
                          }
    """
    train_dataset_path = {'NYUv2_train_reduced01': "D:/depth_map_estimation/dataset/NYUv2/train_reduced01.zip",
                          'NYUv2_train_reduced05': "D:/depth_map_estimation/dataset/NYUv2/train_reduced05.zip",
                          'NYUv2_train_reduced06': "D:/depth_map_estimation/dataset/NYUv2/train_reduced06.zip",
                          'NYUv2_train_reduced10': "D:/depth_map_estimation/dataset/NYUv2/train_reduced10.zip",
                          'NYUv2_train_reduced15': "D:/depth_map_estimation/dataset/NYUv2/train_reduced15.zip",
                          'NYUv2_train_reduced20': "D:/depth_map_estimation/dataset/NYUv2/train_reduced20.zip",
                          'NYUv2_train_reduced30': "D:/depth_map_estimation/dataset/NYUv2/train_reduced30.zip",
                          'SUNRGB_D_train_reduced01': "D:/depth_map_estimation/dataset/SUNRGB_D/train_reduced01.zip",
                          'SUNRGB_D_train_reduced02': "D:/depth_map_estimation/dataset/SUNRGB_D/train_reduced02.zip",
                          'Matterport3D_train_reduced01': "D:/depth_map_estimation/dataset/Matterport3D/train_reduced01.zip",
                          'Matterport3D_train_reduced05': "D:/depth_map_estimation/dataset/Matterport3D/train_reduced05.zip",
                          'Matterport3D_train_reduced10': "D:/depth_map_estimation/dataset/Matterport3D/train_reduced10.zip",
                          'KITTI_Eigen_train_reduced01': "D:/depth_map_estimation/dataset/KITTI_Eigen/train_reduced01.zip",
                          'KITTI_Eigen_train_reduced03': "D:/depth_map_estimation/dataset/KITTI_Eigen/train_reduced03.zip",
                          'Make3D_train_reduced01': "D:/depth_map_estimation/dataset/Make3D/train_reduced01.zip",
                          'ReDWeb_V1_train_reduced01': "D:/depth_map_estimation/dataset/ReDWeb_V1/train_reduced01.zip"
                          }
    """
    """
    # path for ur
    train_dataset_path = {'NYUv2_train_reduced01': "/media/HDD1/jhlee/dataset/NYUv2/train_reduced01.zip",
                          'NYUv2_train_reduced05': "/media/HDD1/jhlee/dataset/NYUv2/train_reduced05.zip",
                          'NYUv2_train_reduced06': "/media/HDD1/jhlee/dataset/NYUv2/train_reduced06.zip",
                          'NYUv2_train_reduced10': "/media/HDD1/jhlee/dataset/NYUv2/train_reduced10.zip",
                          'NYUv2_train_reduced15': "/media/HDD1/jhlee/dataset/NYUv2/train_reduced15.zip",
                          'NYUv2_train_reduced20': "/media/HDD1/jhlee/dataset/NYUv2/train_reduced20.zip",
                          'NYUv2_train_reduced30': "/media/HDD1/jhlee/dataset/NYUv2/train_reduced30.zip",
                          'SUNRGB_D_train_reduced01': "/media/HDD1/jhlee/dataset/SUNRGB_D/train_reduced01.zip",
                          'SUNRGB_D_train_reduced02': "/media/HDD1/jhlee/dataset/SUNRGB_D/train_reduced02.zip",
                          'Matterport3D_train_reduced01': "/media/HDD1/jhlee/dataset/Matterport3D/train_reduced01.zip",
                          'Matterport3D_train_reduced05': "/media/HDD1/jhlee/dataset/Matterport3D/train_reduced05.zip",
                          'Matterport3D_train_reduced10': "/media/HDD1/jhlee/dataset/Matterport3D/train_reduced10.zip",
                          'KITTI_Eigen_train_reduced01': "/media/HDD1/jhlee/dataset/KITTI_Eigen/train_reduced01.zip",
                          'KITTI_Eigen_train_reduced03': "/media/HDD1/jhlee/dataset/KITTI_Eigen/train_reduced03.zip",
                          'Make3D_train_reduced01': "/media/HDD1/jhlee/dataset/Make3D/train_reduced01.zip",
                          'ReDWeb_V1_train_reduced01': "/media/HDD1/jhlee/dataset/ReDWeb_V1/train_reduced01.zip"
                          }
    """
    # location of list(.csv) file
    train_dataset_csv_list = {'NYUv2_train_reduced01': 'train_reduced01/train.csv',
                              'NYUv2_train_reduced05': 'train_reduced05/train.csv',
                              'NYUv2_train_reduced06': 'train_reduced06/train.csv',
                              'NYUv2_train_reduced10': 'train_reduced10/train.csv',
                              'NYUv2_train_reduced15': 'train_reduced15/train.csv',
                              'NYUv2_train_reduced20': 'train_reduced20/train.csv',
                              'NYUv2_train_reduced30': 'train_reduced30/train.csv',
                              'SUNRGB_D_train_reduced01': 'train_reduced01/train.csv',
                              'SUNRGB_D_train_reduced02': 'train_reduced02/train.csv',
                              'Matterport3D_train_reduced01': 'train_reduced01/train.csv',
                              'Matterport3D_train_reduced05': 'train_reduced05/train.csv',
                              'Matterport3D_train_reduced10': 'train_reduced10/train.csv',
                              'KITTI_Eigen_train_reduced01': 'eigen_split_train_resized/train.csv',
                              'KITTI_Eigen_train_reduced03': 'train_reduced03/train.csv',
                              'Make3D_train_reduced01': 'train_reduced01/train.csv',
                              'ReDWeb_V1_train_reduced01': 'train_reduced01/train.csv'
                              }
    return train_dataset_path, train_dataset_csv_list

def getTestDataPath():
    # path for rr/rh
    test_dataset_path = {'NYUv2_test': "C:/moveToD/depth_map_estimation/dataset/NYUv2/test.zip",
                         'NYUv2_test_raw': "C:/moveToD/depth_map_estimation/dataset/NYUv2/test_raw.zip",
                         'KITTI_Eigen_test': "C:/moveToD/depth_map_estimation/dataset/KITTI_Eigen/test.zip",
                         'Make3D_test': "C:/moveToD/depth_map_estimation/dataset/Make3D/test.zip"
                         }
    """
    test_dataset_path = {'NYUv2_test': "D:/depth_map_estimation/dataset/NYUv2/test.zip",
                         'NYUv2_test_raw': "D:/depth_map_estimation/dataset/NYUv2/test_raw.zip",
                         'KITTI_Eigen_test': "D:/depth_map_estimation/dataset/KITTI_Eigen/test.zip",
                         'Make3D_test': "D:/depth_map_estimation/dataset/Make3D/test.zip"
                         }
    """
    """
    # path for ur
    test_dataset_path = {'NYUv2_test': "/media/HDD1/jhlee/dataset/NYUv2/test.zip",
                         'NYUv2_test_raw': "/media/HDD1/jhlee/dataset/NYUv2/test_raw.zip",
                         'KITTI_Eigen_test': "/media/HDD1/jhlee/dataset/KITTI_Eigen/test.zip",
                         'Make3D_test': "/media/HDD1/jhlee/dataset/Make3D/test.zip"
                         }
    """
    # location of list(.csv) file
    test_dataset_csv_list = {'NYUv2_test': 'test/test.csv',
                             'NYUv2_test_raw': 'test_raw/test.csv',
                             'KITTI_Eigen_test': 'eigen_split_test_resized/test.csv',
                             'Make3D_test': 'test/test.csv'
                             }
    return test_dataset_path, test_dataset_csv_list

