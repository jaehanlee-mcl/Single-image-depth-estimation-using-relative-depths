from data import getTrainDataPath, getTestDataPath, loadZipToMem, ToTensor_with_RandomZoom, depthDatasetMemoryTrain
from torchvision import transforms
from torch.utils.data import DataLoader
from nyu_transform import *

def getTestingData(batch_size, test_data_use, data_transform_setting='Default', num_neighborhood=24):
    test_dataset_path, test_dataset_csv_list = getTestDataPath()

    use_NYUv2_test = test_data_use['NYUv2_test']
    use_NYUv2_test_raw = test_data_use['NYUv2_test_raw']
    use_KITTI_Eigen_test = test_data_use['KITTI_Eigen_test']
    use_Make3D_test = test_data_use['Make3D_test']

    dataset_path_NYUv2_test = test_dataset_path['NYUv2_test']
    dataset_path_NYUv2_test_raw = test_dataset_path['NYUv2_test_raw']
    dataset_path_KITTI_Eigen_test = test_dataset_path['KITTI_Eigen_test']
    dataset_path_Make3D_test = test_dataset_path['Make3D_test']

    dataset_csv_NYUv2_test = test_dataset_csv_list['NYUv2_test']
    dataset_csv_NYUv2_test_raw = test_dataset_csv_list['NYUv2_test_raw']
    dataset_csv_KITTI_Eigen_test = test_dataset_csv_list['KITTI_Eigen_test']
    dataset_csv_Make3D_test = test_dataset_csv_list['Make3D_test']

    if use_NYUv2_test == True:
        data_temp, test_temp = loadZipToMem(dataset_path_NYUv2_test, dataset_csv_NYUv2_test)
        data = data_temp.copy()
        test = test_temp

    if use_NYUv2_test_raw == True:
        data_temp, test_temp = loadZipToMem(dataset_path_NYUv2_test_raw, dataset_csv_NYUv2_test_raw)
        if not('data' in locals()):
            data = data_temp.copy()
            test = test_temp
        else:
            data.update(data_temp)
            test = test + test_temp

    if use_KITTI_Eigen_test == True:
        data_temp, test_temp = loadZipToMem(dataset_path_KITTI_Eigen_test, dataset_csv_KITTI_Eigen_test)
        if not('data' in locals()):
            data = data_temp.copy()
            test = test_temp
        else:
            data.update(data_temp)
            test = test + test_temp

    if use_Make3D_test == True:
        data_temp, test_temp = loadZipToMem(dataset_path_Make3D_test, dataset_csv_Make3D_test)
        if not('data' in locals()):
            data = data_temp.copy()
            test = test_temp
        else:
            data.update(data_temp)
            test = test + test_temp

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    if data_transform_setting == 'Default':
        transformed_testing = transforms.Compose([
            Scale(480),
            ToTensor_with_RandomZoom(ratio=1.00),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std']),
            ])
    elif data_transform_setting == 'OrdinaryRelativeDepth':
        transformed_testing = transforms.Compose([
            Scale(480),
            ToTensor_with_RandomZoom(ratio=1.00),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std']),
            LogDepth(),
            OrdinaryRelativeDepth(num_neighborhood=num_neighborhood)
        ])

    transformed_testing = depthDatasetMemoryTrain(data, test, transform=transformed_testing)

    return DataLoader(transformed_testing, batch_size, shuffle=False), len(test)

def getTrainingData(batch_size, train_dataset_use, data_transform_setting='Default', num_neighborhood=24):
    # data_transform_setting:
    #   - default (get 288x384 depth map)
    #   - OrdinaryRelativeDepth24 (get 288x384, 144x192, 72x96, 36x48, 18x24, 9x12 ordinary & relative depth map(24 neighborhood))
    train_dataset_path, train_dataset_csv_list = getTrainDataPath()

    use_NYUv2_reduced01 = train_dataset_use['NYUv2_train_reduced01']
    use_NYUv2_reduced05 = train_dataset_use['NYUv2_train_reduced05']
    use_NYUv2_reduced06 = train_dataset_use['NYUv2_train_reduced06']
    use_NYUv2_reduced10 = train_dataset_use['NYUv2_train_reduced10']
    use_NYUv2_reduced15 = train_dataset_use['NYUv2_train_reduced15']
    use_NYUv2_reduced20 = train_dataset_use['NYUv2_train_reduced20']
    use_NYUv2_reduced30 = train_dataset_use['NYUv2_train_reduced30']
    use_SUNRGB_D_reduced01 = train_dataset_use['SUNRGB_D_train_reduced01']
    use_SUNRGB_D_reduced02 = train_dataset_use['SUNRGB_D_train_reduced02']
    use_Matterport3D_reduced01 = train_dataset_use['Matterport3D_train_reduced01']
    use_Matterport3D_reduced05 = train_dataset_use['Matterport3D_train_reduced05']
    use_Matterport3D_reduced10 = train_dataset_use['Matterport3D_train_reduced10']
    use_KITTI_Eigen_reduced01 = train_dataset_use['KITTI_Eigen_train_reduced01']
    use_KITTI_Eigen_reduced03 = train_dataset_use['KITTI_Eigen_train_reduced03']
    use_Make3D_reduced01 = train_dataset_use['Make3D_train_reduced01']
    use_ReDWeb_V1_reduced01 = train_dataset_use['ReDWeb_V1_train_reduced01']

    dataset_path_NYUv2_reduced01 = train_dataset_path['NYUv2_train_reduced01']
    dataset_path_NYUv2_reduced05 = train_dataset_path['NYUv2_train_reduced05']
    dataset_path_NYUv2_reduced06 = train_dataset_path['NYUv2_train_reduced06']
    dataset_path_NYUv2_reduced10 = train_dataset_path['NYUv2_train_reduced10']
    dataset_path_NYUv2_reduced15 = train_dataset_path['NYUv2_train_reduced15']
    dataset_path_NYUv2_reduced20 = train_dataset_path['NYUv2_train_reduced20']
    dataset_path_NYUv2_reduced30 = train_dataset_path['NYUv2_train_reduced30']
    dataset_path_SUNRGB_D_reduced01 = train_dataset_path['SUNRGB_D_train_reduced01']
    dataset_path_SUNRGB_D_reduced02 = train_dataset_path['SUNRGB_D_train_reduced02']
    dataset_path_Matterport3D_reduced01 = train_dataset_path['Matterport3D_train_reduced01']
    dataset_path_Matterport3D_reduced05 = train_dataset_path['Matterport3D_train_reduced05']
    dataset_path_Matterport3D_reduced10 = train_dataset_path['Matterport3D_train_reduced10']
    dataset_path_KITTI_Eigen_reduced01 = train_dataset_path['KITTI_Eigen_train_reduced01']
    dataset_path_KITTI_Eigen_reduced03 = train_dataset_path['KITTI_Eigen_train_reduced03']
    dataset_path_Make3D_reduced01 = train_dataset_path['Make3D_train_reduced01']
    dataset_path_ReDWeb_V1_reduced01 = train_dataset_path['ReDWeb_V1_train_reduced01']

    dataset_csv_NYUv2_reduced01 = train_dataset_csv_list['NYUv2_train_reduced01']
    dataset_csv_NYUv2_reduced05 = train_dataset_csv_list['NYUv2_train_reduced05']
    dataset_csv_NYUv2_reduced06 = train_dataset_csv_list['NYUv2_train_reduced06']
    dataset_csv_NYUv2_reduced10 = train_dataset_csv_list['NYUv2_train_reduced10']
    dataset_csv_NYUv2_reduced15 = train_dataset_csv_list['NYUv2_train_reduced15']
    dataset_csv_NYUv2_reduced20 = train_dataset_csv_list['NYUv2_train_reduced20']
    dataset_csv_NYUv2_reduced30 = train_dataset_csv_list['NYUv2_train_reduced30']
    dataset_csv_SUNRGB_D_reduced01 = train_dataset_csv_list['SUNRGB_D_train_reduced01']
    dataset_csv_SUNRGB_D_reduced02 = train_dataset_csv_list['SUNRGB_D_train_reduced02']
    dataset_csv_Matterport3D_reduced01 = train_dataset_csv_list['Matterport3D_train_reduced01']
    dataset_csv_Matterport3D_reduced05 = train_dataset_csv_list['Matterport3D_train_reduced05']
    dataset_csv_Matterport3D_reduced10 = train_dataset_csv_list['Matterport3D_train_reduced10']
    dataset_csv_KITTI_Eigen_reduced01 = train_dataset_csv_list['KITTI_Eigen_train_reduced01']
    dataset_csv_KITTI_Eigen_reduced03 = train_dataset_csv_list['KITTI_Eigen_train_reduced03']
    dataset_csv_Make3D_reduced01 = train_dataset_csv_list['Make3D_train_reduced01']
    dataset_csv_ReDWeb_V1_reduced01 = train_dataset_csv_list['ReDWeb_V1_train_reduced01']

    if use_NYUv2_reduced01 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_NYUv2_reduced01, dataset_csv_NYUv2_reduced01)
        data = data_temp.copy()
        train = train_temp

    if use_NYUv2_reduced05 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_NYUv2_reduced05, dataset_csv_NYUv2_reduced05)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_NYUv2_reduced06 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_NYUv2_reduced06, dataset_csv_NYUv2_reduced06)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_NYUv2_reduced10 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_NYUv2_reduced10, dataset_csv_NYUv2_reduced10)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_NYUv2_reduced15 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_NYUv2_reduced15, dataset_csv_NYUv2_reduced15)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_NYUv2_reduced20 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_NYUv2_reduced20, dataset_csv_NYUv2_reduced20)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_NYUv2_reduced30 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_NYUv2_reduced30, dataset_csv_NYUv2_reduced30)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_SUNRGB_D_reduced01 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_SUNRGB_D_reduced01, dataset_csv_SUNRGB_D_reduced01)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_SUNRGB_D_reduced02 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_SUNRGB_D_reduced02, dataset_csv_SUNRGB_D_reduced02)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_Matterport3D_reduced01 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_Matterport3D_reduced01, dataset_csv_Matterport3D_reduced01)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_Matterport3D_reduced05 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_Matterport3D_reduced05, dataset_csv_Matterport3D_reduced05)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_Matterport3D_reduced10 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_Matterport3D_reduced10, dataset_csv_Matterport3D_reduced10)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_KITTI_Eigen_reduced01 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_KITTI_Eigen_reduced01, dataset_csv_KITTI_Eigen_reduced01)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_KITTI_Eigen_reduced03 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_KITTI_Eigen_reduced03, dataset_csv_KITTI_Eigen_reduced03)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_Make3D_reduced01 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_Make3D_reduced01, dataset_csv_Make3D_reduced01)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    if use_ReDWeb_V1_reduced01 == True:
        data_temp, train_temp = loadZipToMem(dataset_path_ReDWeb_V1_reduced01, dataset_csv_ReDWeb_V1_reduced01)
        if not('data' in locals()):
            data = data_temp.copy()
            train = train_temp
        else:
            data.update(data_temp)
            train = train + train_temp

    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    if data_transform_setting == 'Default':
        transformed_training = transforms.Compose([
            Scale(288),
            RandomHorizontalFlip(),
            RandomRotate(5),
            #RandomChannelSwap(0.5),
            ToTensor_with_RandomZoom(ratio=1.00),
            Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, ),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std']),
            LogDepth()
            ])
    elif data_transform_setting == 'OrdinaryRelativeDepth':
        transformed_training = transforms.Compose([
            Scale(288),
            RandomHorizontalFlip(),
            # RandomRotate(5),
            # RandomChannelSwap(0.5),
            ToTensor_with_RandomZoom(ratio=1.00),
            Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, ),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std']),
            LogDepth(),
            OrdinaryRelativeDepth(num_neighborhood=num_neighborhood)
        ])

    transformed_training = depthDatasetMemoryTrain(data, train, transform=transformed_training)

    return DataLoader(transformed_training, batch_size, shuffle=True), len(train)
