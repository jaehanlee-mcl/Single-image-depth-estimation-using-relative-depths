# [JVCI 2022] Single-image depth estimation using relative depths

[Paper link](https://www.sciencedirect.com/science/article/abs/pii/S1047320322000190)

If you use our code or results, please cite:
```
@article{lee2022single,
  title={Single-image depth estimation using relative depths},
  author={Lee, Jae-Han and Kim, Chang-Su},
  journal={Journal of Visual Communication and Image Representation},
  volume={84},
  pages={103459},
  year={2022},
  publisher={Elsevier}
}
```

https://drive.google.com/file/d/11M53vnKEM_i0Uu-HXy-ksfZLSXPJsvZG/view?usp=sharing

------
### 1. Preparation
1. Download a [model](https://drive.google.com/file/d/11M53vnKEM_i0Uu-HXy-ksfZLSXPJsvZG/view?usp=sharing) and place them in 'runs\'

------
### 2. NYUv2 dataset
NYUv2 training and evaluation data are obtained by reprocessing what is provided on the [official site](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). You can easily use the reprocessed data with the link below.

All depth maps are saved as 16bit png files.

Depths are mapped by the following formula: depth = double(value) / (2^16 - 1) * 10

- 654 RGBD pairs for evaluation: [test654.zip](https://drive.google.com/file/d/1JYjSgf5Fn6eg2gJkqmuBZXI2xMTdyIxX/view?usp=sharing)

- 795 RGBD pairs for training: [train795.zip](https://drive.google.com/file/d/1VNRsXzc0MMjjXLdJpcwBTh1eosif7orU/view?usp=sharing)

- RGBD pairs obtained from the training sequence: [train_reduced05.zip](https://drive.google.com/file/d/1s6-4mm-wDwo0bwEG1LKLsadjB0K5EosP/view?usp=sharing) (8 GB)

Dataset files should be placed in 'dataset\' directory without extracting them.

------
