# Image Generation from Small Datasets via Batch Statistics Adaptation

The author's official **minimal** implementation of [Image Generation from Small Datasets via Batch Statistics Adaptation](https://arxiv.org/abs/1904.01774).

Clean codes and optimal hyperparameters will be available soon.

## Requirements

```angular2
chainer>=5.0.0
opencv_python
numpy
scipy
Pillow
PyYAML
```

## Run
- Place `sngan.npz` and `biggan.npz` in this directory

- Multiple GPU training is supported only for BigGAN.

for BigGAN (We used 4 GPUs for training)
```
mpirun python ./train.py --config_path configs/default.yml
```

for SNGAN
```
python ./train.py --config_path configs/default.yml
```

## Acknowledgement
Pytorch [re-implementation](https://github.com/apple2373/PyTorch-SmallGAN) from Satoshi Tsutsui and Minjun Li. 
