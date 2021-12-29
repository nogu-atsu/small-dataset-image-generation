# Image Generation from Small Datasets via Batch Statistics Adaptation

The author's official **minimal** implementation of [Image Generation from Small Datasets via Batch Statistics Adaptation](https://arxiv.org/abs/1904.01774).

## Requirements

```angular2
chainer>=5.0.0
opencv_python
numpy
scipy
Pillow
PyYAML
```

## Dataset preparation
```angular2
data_path
├── dataset1
├── dataset2
...
```
- Place all training sample in the same directory for each dataset.
- Specify the root path to `data_path` in `configs/default.yml`.
- Specify the dataset directory name to `dataset` in `configs/default.yml`.


## Run
- Download SNGAN-128 or BigGAN-256 generator pre-trained on ImageNet.
    - SNGAN generator can be downloaded from https://drive.google.com/drive/folders/1m04Db3HbN10Tz5XHqiPpIg8kw23fkbSi (ResNetGenerator_850000.npz)
    - BigGAN generator can be saved by https://github.com/nogu-atsu/chainer-BIGGAN
- Specify paths for them to `snapshot` in `configs/default.yml`
- Specify `gan_type` in `configs/default.yml`

For single GPU training, run
```
python ./train.py --config_path configs/default.yml
```

For Multiple GPU training, run 
```
mpirun python ./train.py --config_path configs/default.yml
```
Multiple GPU training is supported only for BigGAN. For BigGAN ,we used 4 GPUs for training.

## Inference
Initialize the generator and load pretrained weights.

e.g. biggan based model
```
gen = AdaBIGGAN(config, datasize, comm=comm)
chainer.serializers.load_npz("your_pretrained_model.h5", gen.gen)
gen.to_gpu(device) # send to gpu if necessary
gen.gen.to_gpu(device) # send to gpu if necessary
```

Random sampling

e.g. randomly sample 5 images with temperature=0.5 without truncation
```
random_imgs = gen.random(tmp=0.5, n=5, truncate=False)
```

Interpolation

e.g. interpolation between 0th and 1st image
```
interpolated_imgs = gen.interpolate(self, source=0, dest=1, num=5)
```

## Acknowledgement
Pytorch [re-implementation](https://github.com/apple2373/PyTorch-SmallGAN) from Satoshi Tsutsui and Minjun Li. 
