# Small dataset image generation

Official minimal implementation of small dataset image generation.

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


