# Small dataset image generation

Chainer implementation of small dataset image generation.

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
- Multiple GPU training is supported only for BigGAN.

for BigGAN
```
mpirun python ./train.py --config_path configs/default.yml
```

for SNGAN
```
python ./train.py --config_path configs/default.yml
```


