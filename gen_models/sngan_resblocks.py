import math
import chainer
import chainer.links as L
from chainer import functions as F
from source.links.categorical_conditional_batch_normalization import CategoricalConditionalBatchNormalization


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if n_classes > 0:
                self.b1 = CategoricalConditionalBatchNormalization(in_channels, n_cat=n_classes)
                self.b2 = CategoricalConditionalBatchNormalization(hidden_channels, n_cat=n_classes)
            else:
                self.b1 = L.BatchNormalization(in_channels)
                self.b2 = L.BatchNormalization(hidden_channels)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def initialize_params(self, gamma, beta):
        if self.n_classes > 0:
            self.b1.initialize_params(gamma, beta)
            self.b2.initialize_params(gamma, beta)
        else:
            if gamma:
                self.b1.gamma.array[:] = 1.
                self.b2.gamma.array[:] = 1.
            if beta:
                self.b1.beta.array[:] = 0.
                self.b2.beta.array[:] = 0.

    def residual(self, x, y=None, z=None, gamma=None, beta=None, **kwargs):
        assert (gamma is None) or isinstance(gamma, list), "gamma is not list"
        h = x
        h = self.b1(h, y, **kwargs) if y is not None and y.shape[1] > 0 else self.b1(h, **kwargs)
        if gamma and beta:
            h = self.shift(h, gamma[0], beta[0])

        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None and y.shape[1] > 0 else self.b2(h, **kwargs)
        if gamma and beta:
            h = self.shift(h, gamma[1], beta[1])

        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x, y=None, z=None, gamma=None, beta=None, **kwargs):
        return self.residual(x, y, z, gamma, beta, **kwargs) + self.shortcut(x)

    def shift(self, x, gamma, beta):
        if gamma.ndim == 1 and beta.ndim == 1:
            x = x * F.broadcast_to(gamma[None, :, None, None], x.shape) + \
                F.broadcast_to(beta[None, :, None, None], x.shape)
        elif gamma.ndim == 2 and beta.ndim == 2:
            x = x * F.broadcast_to(gamma[:, :, None, None], x.shape) + \
                F.broadcast_to(beta[:, :, None, None], x.shape)
        return x

    def start_finetuning(self):
        self.b1.start_finetuning()
        self.b2.start_finetuning()
