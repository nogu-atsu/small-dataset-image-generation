# Chainer reimplementation of BigGAN-256.
import chainer
from chainer import functions as F
from chainer import links as L
from source.links.hyper_batch_normalization import HyperBatchNormalization
from source.links.sn_convolution_2d import SNConvolution2D
from source.links.sn_linear import SNLinear

import numpy as np


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class ResBlock(chainer.Chain):
    def __init__(self, in_size, in_channel, out_channel):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.HyperBN = HyperBatchNormalization(in_size, in_channel)
            self.conv0 = SNConvolution2D(in_channel, out_channel, 3, 1, 1)
            self.HyperBN_1 = HyperBatchNormalization(in_size, out_channel)
            self.conv1 = SNConvolution2D(out_channel, out_channel, 3, 1, 1)
            self.conv_sc = SNConvolution2D(in_channel, out_channel, 1, 1, 0)

    def __call__(self, x, z, c):
        h = self.HyperBN(x, z, c)
        h = upsample_conv(F.relu(h), self.conv0)
        h = self.conv1(F.relu(self.HyperBN_1(h, z, c)))
        return h + upsample_conv(x, self.conv_sc)


class NonLocalBlock(chainer.Chain):
    def __init__(self, ch):
        self.ch = ch
        super(NonLocalBlock, self).__init__()
        with self.init_scope():
            self.theta = SNConvolution2D(ch, ch // 8, 1, 1, 0, nobias=True)
            self.phi = SNConvolution2D(ch, ch // 8, 1, 1, 0, nobias=True)
            self.g = SNConvolution2D(ch, ch // 2, 1, 1, 0, nobias=True)
            self.o_conv = SNConvolution2D(ch // 2, ch, 1, 1, 0, nobias=True)
            self.gamma = L.Parameter(np.array(0, dtype="float32"))

    def __call__(self, x):
        batchsize, _, w, _ = x.shape
        f = self.theta(x).reshape(batchsize, self.ch // 8, -1)
        g = self.phi(x)
        g = F.max_pooling_2d(g, 2, 2).reshape(batchsize, self.ch // 8, -1)
        attention = F.softmax(F.matmul(f, g, transa=True), axis=2)
        h = self.g(x)
        h = F.max_pooling_2d(h, 2, 2).reshape(batchsize, self.ch // 2, -1)
        o = F.matmul(h, attention, transb=True).reshape(batchsize, self.ch // 2, w, w)
        o = self.o_conv(o)
        return x + self.gamma.W * o


class BIGGAN(chainer.Chain):
    def __init__(self, ch=96):
        self.ch = ch
        super(BIGGAN, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(1000, 128, nobias=True)
            self.G_linear = SNLinear(20, 4 * 4 * 16 * ch)
            self.GBlock = ResBlock(148, 16 * ch, 16 * ch)
            self.GBlock_1 = ResBlock(148, 16 * ch, 8 * ch)
            self.GBlock_2 = ResBlock(148, 8 * ch, 8 * ch)
            self.GBlock_3 = ResBlock(148, 8 * ch, 4 * ch)
            self.GBlock_4 = ResBlock(148, 4 * ch, 2 * ch)
            self.attention = NonLocalBlock(2 * ch)
            self.GBlock_5 = ResBlock(148, 2 * ch, ch)
            self.ScaledCrossReplicaBN = L.BatchNormalization(ch)
            self.conv_2d = SNConvolution2D(ch, 3, 3, 1, 1)

    def forward(self, z, c):
        c = self.linear(c)
        z = F.split_axis(z, 7, axis=1)
        h = self.G_linear(z[0]).reshape(-1, 4, 4, 16 * self.ch).transpose(0, 3, 1, 2)
        h = self.GBlock(h, z[1], c)
        h = self.GBlock_1(h, z[2], c)
        h = self.GBlock_2(h, z[3], c)
        h = self.GBlock_3(h, z[4], c)
        h = self.GBlock_4(h, z[5], c)
        h = self.attention(h)
        h = self.GBlock_5(h, z[6], c)
        h = F.relu(self.ScaledCrossReplicaBN(h))
        h = F.tanh(self.conv_2d(h))
        return h
