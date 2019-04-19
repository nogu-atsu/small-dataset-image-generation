# Chainer reimplementation of BigGAN-256.
import chainer
from chainer import functions as F
from chainer import links as L
from source.links.shared_embedding_batch_normalization import SharedEmbeddingBatchNormalization
from source.links.sn_convolution_2d import SNConvolution2D
from source.links.sn_linear import SNLinear

import numpy as np


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


def copy_conv(link, w, b=None, u=None):
    if u is not None:
        link.u[:] = u
    link.W.data[:] = w.transpose(3, 2, 0, 1)
    if b is not None:
        link.b.data[:] = b


def copy_hbn(link, gamma, gamma_u, beta, beta_u, avg_mean, avg_var):
    copy_linear(link.linear_gamma, gamma, u=gamma_u)
    copy_linear(link.linear_beta, beta, u=beta_u)
    link.avg_mean[:] = avg_mean
    link.avg_var[:] = avg_var
    # layer.eps = eps


def copy_bn(link, gamma, beta, avg_mean, avg_var):
    link.beta.data[:] = beta
    link.gamma.data[:] = gamma
    link.avg_mean[:] = avg_mean
    link.avg_var[:] = avg_var


def copy_linear(link, w, b=None, u=None):
    if u is not None:
        link.u[:] = u
    link.W.data[:] = w.transpose()
    if b is not None:
        link.b.data[:] = b


class ResBlock(chainer.Chain):
    def __init__(self, in_size, in_channel, out_channel):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.HyperBN = SharedEmbeddingBatchNormalization(in_size, in_channel)
            self.conv0 = SNConvolution2D(in_channel, out_channel, 3, 1, 1)
            self.HyperBN_1 = SharedEmbeddingBatchNormalization(in_size, out_channel)
            self.conv1 = SNConvolution2D(out_channel, out_channel, 3, 1, 1)
            self.conv_sc = SNConvolution2D(in_channel, out_channel, 1, 1, 0)

    def __call__(self, x, z, c):
        h = self.HyperBN(x, z, c)
        h = upsample_conv(F.relu(h), self.conv0)
        h = self.conv1(F.relu(self.HyperBN_1(h, z, c)))
        return h + upsample_conv(x, self.conv_sc)

    def copy_from_tf(self, weights_dict, name):
        copy_conv(self.conv0, weights_dict[name + '/conv0/w/ema_b999900'],
                  weights_dict[name + '/conv0/b/ema_b999900'],
                  weights_dict[name + '/conv0/u0'], )
        copy_conv(self.conv1, weights_dict[name + '/conv1/w/ema_b999900'],
                  weights_dict[name + '/conv1/b/ema_b999900'],
                  weights_dict[name + '/conv1/u0'])
        copy_conv(self.conv_sc, weights_dict[name + '/conv_sc/w/ema_b999900'],
                  weights_dict[name + '/conv_sc/b/ema_b999900'],
                  weights_dict[name + '/conv_sc/u0'])
        copy_hbn(self.HyperBN, weights_dict[name + '/HyperBN/gamma/w/ema_b999900'],
                 weights_dict[name + '/HyperBN/gamma/u0'],
                 weights_dict[name + '/HyperBN/beta/w/ema_b999900'],
                 weights_dict[name + '/HyperBN/beta/u0'],
                 weights_dict[name + '/CrossReplicaBN/accumulated_mean'],
                 weights_dict[name + '/CrossReplicaBN/accumulated_var'])
        copy_hbn(self.HyperBN_1, weights_dict[name + '/HyperBN_1/gamma/w/ema_b999900'],
                 weights_dict[name + '/HyperBN_1/gamma/u0'],
                 weights_dict[name + '/HyperBN_1/beta/w/ema_b999900'],
                 weights_dict[name + '/HyperBN_1/beta/u0'],
                 weights_dict[name + '/CrossReplicaBN_1/accumulated_mean'],
                 weights_dict[name + '/CrossReplicaBN_1/accumulated_var'])


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

    def copy_from_tf(self, weights_dict, name):
        copy_conv(self.theta, weights_dict[name + '/theta/w/ema_b999900'],
                  u=weights_dict[name + '/theta/u0'])
        copy_conv(self.phi, weights_dict[name + '/phi/w/ema_b999900'],
                  u=weights_dict[name + '/phi/u0'])
        copy_conv(self.g, weights_dict[name + '/g/w/ema_b999900'],
                  u=weights_dict[name + '/g/u0'])
        copy_conv(self.o_conv, weights_dict[name + '/o_conv/w/ema_b999900'],
                  u=weights_dict[name + '/o_conv/u0'])
        self.gamma.W.data = weights_dict[name + '/gamma/ema_b999900']


class Generator(chainer.Chain):
    def __init__(self, ch=96):
        self.ch = ch
        super(Generator, self).__init__()
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

    def copy_params_from_tf(self, weights_dict):
        copy_linear(self.linear, weights_dict['linear/w/ema_b999900'])
        copy_linear(self.G_linear, weights_dict['Generator/G_Z/G_linear/w/ema_b999900'],
                    weights_dict['Generator/G_Z/G_linear/b/ema_b999900'],
                    weights_dict['Generator/G_Z/G_linear/u0'])
        self.GBlock.copy_from_tf(weights_dict, 'Generator/GBlock')
        self.GBlock_1.copy_from_tf(weights_dict, 'Generator/GBlock_1')
        self.GBlock_2.copy_from_tf(weights_dict, 'Generator/GBlock_2')
        self.GBlock_3.copy_from_tf(weights_dict, 'Generator/GBlock_3')
        self.GBlock_4.copy_from_tf(weights_dict, 'Generator/GBlock_4')
        self.attention.copy_from_tf(weights_dict, 'Generator/attention')
        self.GBlock_5.copy_from_tf(weights_dict, 'Generator/GBlock_5')
        copy_bn(self.ScaledCrossReplicaBN, weights_dict['Generator/ScaledCrossReplicaBN/gamma/ema_b999900'],
                weights_dict['Generator/ScaledCrossReplicaBN/beta/ema_b999900'],
                weights_dict['Generator/ScaledCrossReplicaBNbn/accumulated_mean'],
                weights_dict['Generator/ScaledCrossReplicaBNbn/accumulated_var'])
        copy_conv(self.conv_2d, weights_dict['Generator/conv_2d/w/ema_b999900'],
                  weights_dict['Generator/conv_2d/b/ema_b999900'],
                  weights_dict['Generator/conv_2d/u0'])
