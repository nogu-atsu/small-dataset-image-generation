import chainer, chainermn
from chainer import functions as F
from chainer import links as L
from chainer import optimizers


class PatchDiscriminator(chainer.Chain):
    # discriminator for patch
    def __init__(self, comm=None):
        self.comm = comm
        super(PatchDiscriminator, self).__init__()
        with self.init_scope():
            self.block1 = ConvBlock(3, 32)
            self.block2 = ConvBlock(32, 64)
            self.block3 = ConvBlock(64, 128)
            self.block4 = ConvBlock(128, 256)
            self.c = L.Convolution2D(256, 1)
        self.setup_optimizer()

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.c(x)
        return x  # bx1x16x16

    def setup_optimizer(self):
        if self.comm is None:
            self.optimizer = optimizers.Adam(0.0005)
        else:
            self.optimizer = chainermn.create_multi_node_optimizer(optimizers.Adam(0.0005), self.comm)
        self.optimizer.setup(self)


class ConvBlock(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(ConvBlock, self).__init__()
        initialW = chainer.initializers.HeNormal(3)
        with self.init_scope():
            self.c1 = L.Convolution2D(in_dim, out_dim, 3, 1, 1, initialW=initialW)
            self.c2 = L.DepthwiseConvolution2D(out_dim, 1, 3, 2, 1, initialW=initialW)

    def __call__(self, x):
        x = F.leaky_relu(self.c1(x))
        x = F.leaky_relu(self.c2(x))
        return x
