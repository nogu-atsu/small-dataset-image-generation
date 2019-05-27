import chainer
import chainer.links as L
from chainer import functions as F
from gen_models.sngan_resblocks import Block
from source.miscs.random_samples import sample_categorical, sample_continuous


class SNGAN(chainer.Chain):
    # all bn parameters are initialized to 1 and 0
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal",
                 normalize_stat=False):
        super(SNGAN, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.ch = ch
        self.normalize_stat = normalize_stat
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 16, initialW=initializer)
            self.block2 = Block(ch * 16, ch * 16, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block5 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
            self.block6 = Block(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b7 = L.BatchNormalization(ch)
            self.l7 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, batchsize=64, z=None, gamma=None, beta=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if gamma is None:
            gamma = [None] * 12
            beta = [None] * 12
        y = self.xp.zeros((1, self.n_classes), dtype="float32")
        if self.n_classes > 0:
            y[0, 0] = 1
        h = z
        h = self.l1(h)
        h = F.reshape(h, (batchsize, -1, self.bottom_width, self.bottom_width))
        if self.normalize_stat:
            gamma = [self.normalize(g) * 0.2 + 1 for g in gamma]
            beta = [self.normalize(b) * 0.15 for b in beta]

        h = self.shift(h, gamma[0], beta[0])
        h = self.block2(h, y, None, gamma[1:3], beta[1:3], **kwargs)
        h = self.block3(h, y, None, gamma[3:5], beta[3:5], **kwargs)
        h = self.block4(h, y, None, gamma[5:7], beta[5:7], **kwargs)
        h = self.block5(h, y, None, gamma[7:9], beta[7:9], **kwargs)
        h = self.block6(h, y, None, gamma[9:11], beta[9:11], **kwargs)
        h = self.b7(h)
        h = self.activation(h)
        h = F.tanh(self.l7(h))
        return h

    def normalize(self, g):
        mu = F.mean(g)
        sigma = F.sqrt(F.mean(g ** 2) - mu ** 2 + 1e-5)
        return (g - mu) / (sigma + 1e-5)

    def shift(self, x, gamma, beta):
        if gamma is None:
            return x
        batchsize = x.shape[0]
        if gamma.ndim == 1 and beta.ndim == 1:
            x = x * F.tile(gamma[None, :, None, None], (batchsize, 1, self.bottom_width, self.bottom_width)) + \
                F.tile(beta[None, :, None, None], (batchsize, 1, self.bottom_width, self.bottom_width))
        elif gamma.ndim == 2 and beta.ndim == 2:
            x = x * F.tile(gamma[:, :, None, None], (1, self.bottom_width, self.bottom_width)) + \
                F.tile(beta[:, :, None, None], (1, self.bottom_width, self.bottom_width))
        return x

    def get_gamma(self):
        xp = self.xp
        gamma = [L.Parameter(xp.ones(self.ch * i, dtype="float32")) for i in [16, 16, 16, 16, 8, 8, 4, 4, 2, 2, 1]]
        return gamma

    def get_beta(self):
        xp = self.xp
        beta = [L.Parameter(xp.zeros(self.ch * i, dtype="float32")) for i in [16, 16, 16, 16, 8, 8, 4, 4, 2, 2, 1]]
        return beta

    def initialize_params(self, gamma=True, beta=True):
        self.block2.initialize_params(gamma, beta)
        self.block3.initialize_params(gamma, beta)
        self.block4.initialize_params(gamma, beta)
        self.block5.initialize_params(gamma, beta)
        self.block6.initialize_params(gamma, beta)

    def start_finetuning(self):
        self.block2.start_finetuning()
        self.block3.start_finetuning()
        self.block4.start_finetuning()
        self.block5.start_finetuning()
        self.block6.start_finetuning()
        self.b7.start_finetuning()
