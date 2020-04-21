import numpy as np
from scipy.stats import truncnorm
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Variable, optimizers, cuda, initializers
import chainermn
from PIL import Image

from gen_models.biggan import BIGGAN
from gen_models.sngan import SNGAN


def backward(loss):
    loss.backward()
    return loss.array


class AdaGenerator(chainer.Chain):
    def __init__(self, **params):
        super(AdaGenerator, self).__init__(**params)

    def forward(self, z):
        raise NotImplementedError()

    def __call__(self, perm, eps=0.01, return_z=False):
        xp = self.xp
        z = self.z.W[perm]
        z_eps = xp.random.normal(0, eps, size=z.shape).astype("float32")
        z = z + z_eps
        #
        h = self.forward(z)
        if return_z:
            return h, z
        return h

    def random(self, tmp=0.5, n=5, truncate=False):

        xp = self.xp
        if truncate:
            z = truncnorm(-tmp, tmp).rvs(n * self.dim_z).astype("float32").reshape(n, self.dim_z)
            z = xp.array(z)
        else:
            z = xp.random.normal(0, tmp, size=(n, self.dim_z)).astype("float32")

        h = self.forward(z)
        return h

    def interpolate(self, source=0, dest=1, num=5):
        xp = self.xp
        z = self.z.W[source] * xp.linspace(1, 0, num)[:, None] + self.z.W[dest] * xp.linspace(0, 1, num)[:, None]
        h = self.forward(z)
        return h

    def setup_optimizer(self, alpha=0.0005):
        if self.comm is None:
            self.optimizer = optimizers.Adam(alpha)
        else:
            self.optimizer = chainermn.create_multi_node_optimizer(optimizers.Adam(alpha), self.comm)
        self.optimizer.setup(self)

    def get_loss(self, perm, vgg, target, dis=None, layers=[]):
        xp = self.xp
        target_ = target[perm]
        x, z = self(perm, return_z=True)
        losses = []

        loss = F.mean_absolute_error(x, target_)
        if self.config.normalize_l1_loss:
            losses.append(backward(loss / loss.array))
        else:
            losses.append(backward(loss))

        if vgg is not None:
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                mean = xp.array([103.939, 116.779, 123.68], dtype="float32")[None, :, None, None]
                target_features = vgg((target_ + 1) * 127.5 - mean, layers=layers)

            mean = xp.array([103.939, 116.779, 123.68], dtype="float32")[None, :, None, None]
            vgg_features = vgg((x + 1) * 127.5 - mean, layers=layers)

            for layer in layers:
                if self.config.perceptual_type == "l1":
                    loss = F.mean_absolute_error(target_features[layer], vgg_features[layer])
                elif self.config.perceptual_type == "l2":
                    loss = F.mean_squared_error(target_features[layer], vgg_features[layer])
                l_per = 1 / loss.array / 10 if self.l_per < 0 or self.l_per > 1000 else self.l_per
                losses.append(backward(loss * l_per))
                chainer.reporter.report({f'loss_{layer}': loss})

        if self.l_emd > 0:
            losses.append(backward(self.EMD(self.z.W[perm]) * self.l_emd))

        if self.l_re > 0:
            losses.append(backward(self.bs_reg() * self.l_re))

        if self.l_patch_dis > 0:
            losses.append(backward(self.patch_loss_gen(dis) * self.l_patch_dis))

        if self.l_gp > 0:
            losses.append(backward(self.gp_loss(x, z) * self.l_gp))
        return x, losses

    def patch_loss_gen(self, dis):
        fake = self.random(1, n=self.config.batchsize[self.config.gan_type])
        dis_fake = dis(fake)
        if self.config.loss_type == "wgan-gp":
            loss = F.mean(dis_fake)
        elif self.config.loss_type == "nsgan":
            loss = F.sigmoid_cross_entropy(dis_fake, self.xp.ones(dis_fake.shape, dtype="int32"))
        else:
            assert "loss type: {self.config.loss_type} is not supported"
        return loss

    def patch_loss_dis(self, real, dis, n_dis=5):
        for _ in range(n_dis):
            fake = self.random(n=self.config.batchsize[self.config.gan_type]).array
            if not isinstance(real, Variable):
                real = Variable(real)
            dis_real = dis(real)
            dis_fake = dis(fake)
            if self.config.loss_type == "wgan-gp":
                loss = F.mean(dis_real) - F.mean(dis_fake)
                loss += F.mean(
                    F.batch_l2_norm_squared(chainer.grad([loss], [real], enable_double_backprop=True)[0])) * 1000
            elif self.config.loss_type == "nsgan":
                loss = F.sigmoid_cross_entropy(dis_real, self.xp.ones(dis_real.shape, dtype="int32")) + \
                       F.sigmoid_cross_entropy(dis_fake, self.xp.zeros(dis_fake.shape, dtype="int32"))
                loss += F.mean(
                    F.batch_l2_norm_squared(chainer.grad([loss], [real], enable_double_backprop=True)[0])) * 1000
            else:
                assert "loss type: {self.config.loss_type} is not supported"
        return loss

    def gp_loss(self, x, z):
        h = F.mean(x) / x.shape[0]
        grad, = chainer.grad([h], [z], enable_double_backprop=True)
        return F.mean(F.batch_l2_norm_squared(grad))

    def bs_reg(self):
        bs_re = F.mean(F.square(self.linear.W))
        return bs_re

    def EMD(self, z):
        """
        earth mover distance between z and standard normal
        :param z:
        :return:
        """
        xp = cuda.get_array_module(z)
        dim_z = z.shape[1]
        n = z.shape[0]
        t = xp.random.normal(size=(n * 10, dim_z)).astype("float32")
        dot = F.matmul(z, t, transb=True)
        dist = F.sum(z ** 2, axis=1, keepdims=True) - 2 * dot + xp.sum(t ** 2, axis=1)
        return F.mean(F.min(dist, axis=0)) + F.mean(F.min(dist, axis=1))

    def train_one(self, iter, target, vgg, return_image=False, dis=None, layers=[]):

        perm = np.array(iter.next())

        self.cleargrads()
        sum_loss = 0
        xs = []
        x, losses = self.get_loss(perm, vgg, target, dis=dis, layers=layers)
        for loss in losses:
            sum_loss += loss
        if return_image:
            xs.append(cuda.to_cpu(x.array))
        self.optimizer.update()

        if self.config.l_patch_dis > 0:
            dis.cleargrads()
            real = target[np.random.choice(target.shape[0], self.config.batchsize, replace=False)]
            dis_loss = backward(self.patch_loss_dis(real, dis))
            dis.optimizer.update()

        if return_image:
            return xs, sum_loss / target.shape[0]
        else:
            return None, sum_loss / target.shape[0]

    def evaluation(self, test_image_folder):

        @chainer.training.make_extension()
        def evaluation(trainer):
            with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
                iteration = trainer.updater.iteration
                tmp = self.config.tmp_for_test
                xs = []
                for i in range(5):
                    x = self.random(tmp=tmp, truncate=True)
                    xs.append(chainer.cuda.to_cpu(x.data))

                xs = np.concatenate(xs)
                xs = np.asarray(np.clip(xs * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)

                _, _, h, w = xs.shape
                xs = xs.reshape((5, 5, 3, h, w))
                xs = xs.transpose(0, 3, 1, 4, 2)
                xs = xs.reshape((5 * h, 5 * w, 3))
                Image.fromarray(xs).save(f"{test_image_folder}/{iteration}_random_{tmp}.jpg")

                xs = []
                for i in range(5):
                    x = self(perm=np.arange(i * 5, i * 5 + 5))
                    xs.append(chainer.cuda.to_cpu(x.data))

                xs = np.concatenate(xs)
                xs = np.asarray(np.clip(xs * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)

                _, _, h, w = xs.shape
                xs = xs.reshape((5, 5, 3, h, w))
                xs = xs.transpose(0, 3, 1, 4, 2)
                xs = xs.reshape((5 * h, 5 * w, 3))
                Image.fromarray(xs).save(f"{test_image_folder}/{iteration}_recon_{tmp}.jpg")

                xs = []
                for i in range(5):
                    x = self.interpolate(i, i + 5, num=10)
                    xs.append(chainer.cuda.to_cpu(x.data))

                xs = np.concatenate(xs)
                xs = np.asarray(np.clip(xs * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)

                _, _, h, w = xs.shape
                xs = xs.reshape((5, 10, 3, h, w))
                xs = xs.transpose(0, 3, 1, 4, 2)
                xs = xs.reshape((5 * h, 10 * w, 3))
                Image.fromarray(xs).save(f"{test_image_folder}/{iteration}_interpolate_{tmp}.jpg")

        return evaluation


class AdaBIGGAN(AdaGenerator):
    def __init__(self, config, batchsize, comm=None, test=False):
        if not test:
            self.l_emd = config.l_emd
            self.l_re = config.l_re
            self.l_patch_dis = config.l_patch_dis
            self.l_gp = config.l_gp
            self.l_per = config.l_per
            self.config = config
            self.comm = comm

        self.gen = BIGGAN()
        self.dim_z = 140
        params = {}
        if config.initial_z == "zero":
            params["z"] = L.Parameter(np.zeros((batchsize, self.dim_z)).astype("float32"))
        elif config.initial_z == "random":
            params["z"] = L.Parameter(np.random.normal(size=(batchsize, self.dim_z)).astype("float32"))

        initialW = initializers.HeNormal(0.001 ** 0.5)
        params["linear"] = L.Linear(1, 128, initialW=initialW, nobias=True)
        params["BSA_linear"] = L.Scale(W_shape=(16 * self.gen.ch), bias_term=True)
        for i, (k, l) in enumerate(self.gen.namedlinks()):
            if "Hyper" in k.split("/")[-1]:
                params[f"hyper_bn{i}"] = l
        if config.lr_g_linear > 0:
            params["g_linear"] = self.gen.G_linear
        super(AdaBIGGAN, self).__init__(**params)
        if not test:
            self.setup_optimizer()
            self.z.W.update_rule.hyperparam.alpha = 0.05
            self.linear.W.update_rule.hyperparam.alpha = 0.001
            if config.lr_g_linear > 0:
                self.g_linear.W.update_rule.hyperparam.alpha = config.lr_g_linear

    def forward(self, z):
        xp = self.xp
        c = xp.ones((z.shape[0], 1)).astype("float32")
        c = self.linear(c)
        z = F.split_axis(z, 7, axis=1)
        h = self.gen.G_linear(z[0]).reshape(-1, 4, 4, 16 * self.gen.ch).transpose(0, 3, 1, 2)
        h = self.BSA_linear(h)
        h = self.gen.GBlock(h, z[1], c)
        h = self.gen.GBlock_1(h, z[2], c)
        h = self.gen.GBlock_2(h, z[3], c)
        h = self.gen.GBlock_3(h, z[4], c)
        h = self.gen.GBlock_4(h, z[5], c)
        h = self.gen.attention(h)
        h = self.gen.GBlock_5(h, z[6], c)
        h = F.relu(self.gen.ScaledCrossReplicaBN(h))
        h = F.tanh(self.gen.conv_2d(h))
        return h


class AdaSNGAN(AdaGenerator):
    def __init__(self, config, batchsize, comm=None, test=False):
        if not test:
            self.l_emd = config.l_emd
            self.l_re = config.l_re
            self.l_patch_dis = config.l_patch_dis
            self.l_gp = config.l_gp
            self.l_per = config.l_per
            self.comm = comm
        self.config = config
        self.normalize_stat = self.config.normalize_stat if hasattr(self.config, "normalize_stat") else False
        if self.normalize_stat:
            self.l_re = 0

        self.batchsize = batchsize

        self.gen = SNGAN(n_classes=config.n_classes, normalize_stat=self.normalize_stat)
        self.gen.initialize_params()  # initialize gamma and beta

        self.dim_z = 128
        params = {}
        if config.initial_z == "zero":
            params["z"] = L.Parameter(np.zeros((batchsize, self.dim_z)).astype("float32"))
        elif config.initial_z == "random":
            params["z"] = L.Parameter(np.random.normal(size=(batchsize, self.dim_z)).astype("float32"))

        gamma = self.gen.get_gamma()
        beta = self.gen.get_beta()

        for i in range(len(gamma)):
            if not config.not_initial_gamma:
                params[f"gamma{i + 1}"] = gamma[i]
            if not config.not_initial_beta:
                params[f"beta{i + 1}"] = beta[i]
        # get variables of parameters
        self.gamma = [g.W for g in gamma]
        self.beta = [b.W for b in beta]

        if config.lr_g_linear > 0:
            params["g_linear"] = self.gen.l1
        super(AdaSNGAN, self).__init__(**params)
        if not test:
            self.setup_optimizer(config.init_lr)
            if config.lr_g_linear > 0:
                self.g_linear.W.update_rule.hyperparam.alpha = config.lr_g_linear

    def forward(self, z):
        h = self.gen(z.shape[0], z=z, gamma=self.gamma, beta=self.beta)
        return h

    def bs_reg(self):
        xp = self.xp
        bs_re = 0
        for g in self.gamma:
            bs_re += F.mean_squared_error(g, xp.ones(g.shape, dtype="float32"))
        for b in self.beta:
            bs_re += F.mean_squared_error(b, xp.zeros(b.shape, dtype="float32"))

        return bs_re
