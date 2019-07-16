import chainer
import numpy as np


def backward(loss):
    loss.backward()
    return loss.array


def exponential_shift(nets=[], scale=0.1):
    for net in nets:
        if net is not None:
            for p in net.params():
                p.update_rule.hyperparam.alpha = p.update_rule.hyperparam.alpha * scale


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config')
        self.gen = kwargs.pop('gen')
        self.dis = kwargs.pop('dis')
        self.target = kwargs.pop('target')
        self.vgg = kwargs.pop('vgg')
        self.layers = kwargs.pop('layers')

        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        perm = np.array(self.get_iterator('main').next())

        if self.config.exponential_shift_interval > 0 and \
                (self.iteration + 1) % self.config.exponential_shift_interval == 0:
            exponential_shift([self.gen, self.dis], scale=self.config.lr_scale)

        with chainer.using_config('train', self.config.train):
            self.gen.cleargrads()
            gen_loss = 0
            x, losses = self.gen.get_loss(perm, self.vgg, self.target, dis=self.dis, layers=self.layers)
            for loss in losses:
                gen_loss += loss

            optimizer_gen = self.get_optimizer('opt_gen')
            optimizer_gen.update()

            chainer.reporter.report({'loss_gen': gen_loss})

            if self.config.l_patch_dis > 0:
                optimizer_dis = self.get_optimizer('opt_dis')
                self.dis.cleargrads()
                real = self.target[np.random.choice(self.target.shape[0], self.config.batchsize[self.config.gan_type],
                                                    replace=False)]
                dis_loss = backward(self.gen.patch_loss_dis(real, self.dis))
                optimizer_dis.update()
                chainer.reporter.report({'loss_dis': dis_loss})
