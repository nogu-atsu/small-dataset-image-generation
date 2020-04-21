import numpy
import chainer.functions as F
from source.links._conditional_batch_normalization import ConditionalBatchNormalization
from source.links.sn_linear import SNLinear


class HyperBatchNormalization(ConditionalBatchNormalization):

    def __init__(self, in_size, out_size, decay=0.9, eps=2e-5, dtype=numpy.float32, use_moving_average=True):
        super(HyperBatchNormalization, self).__init__(
            size=out_size, decay=decay, eps=eps, dtype=dtype, use_moving_average=use_moving_average)

        with self.init_scope():
            self.linear_gamma = SNLinear(in_size, out_size, nobias=True)
            self.linear_beta = SNLinear(in_size, out_size, nobias=True)

    def __call__(self, x, z, c, finetune=False, **kwargs):
        gamma_c = self.linear_gamma(F.concat([z, c])) + 1
        beta_c = self.linear_beta(F.concat([z, c]))
        return super(HyperBatchNormalization, self).__call__(x, gamma_c, beta_c, **kwargs)

    def start_finetuning(self):
        self.N = 0
