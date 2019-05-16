# this code is based on https://github.com/pfnet-research/sngan_projection/blob/master/source/links/categorical_conditional_batch_normalization.py
import numpy

from chainer import initializers
from chainer.links import EmbedID
import chainer.functions as F
from source.links.conditional_batch_normalization import ConditionalBatchNormalization


class CategoricalConditionalBatchNormalization(ConditionalBatchNormalization):
    """
    Conditional Batch Normalization
    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        n_cat (int): the number of categories of categorical variable.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`
    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.
    """

    def __init__(self, size, n_cat, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 initial_gamma=None, initial_beta=None):
        super(CategoricalConditionalBatchNormalization, self).__init__(
            size=size, n_cat=n_cat, decay=decay, eps=eps, dtype=dtype)

        with self.init_scope():
            if initial_gamma is None:
                initial_gamma = 1
            initial_gamma = initializers._get_initializer(initial_gamma)
            initial_gamma.dtype = dtype
            self.gammas = EmbedID(n_cat, size, initialW=initial_gamma)
            if initial_beta is None:
                initial_beta = 0
            initial_beta = initializers._get_initializer(initial_beta)
            initial_beta.dtype = dtype
            self.betas = EmbedID(n_cat, size, initialW=initial_beta)

    def __call__(self, x, c, finetune=False, **kwargs):
        """__call__(self, x, c, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluatino during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            c (Variable): Input variable for conditioning gamma and beta
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        if c.ndim == 2:
            gamma_c = F.matmul(c, self.gammas.W)
            beta_c = F.matmul(c, self.betas.W)
        else:
            gamma_c = self.gammas(c)
            beta_c = self.betas(c)
        return super(CategoricalConditionalBatchNormalization, self).__call__(x, gamma_c, beta_c, **kwargs)

    def initialize_params(self, gamma, beta):
        if gamma:
            self.gammas.W.array = self.xp.ones_like(self.gammas.W.array)
        if beta:
            self.betas.W.array = self.xp.zeros_like(self.betas.W.array)

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.
        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.
        """
        self.N = 0
