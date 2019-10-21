import chainer
from chainer.functions.pooling import pooling_nd
from chainer.functions.pooling.max_pooling_nd import MaxPoolingND


def max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    if len(x.shape[2:]) != 2:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 2. But the actual ndim was {}.'.format(len(x.shape[2:])))

    ndim = len(x.shape[2:])

    func = MaxPoolingND(ndim, ksize, stride, pad, cover_all, return_indices)
    if return_indices:
        with chainer.using_config('use_cudnn', 'never'):
            out = func.apply((x,))[0]
        return out, func.indexes

    return func.apply((x,))[0]



