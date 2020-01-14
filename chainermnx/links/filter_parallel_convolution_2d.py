import chainer
import chainermn
import numpy as np
import chainer.functions as F


class FilterParallelConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.filters = out_channels
        indices = np.arange(self.filters)
        indices = indices[indices % self.comm.size == 0] + self.comm.rank
        self.filter_indices = [i for i in indices if i < self.filters]
        self.new_filters = len(self.filter_indices)
        super(FilterParallelConvolution2D, self).__init__(self.in_channels, self.new_filters, *args, **kwargs)

    def __call__(self, x):
        y = super(FilterParallelConvolution2D, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        # Backward will be invoked as well as the ordinary chainer functions,
        # where gradients are reduced to each process
        # Here the default chainermn allgather function is used
        return F.concat(ys, axis=1)

