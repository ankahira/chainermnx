import chainer
import chainermn
import chainermnx
import numpy as np
import chainer.functions as F
import chainermnx.functions as FX


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
        ys = chainermn.functions.allgather(self.comm, y) # The mpi allgather
        # ys = FX.allgather(self.comm, y) # MPI allgther but from chainermnx for debuging
        # ys = FX.filter_allgather(self.comm, y) # NCCL Allgather
        return F.concat(ys, axis=1)


