import chainer
import chainermn
import numpy as np
import chainer.functions as F


class FilterParallelFC(chainer.links.Linear):
    def __init__(self, comm, in_size, out_size):
        self.comm = comm
        self.in_size = in_size
        self.out_size = out_size
        indices = np.arange(self.out_size)
        indices = indices[indices % self.comm.size == 0] + self.comm.rank
        self.out_indices = [i for i in indices if i < self.out_size]
        self.new_out_size = len(self.out_indices)
        super(FilterParallelFC, self).__init__(self.in_size, self.new_out_size)

    def __call__(self, x):
        y = super(FilterParallelFC, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        return F.concat(ys, axis=1)

