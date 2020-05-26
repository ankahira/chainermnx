import chainer
import chainermn
import chainermnx
import numpy as np
import chainer.functions as F
import chainermnx.functions as FX

import numpy as np
import cupy as cp

import time
import torch

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
        xp = chainer.backend.get_array_module(x)
        if xp is not np:
            chainer.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        y = super(FilterParallelConvolution2D, self).__call__(x)
        if xp is not np:
            chainer.cuda.Stream.null.synchronize()
        stop1 = time.perf_counter()
        # yys = chainermn.functions.allgather(self.comm, y) # The mpi allgather
        # yys = FX.allgather(self.comm, y) # MPI allgther but from chainermnx for debuging
        ys = FX.filter_allgather(self.comm, y)  # NCCL Allgather
        if xp is not np:
            chainer.cuda.Stream.null.synchronize()
        stop2 = time.perf_counter()
        ys = FX.concat(ys,self.comm,axis=1)
        if xp is not np:
            chainer.cuda.Stream.null.synchronize()
        stop3 = time.perf_counter()
        if self.comm.rank == 0:
            print("{:.10f}".format(stop1-start),"\t{:.10f}".format(stop2-stop1),"\t{:.10f}".format(stop3-stop2))
        
        return ys
        #return F.concat(ys, axis=1)


