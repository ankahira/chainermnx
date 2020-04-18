import chainer
# import cupy as cp
import numpy as np
from chainermnx.functions import allreduce

import time


class ChannelParallelConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.out_channels = out_channels
        self.in_channels = in_channels
        start = time.time()
        if self.comm.size > self.in_channels:
            raise ValueError("Input channels must  be greater or equal to Ranks")
        else:
            indices = np.arange(in_channels)
            indices = indices[indices % self.comm.size == 0] + self.comm.rank
            self._channel_indices = [i for i in indices if i < self.in_channels]
            self.new_in_channels = len(self._channel_indices)
            super(ChannelParallelConvolution2D, self).__init__(self.new_in_channels, self.out_channels, *args, **kwargs)

        stop = time.time()

    def __call__(self, x, *args, **kwargs):
        # Each process gets C/P channels
        x = x[:, self._channel_indices, :, :]
        y = super(ChannelParallelConvolution2D, self).__call__(x)
        print("Shape before allreduce", y.shape)
        ys = allreduce(self.comm, y)

        return ys



## the all reduce function is implemented to facilate all gather in back prop.

