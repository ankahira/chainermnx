import chainer
import numpy as np
from chainermnx.functions import allreduce


class ChannelParallelFC(chainer.links.Linear):
    def __init__(self, comm, in_size, out_size):
        self.comm = comm
        self.in_size = in_size
        if self.in_size is None:
            self.out_size = out_size
            self.first_fc = True
            super(ChannelParallelFC, self).__init__(None, self.out_size)
        else:
            self.first_fc = False
            self.comm = comm
            self.in_size = in_size
            self.out_size = out_size
            indices = np.arange(self.in_size, dtype=int)
            indices = indices[indices % self.comm.size == 0] + self.comm.rank
            self._channel_indices = [i for i in indices if i < self.in_size]
            self.new_in_channels = len(self._channel_indices)
            super(ChannelParallelFC, self).__init__(self.new_in_channels, self.out_size)

    def __call__(self, x):

        if self.first_fc:
            in_channels = x.shape[1]
            indices = np.arange(in_channels)
            indices = indices[indices % self.comm.size == 0] + self.comm.rank
            _channel_indices = [i for i in indices if i < in_channels]
            x = x[:, _channel_indices, :, :]
            y = super(ChannelParallelFC, self).__call__(x)
            ys = allreduce(self.comm, y)
            return ys
        else:
            x = x[:, self._channel_indices]
            y = super(ChannelParallelFC, self).__call__(x)
            ys = allreduce(self.comm, y)
            return ys


