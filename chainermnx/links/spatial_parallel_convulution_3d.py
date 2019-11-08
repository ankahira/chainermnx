import chainer
import chainermn
import chainer.functions as F
from chainermnx.functions import concat
import time


class SpatialConvolution3D(chainer.links.Convolution3D):
    def __init__(self, comm, conv_index, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.index = conv_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_proc = self.comm.size
        self.i_proc = self.comm.rank
        self.k_size = args[0]
        if (self.k_size % 2) == 0:
            self.halo_size = self.k_size//2
        else:
            self.halo_size = (self.k_size-1) // 2

        super(SpatialConvolution3D, self).__init__(self.in_channels, self.out_channels, *args, **kwargs)

    def __call__(self, x):
        x = self.halo_exchange_forward(x)
        # x = self.halo_exchange_backward(x)
        y = super(SpatialConvolution3D, self).__call__(x)
        if self.comm.rank > 0 and self.index != 1:
            y = y[:, :, :, :-self.halo_size]
            return y
            # print("Rank is ", self.comm.rank, "Shape of X", x.shape, "Shape of Y", y.shape)
        else:
            return y

    def halo_exchange_forward(self, x):
        halo_region_send = x[:, :, :, -self.halo_size:]
        if self.comm.rank < 3 :
            chainermn.functions.send(halo_region_send, self.comm, rank=self.comm.rank + 1, tag=(self.comm.rank +1)* self.index)
        if self.comm.rank > 0 :
            received_halo_region = chainermn.functions.recv(self.comm, rank=self.comm.rank -1, tag=self.comm.rank*self.index)
            x = concat((x, received_halo_region), axis=-1)

        return x

    def halo_exchange_backward(self, x):
        halo_region_send = x[:, :, :, :self.halo_size]
        if self.comm.rank > 0:
            chainermn.functions.send(halo_region_send, dest=self.comm.rank - 1, tag=(self.comm.rank - 1) * self.index * 2)
        if self.comm.rank < 3:
            received_halo_region = self.comm.mpi_comm.recv(source=self.comm.rank + 1, tag=self.comm.rank * self.index * 2)
            x = F.concat((received_halo_region, x), axis=-1)

        return x





