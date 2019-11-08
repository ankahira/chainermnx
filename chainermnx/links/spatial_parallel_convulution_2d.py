import chainer
import chainermn
import chainer.functions as F
from chainermnx.functions import concat
import cupy as cp


class SpatialConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, conv_index, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.index = conv_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_proc = self.comm.size
        self.i_proc = self.comm.rank
        self.k_size = args[0]
        if (self.k_size % 2) == 0:
            self.halo_size = self.k_size // 2
        else:
            self.halo_size = (self.k_size - 1) // 2

        super(SpatialConvolution2D, self).__init__(self.in_channels, self.out_channels, *args, **kwargs)

    def __call__(self, x):
        # npad = ((0, 0), (0, 0), (1, 1), (0, 0))
        # x = cp.pad(x, pad_width=npad, mode="constant")
        if self.comm.rank == 0:
            # npad = ((0, 0), (0, 0), (0, 0), (self.halo_size, 0))
            # x = chainer.functions.pad(x, pad_width=npad, mode="constant")

            if hasattr(x, "array"):
                npad = ((0, 0), (0, 0), (0, 0), (self.halo_size, 0))
                x.array = cp.pad(x.array, pad_width=npad, mode="constant")
            else:
                npad = ((0, 0), (0, 0), (0, 0), (self.halo_size, 0))
                x = cp.pad(x, pad_width=npad, mode="constant")

        if self.comm.rank == 3:
            # npad = ((0, 0), (0, 0), (0, 0), (0, self.halo_size))
            # x = chainer.functions.pad(x, pad_width=npad, mode="constant")
            if hasattr(x, "array"):
                npad = ((0, 0), (0, 0), (0, 0), (0, self.halo_size))
                x.array = cp.pad(x.array, pad_width=npad, mode="constant")

            else:
                npad = ((0, 0), (0, 0), (0, 0), (0, self.halo_size))
                x = cp.pad(x, pad_width=npad, mode="constant")

        x = self.halo_exchange_forward(x)
        x = self.halo_exchange_backward(x)
        y = super(SpatialConvolution2D, self).__call__(x)
        return y

    def halo_exchange_forward(self, x):
        halo_region_send = x[:, :, :, -self.halo_size:]
        if self.comm.rank < 3:
            sent = chainermn.functions.send(
                halo_region_send,
                self.comm,
                rank=self.comm.rank + 1,
                tag=(self.comm.rank + 1) * self.index)

        if self.comm.rank > 0:
            received_halo_region = chainermn.functions.recv(
                self.comm,
                rank=self.comm.rank - 1,
                tag=self.comm.rank * self.index)
            x = concat((x, received_halo_region), axis=-1)

        return x

    def halo_exchange_backward(self, x):
        halo_region_send = x[:, :, :, :self.halo_size]
        if self.comm.rank > 0:
            sent = chainermn.functions.send(
                halo_region_send,
                self.comm,
                rank=self.comm.rank - 1,
                tag=(self.comm.rank - 1) * self.index * 2)
        if self.comm.rank < 3:
            received_halo_region = chainermn.functions.recv(
                self.comm,
                rank=self.comm.rank + 1,
                tag=self.comm.rank * self.index * 2)

            x = concat((received_halo_region, x), axis=-1)

        return x


class SpatialConvolution2DFinal(chainer.links.Convolution2D):
    def __init__(self, comm, conv_index, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.index = conv_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_proc = self.comm.size
        self.i_proc = self.comm.rank
        self.k_size = args[0]
        if (self.k_size % 2) == 0:
            self.halo_size = self.k_size // 2
        else:
            self.halo_size = (self.k_size - 1) // 2

        super(SpatialConvolution2DFinal, self).__init__(self.in_channels, self.out_channels, *args, **kwargs)

    def __call__(self, x):
        # npad = ((0, 0), (0, 0), (1, 1), (0, 0))
        # x = cp.pad(x, pad_width=npad, mode="constant")
        if self.comm.rank == 0:
            # npad = ((0, 0), (0, 0), (0, 0), (self.halo_size, 0))
            # x = chainer.functions.pad(x, pad_width=npad, mode="constant")

            if hasattr(x, "array"):
                npad = ((0, 0), (0, 0), (0, 0), (self.halo_size, 0))
                x.array = cp.pad(x.array, pad_width=npad, mode="constant")
            else:
                npad = ((0, 0), (0, 0), (0, 0), (self.halo_size, 0))
                x = cp.pad(x, pad_width=npad, mode="constant")

        if self.comm.rank == 3:
            # npad = ((0, 0), (0, 0), (0, 0), (0, self.halo_size))
            # x = chainer.functions.pad(x, pad_width=npad, mode="constant")
            if hasattr(x, "array"):
                npad = ((0, 0), (0, 0), (0, 0), (0, self.halo_size))
                x.array = cp.pad(x.array, pad_width=npad, mode="constant")

            else:
                npad = ((0, 0), (0, 0), (0, 0), (0, self.halo_size))
                x = cp.pad(x, pad_width=npad, mode="constant")

        x = self.halo_exchange_forward(x)
        x = self.halo_exchange_backward(x)
        ys = chainermn.functions.allgather(self.comm, x)
        return F.concat(ys, axis=-1)

    def halo_exchange_forward(self, x):
        halo_region_send = x[:, :, :, -self.halo_size:]
        if self.comm.rank < 3:
            sent = chainermn.functions.send(
                halo_region_send,
                self.comm,
                rank=self.comm.rank + 1,
                tag=(self.comm.rank + 1) * self.index)

        if self.comm.rank > 0:
            received_halo_region = chainermn.functions.recv(
                self.comm,
                rank=self.comm.rank - 1,
                tag=self.comm.rank * self.index)
            x = concat((x, received_halo_region), axis=-1)

        return x

    def halo_exchange_backward(self, x):
        halo_region_send = x[:, :, :, :self.halo_size]
        if self.comm.rank > 0:
            sent = chainermn.functions.send(
                halo_region_send,
                self.comm,
                rank=self.comm.rank - 1,
                tag=(self.comm.rank - 1) * self.index * 2)
        if self.comm.rank < 3:
            received_halo_region = chainermn.functions.recv(
                self.comm,
                rank=self.comm.rank + 1,
                tag=self.comm.rank * self.index * 2)

            x = concat((received_halo_region, x), axis=-1)

        return x
