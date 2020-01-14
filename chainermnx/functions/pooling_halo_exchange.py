from abc import ABC
from chainer import FunctionNode
import cupy as cp


class HaloExchangePooling(FunctionNode, ABC):
    def __init__(self, comm, k_size, index):
        self.comm = comm
        self.index = index
        self.k_size = k_size
        if (self.k_size % 2) == 0:
            self.halo_size = self.k_size // 2
        else:
            self.halo_size = (self.k_size - 1) // 2

    def forward(self, inputs):
        x, = inputs

        # pad the top and bottom part
        if self.comm.rank == 0:
            npad = ((0, 0), (0, 0), (self.halo_size, 0), (0, 0))
            x = cp.pad(x, pad_width=npad, mode="constant")

        if self.comm.rank == 3:
            npad = ((0, 0), (0, 0), (0, self.halo_size), (0, 0))
            x = cp.pad(x, pad_width=npad, mode="constant")

        lower_halo_region = x[:, :, -self.halo_size:, :]
        upper_halo_region = x[:, :, :self.halo_size, :]
        # Exchange the lower region first
        if self.comm.rank < 3:
            self.comm.send(lower_halo_region, self.comm.rank + 1, (self.comm.rank + 1) * self.index)

        if self.comm.rank > 0:
            received_halo_region = self.comm.recv(self.comm.rank - 1, self.comm.rank * self.index)
            x = cp.concatenate((received_halo_region, x), axis=-2)

        # Exchange the upper region
        if self.comm.rank > 0:
            self.comm.send(upper_halo_region, self.comm.rank - 1, (self.comm.rank - 1) * self.index * 2)

        if self.comm.rank < 3:
            received_halo_region = self.comm.recv(self.comm.rank + 1, self.comm.rank * self.index * 2)
            x = cp.concatenate((x, received_halo_region), axis=-2)

        return x,

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        end = (gy.shape)[-2] - self.halo_size
        gy = gy[:, :, self.halo_size:end, :]

        return gy,


def pooling_halo_exchange(comm, x, k_size, index):
    func = HaloExchangePooling(comm=comm, k_size=k_size, index=index)
    return func.apply((x,))[0]


