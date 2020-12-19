import os
from abc import ABC
from chainer import FunctionNode
import numpy as np

import time
import torch

class HaloExchangePooling(FunctionNode, ABC):
    def __init__(self,  original_comm, local_comm, k_size, index, out):
        self.comm = local_comm
        self.original_comm = original_comm
        self.index = index
        self.k_size = k_size
        self.out = out
        self.forward_halo_exchange_time_file = open(os.path.join(self.out, "pooling_halo_exchange_time.log"), "a",
                                                    buffering=1)
        if (self.k_size % 2) == 0:
            self.halo_size = self.k_size // 2
        else:
            self.halo_size = (self.k_size - 1) // 2

    def forward(self, inputs):
        x, = inputs
        start = time.time()

        # pad the top and bottom part
        if self.comm.rank == 0:
            npad = ((0, 0), (0, 0), (self.halo_size, 0), (0, 0))
            x = np.pad(x, pad_width=npad, mode="constant")

        if self.comm.rank == 3:
            npad = ((0, 0), (0, 0), (0, self.halo_size), (0, 0))
            x = np.pad(x, pad_width=npad, mode="constant")

        lower_halo_region = x[:, :, -self.halo_size:, :]
        upper_halo_region = x[:, :, :self.halo_size, :]
        # Exchange the lower region first
        torch.cuda.synchronize()
        if self.comm.rank < 3:
            self.comm.send(lower_halo_region, self.comm.rank + 1, (self.comm.rank + 1) * self.index)

        if self.comm.rank > 0:
            received_halo_region = self.comm.recv(self.comm.rank - 1, self.comm.rank * self.index)
            x = np.concatenate((received_halo_region, x), axis=-2)

        # Exchange the upper region
        torch.cuda.synchronize()
        if self.comm.rank > 0:
            self.comm.send(upper_halo_region, self.comm.rank - 1, (self.comm.rank - 1) * self.index * 2)

        if self.comm.rank < 3:
            received_halo_region = self.comm.recv(self.comm.rank + 1, self.comm.rank * self.index * 2)
            x = np.concatenate((x, received_halo_region), axis=-2)
        torch.cuda.synchronize()
        stop = time.time()
        if self.original_comm.rank == 0:
            print("{:.10f}".format(stop - start), "\t", self.index, file=self.forward_halo_exchange_time_file)
        return x,

    def backward(self, target_input_indexes, grad_outputs):
        #TODO
        # What do we do in back prop for poolng ?
        gy, = grad_outputs
        end = (gy.shape)[-2] - self.halo_size
        gy = gy[:, :, self.halo_size:end, :]

        return gy,


def pooling_halo_exchange(original_comm, local_comm, x, k_size, index, out):
    func = HaloExchangePooling(original_comm=original_comm, local_comm=local_comm, k_size=k_size, index=index, out=out)
    return func.apply((x,))[0]


