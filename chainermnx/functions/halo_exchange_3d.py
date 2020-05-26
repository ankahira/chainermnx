import os
from abc import ABC
from chainer import FunctionNode
import cupy as cp

import time
import torch

class HaloExchange3D(FunctionNode, ABC):
    def __init__(self,  original_comm, local_comm, k_size, index, pad, out):
        self.comm = local_comm
        self.original_comm = original_comm
        self.index = index
        self.k_size = k_size
        self.pad = pad
        self.out = out
        self.forward_halo_exchange_time_file = open(os.path.join(self.out, "forward_halo_exchange_time.log"), "a", buffering=1)
        if (self.k_size % 2) == 0:
            self.halo_size = self.k_size // 2
        else:
            self.halo_size = (self.k_size - 1) // 2

    def forward(self, inputs):
        x, = inputs
        start = time.time()
        if self.comm is None:
            return x,

        elif self.halo_size == 0:
            return x,
        else:
            start = time.time()
            # pad the top and bottom part
            # in the case where there is padding, removing the halo region is done the same since the pad is equal to
            # halo regions . regions takes place in all ranks
            # In the case where there is no padding, be careful to remove the halos from just the top and bottom part
            # ranks 0 and 3

            # This padding thing is not working. Much detailed attention is required to find the right dimensions to pad
            if self.pad != 0:
                if self.comm.rank == 0:
                    npad = ((0, 0), (0, 0), (self.pad, 0), (0, 0))
                    x = cp.pad(x, pad_width=npad, mode="constant")

                if self.comm.rank == 3:
                    npad = ((0, 0), (0, 0), (0, self.pad), (0, 0))
                    x = cp.pad(x, pad_width=npad, mode="constant")

            lower_halo_region = x[:, :, :, -self.halo_size:, :]
            upper_halo_region = x[:, :, :, :self.halo_size, :]
            # Exchange the lower region first
            torch.cuda.synchronize()
            if self.comm.rank < 3:
                self.comm.send(lower_halo_region, self.comm.rank + 1, (self.comm.rank + 1) * self.index)

            if self.comm.rank > 0:
                received_halo_region = self.comm.recv(self.comm.rank - 1, self.comm.rank * self.index)
                x = cp.concatenate((received_halo_region, x), axis=-2)

            # Exchange the upper region
            torch.cuda.synchronize()
            if self.comm.rank > 0:
                self.comm.send(upper_halo_region, self.comm.rank - 1, (self.comm.rank - 1) * self.index * 2)

            if self.comm.rank < 3:
                received_halo_region = self.comm.recv(self.comm.rank + 1, self.comm.rank * self.index * 2)
                x = cp.concatenate((x, received_halo_region), axis=-2)

            
            torch.cuda.synchronize()
		stop = time.time()

        if self.original_comm.rank == 0:
            print("{:.10f}".format(stop - start), "\t",  self.index,  file=self.forward_halo_exchange_time_file)


        return x,

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        if self.halo_size != 0:
            if self.pad == 0:
                # For cases where pad was zero
                if self.comm.rank == 0:
                    end = gy.shape[-2] - self.halo_size
                    gy = gy[:, :, :, :end, :]
                if self.comm.rank == 3:
                    gy = gy[:, :, :, self.halo_size:, :]
                if self.comm.rank == 1:
                    end = gy.shape[-2] - self.halo_size
                    gy = gy[:, :, :, self.halo_size:end, :]
                if self.comm.rank == 2:
                    end = gy.shape[-2] - self.halo_size
                    gy = gy[:, :, :, self.halo_size:end, :]
            else:
                end = gy.shape[-2] - self.halo_size
                gy = gy[:, :, :, self.halo_size:end, :]

        return gy,


def halo_exchange_3d(original_comm, local_comm,  x, k_size, index, pad, out):
    func = HaloExchange3D(original_comm=original_comm, local_comm=local_comm, k_size=k_size, index=index, pad=pad, out=out)
    return func.apply((x,))[0]


