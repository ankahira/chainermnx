import math
from abc import ABC
import numpy as np
import chainer
import chainermn
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainermnx.functions import concat
import cupy as cp


class HaloExchange(FunctionNode, ABC):
    def __init__(self, comm, k_size, index, pad):
        self.comm = comm
        self.index = index
        self.k_size = k_size
        self.pad = pad
        if (self.k_size % 2) == 0:
            self.halo_size = self.k_size // 2
        else:
            self.halo_size = (self.k_size - 1) // 2

    def forward(self, inputs):
        x, = inputs

        # if self.comm is None:
        #     return x,
        # else:
        #     # pad the top and bottom part
        #     if self.pad != 0:
        #         if self.comm.rank == 0:
        #             npad = ((0, 0), (0, 0), (self.pad, 0), (0, 0))
        #             x = cp.pad(x, pad_width=npad, mode="constant")
        #
        #         if self.comm.rank == 3:
        #             npad = ((0, 0), (0, 0), (0, self.pad), (0, 0))
        #             x = cp.pad(x, pad_width=npad, mode="constant")
        #
        #     lower_halo_region = x[:, :, -self.halo_size:, :]
        #     upper_halo_region = x[:, :, :self.halo_size, :]
        #     # Exchange the lower region first
        #     if self.comm.rank < 3:
        #         self.comm.send(lower_halo_region, self.comm.rank + 1, (self.comm.rank + 1) * self.index)
        #
        #     if self.comm.rank > 0:
        #         received_halo_region = self.comm.recv(self.comm.rank - 1, self.comm.rank * self.index)
        #         x = cp.concatenate((received_halo_region, x), axis=-2)
        #
        #     # Exchange the upper region
        #     if self.comm.rank > 0:
        #         self.comm.send(upper_halo_region, self.comm.rank - 1, (self.comm.rank - 1) * self.index * 2)
        #
        #     if self.comm.rank < 3:
        #
        #         received_halo_region = self.comm.recv(self.comm.rank + 1, self.comm.rank * self.index * 2)
        #         x = cp.concatenate((x, received_halo_region), axis=-2)

        return x,

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        # end = (gy.shape)[-2] - (self.halo_size )
        # gy = gy[:, :, self.halo_size:end, :]
        return gy,


def halo_exchange(comm, x, k_size, index, pad):
    func = HaloExchange(comm=comm, k_size=k_size, index=index, pad=pad)
    return func.apply((x,))[0]


