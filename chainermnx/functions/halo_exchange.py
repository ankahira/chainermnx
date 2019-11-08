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
            npad = ((0, 0), (0, 0), (0, 0), (self.halo_size, 0))
            x = cp.pad(x, pad_width=npad, mode="constant")

        if self.comm.rank == 3:
            npad = ((0, 0), (0, 0), (0, 0), (0, self.halo_size))
            x = cp.pad(x, pad_width=npad, mode="constant")

        # Exchange the upper parts first

        halo_region_send = x[:, :, :, -self.halo_size:]
        if self.comm.rank < 3:
            self.comm.send(halo_region_send, self.comm.rank + 1, (self.comm.rank + 1) * self.index)

        if self.comm.rank > 0:
            received_halo_region = self.comm.recv(self.comm.rank - 1, self.comm.rank * self.index)
            x = cp.concatenate((x, received_halo_region), axis=-1)

        # Exchange the lower parts
        halo_region_send = x[:, :, :, :self.halo_size]
        if self.comm.rank > 0:
            self.comm.send(halo_region_send, self.comm.rank - 1, (self.comm.rank - 1) * self.index * 2)

        if self.comm.rank < 3:
            received_halo_region = self.comm.recv(self.comm.rank + 1, self.comm.rank * self.index * 2)

            x = cp.concatenate((received_halo_region, x), axis=-1)

        return x,

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        end = (gy.shape)[-1] - self.halo_size
        gy = gy[:, :, :, self.halo_size: end]
        return gy,


def halo_exchange(comm, x, k_size, index):
    func = HaloExchange(comm=comm, k_size=k_size, index=index)
    return func.apply((x,))[0]


