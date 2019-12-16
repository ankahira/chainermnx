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


class Checker(FunctionNode, ABC):
    def __init__(self, comm, index,):
        self.comm = comm
        self.index = index

    def forward(self, inputs):
        x, = inputs
        h = x
        if self.comm is None:
            with open('sequential_forward_prop.txt', 'w') as f:
                for i in range(h.shape[-2]):
                    for j in range(h.shape[-1]):
                        print("%01.5f" % h[0, 0, i, j], file=f, end=" ")
                    print("\n", file=f)
        else:
            hs = chainermn.functions.allgather(self.comm, h)
            h = F.concat(hs, -2)
            file_name = "spatial_forward_prop.txt"
            if self.comm.rank == 0:
                with open(file_name, 'w') as f:
                    for i in range(h.shape[-2]):
                        for j in range(h.shape[-1]):
                            print("%01.5f" % h[0, 0, i, j].array, file=f, end=" ")
                        print("\n", file=f)
        return x,

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        if self.comm is None and self.index == 2:
            with open('sequential_back_prop.txt', 'w') as f:
                for i in range(gy.shape[-2]):
                    for j in range(gy.shape[-1]):
                        print("%01.5f" % gy[0, 0, i, j].array, file=f, end=" ")
                    print("\n", file=f)
        else:
            file_name = "spatial_back_prop_rank_{0}.txt".format(self.comm.rank)
            if self.index == 2:
                with open(file_name, 'w') as f:
                    for i in range(gy.shape[-2]):
                        for j in range(gy.shape[-1]):
                            print("%01.5f" % gy[0, 0, i, j].array, file=f, end=" ")
                        print("\n", file=f)

        return gy,


def checker(comm, x,  index):
    func = Checker(comm=comm, index=index)
    return func.apply((x,))[0]


