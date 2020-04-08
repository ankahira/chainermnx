from abc import ABC

import chainer
from chainer import backend
import time


class AllReduce(chainer.Function, ABC):
    def __init__(self, comm, index):
        self.comm = comm
        self.index = index

    def forward(self, inputs):
        x, = inputs
        start = time.time()
        xs = self.comm.allreduce(x)
        stop = time.time()
        # if self.comm.rank ==0:
        #     print("Layer Number", self.index, "Time for Forward ", stop - start)
        return xs,

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        gx, = grad_outputs
        start = time.time()
        gxs = self.comm.allgather(gx)
        stop = time.time()
        # gxs = xp.stack(gxs)
        # if self.comm.rank == 0:
        #     print("Layer Number", self.index, "Time for Backward ", stop - start)
        return gxs


def allreduce(comm, x, index):
    # In forward pass, do an all reduce on y_p
    # On backward pass do an all gather on input gradients
    return AllReduce(comm,  index)(x)
