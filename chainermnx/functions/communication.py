from abc import ABC

import chainer
from chainer import backend


class AllReduce(chainer.Function):
    def __init__(self, comm):
        self.comm = comm

    def forward(self, inputs):
        x, = inputs
        xs = self.comm.allreduce(x)
        return xs,

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        gx, = grad_outputs
        gxs = self.comm.allgather(gx)
        gxs = xp.stack(gxs)
        return gxs


def allreduce(comm, x):
    # In forward pass, do an all reduce on y_p
    # On backward pass do an all gather on input gradients
    return AllReduce(comm)(x)
