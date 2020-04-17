from abc import ABC
import chainer


class AllReduce(chainer.Function, ABC):
    # this was created to faciliate the allreduce/allgather required in the case of channel parallelism
    def __init__(self, comm):
        self.comm = comm

    def forward(self, inputs):
        x, = inputs
        # xs = cp.empty(x.shape, dtype=cp.float32) # temporary receive buffer. Use when required
        # nccl_allreduce(x, self.comm)
        self.comm.nccl_allreduce(x, self.comm)
        return x,

    def backward(self, inputs, grad_outputs):
        gx, = grad_outputs
        return gx,


def allreduce(x, comm):
    # In forward pass, do an all reduce on y_p
    # On backward pass do an all gather on input gradients
    return AllReduce(comm)(x)
