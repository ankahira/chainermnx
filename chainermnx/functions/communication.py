from abc import ABC
import chainer
import cupy as cp


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


class FilterAllGather(chainer.Function, ABC):
    # For filter parallelism
    def __init__(self, comm):
        self.comm = comm

    def forward(self, inputs):
        # Do all gather in forward
        x, = inputs
        xs = self.comm.nccl_allgather(x, self.comm)
        # xs = cp.concatenate(xs, axis=1)
        return xs

    def backward(self, inputs, grad_outputs):
        # Do allreduce in
        gx = cp.stack(grad_outputs).sum(axis=0)
        # print("Grad outputs", grad_outputs[0].shape, len(grad_outputs))
        # gx, = grad_outputs
        self.comm.nccl_allreduce(gx, self.comm)
        # print("GX", type(gx))
        # if self.comm.rank == 0:
        #     print("Backward shape of gradiets", gx.shape)
        return gx,


def filter_allgather(comm, x):
    return FilterAllGather(comm)(x)

