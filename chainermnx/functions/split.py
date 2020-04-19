from abc import ABC
from chainer import FunctionNode
import cupy as cp
import time


class Split(FunctionNode, ABC):
    def __init__(self, comm):
        self.comm = comm

    def forward(self, inputs):
        x, = inputs
        if x.shape[1] < self.comm.size:
            return x,
        else:
            x_list = cp.array_split(x, self.comm.size, axis=1)
            x = x_list[self.comm.rank]
            return x,

    def backward(self, target_input_indexes, grad_outputs):
        # do backward computation
        # x, = self.get_retained_inputs()
        # Gather the gradients here
        gx, = grad_outputs
        gx_array = gx.array
        # gxs = self.comm.allgather(gx_array)
        gxs = self.comm.nccl_allgather(gx_array, self.comm)
        gxs = cp.concatenate(gxs, axis=1)
        gx.array = gxs
        return gx,


def split(comm, x):
    func = Split(comm=comm)
    # remember if you dont put [0], you run into errors
    return func.apply((x,))[0]


