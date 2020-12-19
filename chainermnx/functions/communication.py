import os
from abc import ABC
import chainer
import torch
import numpy as np

import time


class AllReduce(chainer.Function, ABC):
    # this was created to faciliate the allreduce/allgather required in the case of channel parallelism
    def __init__(self, comm):
        self.comm = comm

    def forward(self, inputs):
        start = time.time()
        x, = inputs
        # xs = cp.empty(x.shape, dtype=cp.float32) # temporary receive buffer. Use when required
        # nccl_allreduce(x, self.comm)

        # self.comm.nccl_allreduce(x, self.comm) # For GPU
        self.comm.allreduce(x) # For CPU
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
    def __init__(self, original_comm, comm, out):
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        self.allreduce_time_file = open(os.path.join(self.out, "high_level_allreduce_times.log"), "a", buffering=1)
        self.allgather_time_file = open(os.path.join(self.out, "high_level_allgather_times.log"), "a", buffering=1)

    def forward(self, inputs):
        # Do all gather in forward
        x, = inputs
        torch.cuda.synchronize()
        start = time.perf_counter()
        xs = self.comm.nccl_allgather(x, self.comm)
        torch.cuda.synchronize()
        stop = time.perf_counter()
        if self.original_comm.rank == 0:
            print("{:.10f}".format(stop-start), "\t", file=self.allgather_time_file)
        # xs = cp.concatenate(xs, axis=1)
        return xs

    def backward(self, inputs, grad_outputs):
        # Do allreduce in
        gx = np.stack(grad_outputs).sum(axis=0)
        # print("Grad outputs", grad_outputs[0].shape, len(grad_outputs))
        # gx, = grad_outputs
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.comm.nccl_allreduce(gx, self.comm)
        torch.cuda.synchronize()
        stop = time.perf_counter()
        if self.original_comm.rank == 0:
            print("{:.10f}".format(stop - start), "\t", file=self.allreduce_time_file)

        # print("GX", type(gx))
        # if self.comm.rank == 0:
        #     print("Backward shape of gradiets", gx.shape)
        return gx,


def filter_allgather(original_comm, comm, out, x):
    return FilterAllGather(original_comm, comm, out)(x)

