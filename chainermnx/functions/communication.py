from abc import ABC

import chainer
from chainer import backend
import time
import chainermn


class AllReduce(chainer.Function, ABC):
    # this was created to faciliate the allreduce/allgather required in the case of channel parallelism
    def __init__(self, comm, index):
        self.comm = comm
        self.index = index

    def forward(self, inputs):
        x, = inputs

        xp = backend.get_array_module(*inputs)
        # print(type(x))

        # xs = self.comm.allreduce(x)
        # temporary change allreduce allgorithm.
        start = time.time()
        # self.comm.alltoall(x)
        # xs = chainermn.functions.alltoall(self.comm, x)
        # print("All to all was executed succeesffully----------------------------------")
        # xs = xp.stack(xs).sum(axis=0)
        # stop = time.time()
        # if self.comm.rank == 0:
        #     print("Layer Number", self.index, "Time for Forward ", stop - start, "Shape of xs", xs.shape)

        ## ----------------------try to implement with NCCL-----------------#

        xs = self.comm.channel_allreduce(x)
        print("This part passed---------------------------------------")
        return xs,

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        gx, = grad_outputs
        start = time.time()
        gxs = self.comm.allgather(gx)
        stop = time.time()
        gxs = xp.stack(gxs)
        if self.comm.rank == 0:
            print("Layer Number", self.index, "Time for Backward ", stop - start, "Shape of gxs", gxs[0].shape)
        return gxs


def allreduce(comm, x, index):
    # In forward pass, do an all reduce on y_p
    # On backward pass do an all gather on input gradients
    return AllReduce(comm,  index)(x)
