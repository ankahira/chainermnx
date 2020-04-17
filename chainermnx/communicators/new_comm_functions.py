import warnings

import chainer.cuda

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermnx.communicators import channel_mpi_communicator_base
from chainermnx.communicators.channel_mpi_communicator_base import  _MessageType, _check_dtype, _check_dtypes_are_same, _get_mpi_type, _cnt_to_dsp

from chainermn import nccl
import time
import cupy as cp
import numpy as np


def _get_nccl_dtype_size(input):
    if input.dtype == np.float32:
        nccl_dtype = nccl.NCCL_FLOAT32
        nccl_size = input.size
    elif input.dtype == np.float64:
        nccl_dtype = nccl.NCCL_FLOAT64
        nccl_size = input.size
    elif input.dtype == np.complex64:
        nccl_dtype = nccl.NCCL_FLOAT32
        nccl_size = input.size * 2
    elif input.dtype == np.complex128:
        nccl_dtype = nccl.NCCL_FLOAT64
        nccl_size = input.size * 2
    else:
        raise ValueError(
            'dtype not supported, got {dtype}.'.format(dtype=input.dtype))

    return nccl_dtype, nccl_size


def nccl_allreduce (input, comm):
    nccl_comm = _communication_utility.init_nccl_comm(comm.mpi_comm)
    nccl_dtype, nccl_size = _get_nccl_dtype_size(input)
    nccl_comm.allReduce(input.data.ptr,
                        input.data.ptr,
                        nccl_size, nccl_dtype,
                        nccl.NCCL_SUM,
                        cp.cuda.Stream.null.ptr)

    return


# def nccl_allgather (input, output, comm):
#     nccl_comm = _communication_utility.init_nccl_comm(comm.mpi_comm)
#     nccl_dtype, nccl_size = _get_nccl_dtype_size(input)
#
#     # print(nccl_dtype)
#
#     nccl_size = nccl_size * 6
#
#     nccl_comm.allGather(input.data.ptr,
#                         output.data.ptr,
#                         nccl_size, nccl_dtype,
#                         cp.cuda.Stream.null.ptr)
#
#     return

#
def nccl_allgather_nguyen(x, comm):
    # x (chainer.Variables): Variables to send.
    # Returns:
    # ys (list of chainer.Variables): Received variables.

    nccl_comm = _communication_utility.init_nccl_comm(comm.mpi_comm)
    nccl_dtype, nccl_size = _get_nccl_dtype_size(x)
    # nccl_size = nccl_size * 3 #should replace with nccl_com.rank
    xp = chainer.backend.get_array_module(x)

    y = xp.empty(x.size * 3, x.dtype)

    ys = (y, y, y)

    print("starting allgather -----------------------------------------------")

    nccl_comm.allGather(x.data.ptr, _memory_utility.get_device_memory_pointer(y), nccl_size, nccl_dtype, cp.cuda.Stream.null.ptr)

    print("passed here----------")

    return ys


def nccl_allgather(x, comm):
    # chainer.utils.experimental(
    #     'chainermn.communicators.MpiCommunicatorBase.allgather')

    msgtype = _MessageType(x)
    # _check_dtype('allgather', msgtype)

    # # msgtypes = comm.mpi_comm.allgather(msgtype)
    # _check_dtypes_are_same(msgtypes)
    #
    # # Type check.
    # for msgtype in msgtypes:
    #     if msgtype.is_tuple:
    #         raise TypeError('allgather cannot handle tuple data')
    #
    #     assert len(msgtype.shapes) == 1

    # Collective communication.]
    start = time.time()
    # nccl_comm = _communication_utility.init_nccl_comm(comm.mpi_comm)
    nccl_dtype, nccl_size = _get_nccl_dtype_size(x)
    stop = time.time()

    if comm.rank == 0:
        print("Elapsed time", stop - start)

    xp = chainer.backend.get_array_module(x)
    # shapes = [msgtype.shapes[0] for msgtype in msgtypes]
    shapes = [msgtype.shapes[0], msgtype.shapes[0], msgtype.shapes[0]]
    sbuf = _memory_utility.array_to_buffer_object(x, _get_mpi_type(msgtype))
    rlens = [chainer.utils.size_of_shape(s) for s in shapes]
    # rbuf = xp.empty([sum(rlens)], dtype=msgtype.dtype)
    rbuf = xp.empty(nccl_size*3, dtype=msgtype.dtype)
    if xp is not np:
        chainer.cuda.Stream.null.synchronize()

    # Here we want to replace mpi with nccl

    start = time.time()

    comm.allGather(
        x.data.ptr,
        rbuf.data.ptr,
        nccl_size,
        nccl_dtype,
        cp.cuda.Stream.null.ptr
    )
    stop = time.time()

    # if comm.rank == 0:
    #     print("All gather Elapsed time", stop - start)

    start = time.time()
    ys = [rbuf[i:i + l].reshape(s)
          for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

    stop = time.time()

    # if comm.rank == 0:
    #     print("Loop Elapsed time", stop - start)

    return ys


