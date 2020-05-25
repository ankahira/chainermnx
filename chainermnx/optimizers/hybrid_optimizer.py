import chainer
import copy
import time
import os
import torch


class _HybridMultiNodeOptimizer(object):

    def __init__(self, actual_optimizer, original_communicator, global_communicator, local_communicator, out, zero_fill):

        """

        we have a global comm and a local comm.


        :param actual_optimizer:
        :param global_communicator:
        :param local_communicator:
        :param zero_fill:
        """
        super(_HybridMultiNodeOptimizer, self).__setattr__(
            'original_communicator', original_communicator)
        super(_HybridMultiNodeOptimizer, self).__setattr__(
            'global_communicator', global_communicator)
        super(_HybridMultiNodeOptimizer, self).__setattr__(
            'local_communicator', local_communicator)
        super(_HybridMultiNodeOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_HybridMultiNodeOptimizer, self).__setattr__(
            'target_params', [])
        super(_HybridMultiNodeOptimizer, self).__setattr__(
            'zero_fill', zero_fill)
        self.out = out

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                target.cleargrads()
            else:
                target.zerograds()
            loss.backward(loss_scale=self.actual_optimizer._loss_scale)
            del loss

        if self.is_changed(target):
            self.global_communicator.bcast_data(target)
        else:
            """
            This is a critical part of hybrid. First do an allreduce on each node then do an all reduce globally. 
            However the local allreduce is the modified allreduce such that it doesnt take the sum ie like spatial             
            """
            #TODO
            # There might be an issue with how you perform allreduce.
            # Need to ensure that the correct group in global comm performs allreduce
            # Maybe an if statement before the global allreduce such that only leading GPUs perform allreduce
            # Also remember this is for both spatial hybrid and filter hybrid. Might need to change accordingly
            cpu_start = time.time()
            torch.cuda.synchronize()
            cpu_stop = time.time()
            #self.original_communicator.mpi_barrier()
            start = time.time()
            self.local_communicator.intra_node_mean_grad(target, self.zero_fill)
            torch.cuda.synchronize()
            stop = time.time()
            local_allreduce_time = stop - start

            torch.cuda.synchronize()
            #self.original_communicator.mpi_barrier()
            start = time.time()
            if self.local_communicator.rank == 0:
              self.global_communicator.multi_node_mean_grad(target, self.zero_fill)
            torch.cuda.synchronize()
            self.original_communicator.mpi_barrier()
            stop = time.time()
            global_allreduce_time = stop - start
            
            allreduce_time_file = open(os.path.join(self.out, "gradient_allreduce_times.log"), "a")
            # Find a way to print with just one rank .
            if self.original_communicator.rank == 0:
                print("{:.10f}".format(local_allreduce_time), "\t", "{:.10f}".format(global_allreduce_time),
                      "\t", "{:.10f}".format(cpu_stop - cpu_start), file=allreduce_time_file)
            self.actual_optimizer.update(None, *args, **kwds)

    def is_changed(self, target):
        previous_params = self.target_params
        super(_HybridMultiNodeOptimizer, self).__setattr__(
            'target_params', [(name, param.data is not None)
                              for name, param in sorted(target.namedparams())])
        if len(previous_params) != len(self.target_params):
            return True

        for param1, param2 in zip(self.target_params, previous_params):
            if (param1[0] != param2[0]) or param1[1] != param2[1]:
                return True
        return False

    def setup(self, link):
        self.actual_optimizer.setup(link)
        return self

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)


class _DoubleBufferingOptimizer(object):

    def __init__(self, actual_optimizer, communicator, zero_fill):
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicator', communicator)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'needs_update', False)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicated_target', None)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'target_params_list', [[], []])
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'allreduce_grad_stream', chainer.cuda.Stream(non_blocking=True))
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'zero_fill', zero_fill)

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                target.cleargrads()
            else:
                target.zerograds()
            loss.backward(loss_scale=self.actual_optimizer._loss_scale)
            del loss

        if self.is_changed(target, self.target_params_list[0]):
            self.wait()
            self.communicator.bcast_data(target)
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'communicated_target', copy.deepcopy(target))
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'target_params_list', [
                    list(sorted(self.target.namedparams())),
                    list(sorted(self.communicated_target.namedparams()))])
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'needs_update', False)
        else:
            self.wait()
            self.swap_grad(self.target_params_list[0],
                           self.target_params_list[1])
            self.multi_node_mean_grad_async()
            if self.needs_update:
                self.actual_optimizer.update(None, *args, **kwds)
            else:
                super(_DoubleBufferingOptimizer, self).__setattr__(
                    'needs_update', True)

    def multi_node_mean_grad_async(self):
        self.communicator._multi_node_mean_grad_async(self.communicated_target,
                                                      self.zero_fill,
                                                      self.allreduce_grad_stream)

    def is_changed(self, target, previous_params):
        target_params = list(sorted(target.namedparams()))
        if len(previous_params) != len(target_params):
            return True

        for param1, param2 in zip(target_params, previous_params):
            name1, var1 = param1
            name2, var2 = param2
            if (name1 != name2) or (var1.data is None) != (var2.data is None):
                return True
        return False

    def swap_grad(self, target1_params, target2_params):
        for param1, param2 in zip(target1_params, target2_params):
            _, var1 = param1
            _, var2 = param2
            var1.grad, var2.grad = var2.grad, var1.grad

    def wait(self):
        self.allreduce_grad_stream.synchronize()
        chainer.cuda.Stream.null.synchronize()

    def setup(self, link):
        self.actual_optimizer.setup(link)
        return self

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)


def create_hybrid_multi_node_optimizer(actual_optimizer, original_communicator,  global_communicator, local_communicator, out="result",
                                       double_buffering=False, zero_fill=True):
    """Create a Hybrid multi node optimizer from a Chainer optimizer.
    This function is modified from the origal multinode optimiser so as to perform a a sum all reduce on local
    nodes then a global average all reduce
    At the moment double buffering is not supported

    Args:
        actual_optimizer: Chainer optimizer (e.g., ``chainer.optimizers.Adam``).
        global_communicator: ChainerMN communicator.
        local_communicator: Main communicator split into local communicators
        double_buffering: False by default
        zero_fill: True by default

    Returns:
        The Hybrid  multi node optimizer based on ``actual_optimizer``.


    """
    if double_buffering:
        from chainermn.communicators.pure_nccl_communicator \
            import PureNcclCommunicator
        if not isinstance(global_communicator, PureNcclCommunicator):
            raise ValueError(
                'This communicator does not support double buffering.')
        return _DoubleBufferingOptimizer(actual_optimizer, global_communicator, zero_fill)
    return _HybridMultiNodeOptimizer(actual_optimizer=actual_optimizer,
                                     original_communicator=original_communicator,
                                     global_communicator=global_communicator,
                                     local_communicator=local_communicator,
                                     out=out, zero_fill=zero_fill)
