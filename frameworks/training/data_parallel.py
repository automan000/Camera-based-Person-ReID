from itertools import chain

from torch.nn import DataParallel
from torch.nn.parallel.scatter_gather import scatter, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply


class CamDataParallel(DataParallel):
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        all_inputs = inputs[0]
        all_kwargs = kwargs
        all_outputs = []

        while len(all_inputs) > 0:
            num_required_gpu = min(len(all_inputs), len(self.device_ids))
            actual_inputs = [all_inputs.pop(0) for _ in range(num_required_gpu)]
            inputs, kwargs = self.scatter(actual_inputs, all_kwargs, self.device_ids[:num_required_gpu])
            replicas = self.replicate(self.module, self.device_ids[:num_required_gpu])
            all_outputs.extend(self.parallel_apply(replicas, inputs, kwargs))

        return self.gather(all_outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, input_list, kwargs, device_ids):
        inputs = []
        for input, gpu in zip(input_list, device_ids):
            inputs.extend(scatter(input, [gpu], dim=0))
        kwargs = scatter(kwargs, device_ids, dim=0) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)
