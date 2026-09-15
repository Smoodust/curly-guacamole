"""Copy training batches on a separate CUDA stream without device-wide waits."""

import torch


class BatchTransfer:
    def __init__(self, device, *, asynchronous=True):
        self.device = torch.device(device)
        self.stream = None
        if self.device.type == 'cuda' and asynchronous:
            self.stream = torch.cuda.Stream(device=self.device)
            # Establish ordering once; waiting on compute for every batch would
            # serialize its copy with the preceding batch's queued computation.
            self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def __call__(self, batch):
        if self.stream is None:
            return self._copy(batch)
        with torch.cuda.stream(self.stream):
            result = self._copy(batch)
        compute = torch.cuda.current_stream(self.device)
        compute.wait_stream(self.stream)
        for value in result.values():
            if torch.is_tensor(value) and value.is_cuda:
                value.record_stream(compute)
        for sample in result.get('jpeg', []):
            for value in sample.values():
                if torch.is_tensor(value) and value.is_cuda:
                    value.record_stream(compute)
        return result

    @staticmethod
    def move_jpeg(inputs, device):
        return [{key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                 for key, value in sample.items()} for sample in inputs]

    def _copy(self, batch):
        result = {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                  for key, value in batch.items()}
        if 'jpeg' in batch:
            result['jpeg'] = self.move_jpeg(batch['jpeg'], self.device)
        return result
