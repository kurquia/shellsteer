"""The devices of a stage that runs the network, and its batches spread over them.

`parse_devices` reads the --devices option: every visible GPU by default, the CPU if none is visible. `load_workers`
loads one copy of the network per device, once per stage. `Workers.map` runs batch i on device i mod n, one thread
per device, and returns the results in the order of i. Each batch runs inside `networks.precision(network)`, which
turns TF32 off for ShellGATr.

Batch i draws its noise from `sampler.batch_seed(seed, i)` on its own device, so on GPUs of one type it gives the
same numbers on any of them. The stages assemble the results in batch order and add the sums of a batch in the same
order and dtype as on one device, so their files are bit-identical to a run on one GPU of that type. A type is the
GPU's name and its number of streaming multiprocessors: a MIG slice draws other noise than the full GPU.

The threads share the process. PyTorch releases the GIL while an operator runs, and CUDA kernels run asynchronously,
so the GPUs work at the same time. With one device, `Workers.map` runs every batch in the calling thread.
"""

from __future__ import annotations

import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

import torch

from shellsteer.config import NetworkEntry
from shellsteer.networks import Network, load_network, precision

R = TypeVar("R")
Devices = tuple[torch.device, ...]


class DeviceError(ValueError):
    """The --devices option names a device that this machine does not have, or a list that cannot run."""


def parse_devices(text: str | None) -> Devices:
    """The devices of --devices TEXT, such as "cuda:0,cuda:1", "cpu", or "cpu,cpu" (two CPU workers).

    None gives every visible GPU (CUDA_VISIBLE_DEVICES applies), or the CPU if there is none. Stops on an entry that
    is not cpu or cuda:<index>, on a GPU that is not visible, on a GPU named twice, and on a list that mixes the CPU
    and GPUs.
    """
    if text is None:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return tuple(torch.device("cuda", i) for i in range(count)) or (torch.device("cpu"),)
    devices = []
    for item in text.split(","):
        item = item.strip()
        if not re.fullmatch(r"cpu|cuda:\d+", item):
            raise DeviceError(f"--devices {text}: {item!r} is not cpu or cuda:<index>; write for example "
                              f"cuda:0,cuda:1, or leave out --devices for every visible GPU")
        devices.append(torch.device(item))
    cuda = [device for device in devices if device.type == "cuda"]
    if cuda and len(cuda) != len(devices):
        raise DeviceError(f"--devices {text}: the list mixes the CPU and GPUs; name GPUs only, or cpu only")
    if cuda:
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible == 0:
            raise DeviceError(f"--devices {text}: CUDA is not available here; run on a machine with a CUDA GPU, or "
                              f"pass --devices cpu (slow at the paper's size)")
        missing = [str(device) for device in cuda if device.index >= visible]
        if missing:
            raise DeviceError(f"--devices {text}: {', '.join(missing)} not visible; this machine shows {visible} "
                              f"GPU(s), cuda:0 to cuda:{visible - 1}")
        if len(set(cuda)) != len(cuda):
            raise DeviceError(f"--devices {text}: a GPU is named twice; name each GPU once")
    return tuple(devices)


def device_type(device: torch.device) -> str:
    """"cpu", or the GPU's name and number of streaming multiprocessors, such as "NVIDIA B200, 148 SMs"."""
    if device.type != "cuda":
        return device.type
    properties = torch.cuda.get_device_properties(device)
    return f"{properties.name}, {properties.multi_processor_count} SMs"


def devices_metadata(devices: Devices) -> str | list[str]:
    """The "device" entry of a file's metadata: the one device type of the run, or the type of each device, in the
    order that assigns the batches, if they differ. The number of devices is left out: it does not change the
    numbers."""
    types = [device_type(device) for device in devices]
    return types[0] if len(set(types)) == 1 else types


def describe_devices(devices: Devices) -> list[str]:
    """Log lines: how many devices the stage uses and which; a warning if the GPUs are of different types."""
    types = [device_type(device) for device in devices]
    if devices[0].type == "cpu":
        workers = "1 CPU worker" if len(devices) == 1 else f"{len(devices)} CPU workers"
        lines = [f"devices: {workers}" + ("" if torch.cuda.is_available() else
                                         "; no CUDA GPU is visible (slow at the paper's size)")]
    else:
        names = ", ".join(f"{device} {name}" for device, name in zip(devices, types))
        lines = [f"devices: {len(devices)} GPU{'s' if len(devices) > 1 else ''}: {names}"]
    if len(devices) > 1:
        lines.append(f"batch i runs on device i mod {len(devices)}, in the order above")
    if len(set(types)) > 1:
        lines.append("WARNING: the GPUs are of different types. The numbers of a batch depend on the GPU that runs "
                     "it, so the files differ from those of a run on one GPU.")
    return lines


def on_device(device: torch.device):
    """Make `device` the current CUDA device of this thread inside the `with` block; nothing for the CPU."""
    return torch.cuda.device(device) if device.type == "cuda" else nullcontext()


@dataclass(frozen=True)
class Workers:
    """One copy of the network per device; `map` spreads the batches of a pass over them."""
    networks: tuple[Network, ...]
    devices: Devices

    def __post_init__(self) -> None:
        if not self.devices or len(self.networks) != len(self.devices):
            raise ValueError(f"{len(self.networks)} networks for {len(self.devices)} devices; need one per device")

    @property
    def width(self) -> int:
        """Width of the residual stream of the network."""
        return self.networks[0].model.d_model

    @property
    def first(self) -> torch.device:
        """The first device, where the stages keep their sums."""
        return self.devices[0]

    def map(self, count: int, work: Callable[[int, Network, torch.device], R]) -> Iterator[R]:
        """work(i, network, device) for i = 0 .. count - 1, with batch i on device i mod n; yields the results in
        the order of i.

        With several devices, one thread per device runs its batches, at most 2 n batches ahead of the one the
        caller takes. An error in batch i reaches the caller when it takes batch i, as on one device; the other
        threads then end after their current batch.
        """
        if len(self.devices) == 1:
            network, device = self.networks[0], self.devices[0]
            for index in range(count):
                with on_device(device), precision(network):
                    result = work(index, network, device)
                yield result
            return
        queue = _InOrder(window=2 * len(self.devices))
        n = len(self.devices)
        threads = [threading.Thread(target=queue.work, args=(range(w, count, n), work, network, device),
                                    name=f"shellsteer worker {w} on {device}")
                   for w, (network, device) in enumerate(zip(self.networks, self.devices))]
        for thread in threads:
            thread.start()
        try:
            for index in range(count):
                yield queue.take(index)
        finally:
            queue.stop()
            for thread in threads:
                thread.join()


class _InOrder:
    """Results of the worker threads by batch index, handed to the caller in the order of the index."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.condition = threading.Condition()
        self.done: dict[int, tuple[object, BaseException | None]] = {}
        self.next = 0               # the batch the caller takes next
        self.stopped = False

    def work(self, indices: range, work: Callable, network: Network, device: torch.device) -> None:
        """Thread body: run the batches `indices` in order; store each result, or the error, and stop at an error."""
        for index in indices:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or index < self.next + self.window)
                if self.stopped:
                    return
            try:
                with on_device(device), precision(network):
                    entry = (work(index, network, device), None)
            except BaseException as error:              # handed to the caller, who raises it
                entry = (None, error)
            with self.condition:
                self.done[index] = entry
                self.condition.notify_all()
            if entry[1] is not None:
                return

    def take(self, index: int) -> object:
        """The result of batch `index`, once its thread has stored it; raises the batch's error."""
        with self.condition:
            self.condition.wait_for(lambda: index in self.done)
            result, error = self.done.pop(index)
            self.next = index + 1
            self.condition.notify_all()
        if error is not None:
            raise error
        return result

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


def load_workers(entry: NetworkEntry, devices: Devices) -> Workers:
    """Load the network of `entry` once per device."""
    return Workers(networks=tuple(load_network(entry, device) for device in devices), devices=tuple(devices))
