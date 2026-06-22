"""Spawn a small ``torch.distributed`` world for tests and benchmarks.

``run_distributed`` launches ``world_size`` subprocesses forming a process group and
runs a module-level ``worker(rank, world_size, *args, **kwargs)`` in each. The
worker must be importable/picklable because ``torch.multiprocessing.spawn``
re-imports the module in each child.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Callable

__all__ = ["run_distributed"]


def _find_free_port() -> int:
    """Ask the OS for an unused TCP port for the rendezvous ``MASTER_PORT``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dist_worker_entry(
    rank: int,
    world_size: int,
    backend: str,
    master_port: int,
    device_type: str,
    fn: Callable[..., None],
    args: tuple[object, ...],
    kwargs: dict[str, Any],
) -> None:
    """Initialize a process group, run ``fn``, then tear the group down."""
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    device_id = None
    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device_id = torch.device("cuda", rank)

    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size, device_id=device_id
    )
    try:
        fn(rank, world_size, *args, **kwargs)
        # Keep ranks synchronized so an early teardown does not wedge a peer mid-collective.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_distributed(
    worker: Callable[..., None],
    world_size: int = 2,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, Any] | None = None,
    backend: str = "gloo",
    device_type: str = "cpu",
) -> None:
    """Spawn ``world_size`` subprocesses forming a distributed world and run ``worker``.

    Each subprocess initializes a process group, then calls
    ``worker(rank, world_size, *args, **kwargs)``. Use this to exercise DTensor code
    paths that require a real device mesh and collectives.
    """
    import torch.multiprocessing as mp

    kwargs = kwargs or {}
    master_port = _find_free_port()
    mp.spawn(
        _dist_worker_entry,
        args=(world_size, backend, master_port, device_type, worker, args, kwargs),
        nprocs=world_size,
        join=True,
    )

