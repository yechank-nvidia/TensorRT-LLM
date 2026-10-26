# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request transport tests for the TP shared-tensor broadcast.

Multi-rank cases run on four MPI pool workers through ``mpi_pool_executor``.
CPU cases are marked ``cpu_only``. The CUDA case needs one visible GPU and
runs on GPU stages.
"""

import contextlib
import faulthandler
import gc
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

# An importable namedtuple, so peers can unpickle it.
from urllib.parse import DefragResult as Metadata

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest
from tensorrt_llm._torch.pyexecutor.request_utils import (
    RequestBroadcaster,
    attach_py_objects_to_requests,
)
from tensorrt_llm._torch.pyexecutor.tp_bcast_handles import TpBroadcastHandles
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD_SIZE = 4
WORKER_TIMEOUT_S = 300
ON_FOUR_WORKERS = pytest.mark.parametrize("mpi_pool_executor", [WORLD_SIZE], indirect=True)


class Pause:
    pause = staticmethod(contextlib.nullcontext)


class Context:
    """Per-worker MPI state, built inside the worker and never at import time."""

    def __init__(self, device: str = "cpu") -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.device = device
        size = self.comm.Get_size()
        self.dist = MPIDist(Mapping(world_size=size, rank=self.rank, tp_size=size))
        self.root_pid = self.comm.bcast(os.getpid() if self.rank == 0 else None, root=0)

    def broadcaster(self, handles: bool = True) -> RequestBroadcaster:
        broadcaster = RequestBroadcaster(self.dist, Pause())
        assert broadcaster.tp_handles is not None
        if not handles:
            broadcaster.tp_handles = None
        return broadcaster

    def sync(self) -> None:
        gc.collect()
        self.comm.Barrier()
        if self.device != "cpu":
            torch.cuda.synchronize(0)
            torch.cuda.ipc_collect()
            self.comm.Barrier()

    def assert_clean(self) -> None:
        """Rank 0 owns no leftover shared-memory files once every rank drops its views."""
        self.sync()
        deadline = time.monotonic() + 15
        owned = []
        if self.rank == 0:
            while time.monotonic() < deadline:
                owned = list(Path("/dev/shm").glob("torch_" + str(self.root_pid) + "_*"))
                if not owned:
                    break
                gc.collect()
                time.sleep(0.05)
        owned = self.comm.bcast([str(path) for path in owned], root=0)
        assert not owned, owned


def _guarded(worker: Callable[..., None], *args: Any) -> None:
    """Run one rank's part; exit the worker if peers never finish a collective.

    A failed assertion on one rank leaves the others waiting in a collective.
    The watchdog turns that hang into a broken pool, so the test fails fast.
    """

    def expire() -> None:
        faulthandler.dump_traceback(all_threads=True)
        os._exit(1)

    watchdog = threading.Timer(WORKER_TIMEOUT_S, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        worker(*args)
    finally:
        watchdog.cancel()


def _run(executor: Any, worker: Callable[..., None], *args: Any) -> None:
    results = executor.starmap(_guarded, [(worker, *args)] * WORLD_SIZE)
    assert list(results) == [None] * WORLD_SIZE


def _tensors(device: str) -> list[torch.Tensor]:
    dtypes = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    tensors = [torch.arange(24).reshape(4, 6).to(dtype=dtype, device=device) for dtype in dtypes]
    tensors += [
        torch.arange(40, device=device).reshape(5, 8)[1:5:2, 1:7:2],
        torch.tensor(3.5, device=device),
        torch.tensor([0x7FC00123, 0x80000000, 0x7F800000], dtype=torch.int64, device=device)
        .to(torch.int32)
        .view(torch.float32),
    ]
    return tensors


def _multimodal_data(device: str) -> dict:
    values = _tensors(device)
    return dict(
        image=dict(pixel_values=values[0]),
        # The empty tensor goes as bytes; its neighbour is still shared.
        audio=dict(input_features=values[1], empty=torch.empty(0, device=device)),
        video=values[2:],
        metadata=[1, None, "keep"],
    )


def _shareable(data: dict) -> list[torch.Tensor]:
    return [data["image"]["pixel_values"], data["audio"]["input_features"], *data["video"]]


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _request(data: Any) -> list[RequestQueueItem]:
    request = ExecutorRequest(input_token_ids=[1, 2], max_tokens=4)
    request.py_multimodal_data = data
    request.py_conversation_params = None
    return [RequestQueueItem(id=41, request=request)]


def _transfer(ctx: Context, handles: bool, delay: bool = False) -> None:
    broadcaster = ctx.broadcaster(handles)
    requests = _request(_multimodal_data(ctx.device)) if ctx.rank == 0 else []
    requests, objects = broadcaster.broadcast(requests)
    attach_py_objects_to_requests(requests, objects)
    actual = requests[0].request.py_multimodal_data
    expected = _shareable(_multimodal_data(ctx.device))
    tensors = _shareable(actual)
    assert len(tensors) == len(expected)
    for left, right in zip(expected, tensors):
        assert (left.dtype, left.shape, left.stride(), _bits(left)) == (
            right.dtype,
            right.shape,
            right.stride(),
            _bits(right),
        )
        if right.is_cuda:
            # Consumers may use a different GPU from the exporting rank.
            assert _bits(right.to(torch.cuda.current_device())) == _bits(left)
        if ctx.rank and handles:
            assert right.is_shared()
    empty = actual["audio"]["empty"]
    assert empty.numel() == 0 and empty.device.type == torch.device(ctx.device).type
    if ctx.rank and ctx.device == "cpu":
        assert not empty.is_shared()
    if delay and ctx.rank == WORLD_SIZE - 1:
        # Other ranks discard their request while the final consumer keeps reading.
        time.sleep(0.2)
        assert _bits(tensors[0]) == _bits(expected[0])
    assert actual["metadata"] == [1, None, "keep"]


def _roundtrip(ctx: Context, handles: bool, delay: bool = False) -> None:
    _transfer(ctx, handles, delay)
    ctx.assert_clean()


def _wire(
    ctx: Context, handles: bool = True, remote: bool = False, special: str | None = None
) -> list[bytes]:
    """Return the pickled payloads rank 0 broadcasts for one request."""
    broadcaster = ctx.broadcaster(handles)
    if remote:
        broadcaster.tp_handles.enabled = False
    values = torch.arange(1024)
    if special == "unsupported":
        values = values.to(torch.complex64)
    elif special == "empty":
        values = torch.empty(0)
    elif special == "negative":
        values = torch._neg_view(values)
    requests = _request(dict(image=values)) if ctx.rank == 0 else []
    objects = broadcaster._collect_py_objects(requests) if ctx.rank == 0 else None
    payloads = []
    original = ctx.dist.broadcast

    def capture(value: Any, **kwargs: Any) -> Any:
        if ctx.rank == 0:
            payloads.append(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return original(value, **kwargs)

    with patch.object(ctx.dist, "broadcast", capture):
        result = broadcaster._broadcast_requests(requests, objects)
    assert torch.equal(dict(result[1])["py_multimodal_data"][41]["image"], values)
    return payloads


def _default_and_fallback_worker() -> None:
    ctx = Context()
    default = _wire(ctx)
    byte_transport = _wire(ctx, handles=False)
    compact = len(default[0]) < len(byte_transport[0]) / 2 if ctx.rank == 0 else None
    assert ctx.comm.bcast(compact, root=0), "Default serialized tensor data"
    remote = _wire(ctx, remote=True)
    assert remote == byte_transport, "Multi-host fallback changed serialized bytes"
    # Exercise real hostname detection independently of the transport fallback.
    helper = TpBroadcastHandles(ctx.dist, ("py_multimodal_data",))
    with patch("socket.gethostname", return_value="host-" + str(ctx.rank // 2)):
        result = helper.broadcast([], ())
    assert helper.enabled is False and result == ([], ())
    for special in ("unsupported", "empty", "negative"):
        _wire(ctx, special=special)
    ctx.assert_clean()


def _metadata_worker() -> None:
    ctx = Context()
    for handles in (False, True):
        broadcaster = ctx.broadcaster(handles)
        requests = []
        if ctx.rank == 0:
            requests = _request(
                dict(
                    image=torch.arange(6).reshape(2, 3),
                    metadata=Metadata("image", torch.Size((2, 3))),
                )
            )
            requests[0].request.py_scheduling_params = Metadata("schedule", torch.Size((4,)))
        requests, objects = broadcaster.broadcast(requests)
        attach_py_objects_to_requests(requests, objects)
        metadata = requests[0].request.py_multimodal_data["metadata"]
        other = requests[0].request.py_scheduling_params
        for value, label, size in ((metadata, "image", (2, 3)), (other, "schedule", (4,))):
            assert isinstance(value, Metadata) and isinstance(value[1], torch.Size), (
                type(value),
                type(value[1]),
            )
            assert value[0] == label and value[1] == size
        del requests, objects, metadata, other
        ctx.assert_clean()


def _tensor_variants_worker(handles: bool) -> None:
    _roundtrip(Context(), handles)


def _last_consumer_worker() -> None:
    _roundtrip(Context(), True, delay=True)


def _failure_falls_back(ctx: Context, method: str, failing_rank: int) -> None:
    """A failed export or mapping sends the batch as bytes and disables handles."""
    broadcaster = ctx.broadcaster()
    calls = 0
    original = getattr(SharedTensorContainer, method)

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2 and ctx.rank == failing_rank:
            raise RuntimeError("Deliberate failure")
        return original(*args, **kwargs)

    requests = _request(_multimodal_data(ctx.device)) if ctx.rank == 0 else []
    with patch.object(SharedTensorContainer, method, fail_once):
        requests, objects = broadcaster.broadcast(requests)
    attach_py_objects_to_requests(requests, objects)
    expected = _shareable(_multimodal_data(ctx.device))
    actual = _shareable(requests[0].request.py_multimodal_data)
    assert [_bits(tensor) for tensor in actual] == [_bits(tensor) for tensor in expected]
    assert broadcaster.tp_handles.enabled is False
    # Later batches use the byte path without exporting anything.
    requests = _request(_multimodal_data(ctx.device)) if ctx.rank == 0 else []
    with patch.object(SharedTensorContainer, "from_tensor", side_effect=AssertionError):
        broadcaster.broadcast(requests)
    del requests, objects, actual
    ctx.assert_clean()


def _failure_worker(method: str) -> None:
    ctx = Context()
    _failure_falls_back(ctx, method, failing_rank=0 if method == "dump_to_dict" else 2)


def _serialization_abort(ctx: Context) -> None:
    broadcaster = ctx.broadcaster()
    requests = _request(dict(image=torch.arange(4096, device=ctx.device))) if ctx.rank == 0 else []
    if ctx.rank == 0:
        requests[0].request.py_logits_post_processors = lambda value: value
    failed = False
    try:
        broadcaster.broadcast(requests)
    except RuntimeError:
        failed = True
    assert failed, "Serialization failure must reach every rank"
    requests.clear()
    ctx.assert_clean()


def _serialization_abort_worker() -> None:
    _serialization_abort(Context())


def _retirement_cycles_worker() -> None:
    ctx = Context()
    fds = []
    for cycle in range(20):
        _roundtrip(ctx, True, delay=cycle % 5 == 0)
        fds.append(len(list(Path("/proc/self/fd").iterdir())))
    assert max(fds[2:]) <= fds[1], fds


def _no_tensor_payload_worker(data: dict | None) -> None:
    ctx = Context()
    requests = _request(data) if ctx.rank == 0 else []
    with patch.object(SharedTensorContainer, "from_tensor", side_effect=AssertionError):
        requests, objects = ctx.broadcaster().broadcast(requests)
    attach_py_objects_to_requests(requests, objects)
    if data is None:
        assert "py_multimodal_data" not in dict(objects)
    else:
        assert requests[0].request.py_multimodal_data == data


def _view_failure_worker() -> None:
    # CPU from_dict has consumed the exported reference before view creation.
    # A failing view must not make rank 0 release that reference a second time.
    ctx = Context()
    released = 0

    def record_release(_container: SharedTensorContainer) -> None:
        nonlocal released
        released += 1

    with patch.object(SharedTensorContainer, "release_unconsumed", record_release):
        _failure_falls_back(ctx, "get_local_view", failing_rank=2)
    released = ctx.comm.bcast(released if ctx.rank == 0 else None, root=0)
    assert released == 0, "An opened CPU storage reference was released twice"


@pytest.mark.cpu_only
def test_activation() -> None:
    class Unsupported(MPIDist):
        """The gate reads the capability, not the transport class."""

        supports_local_tensor_handles = False

    for tp, pp, cp in [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 1, 2), (4, 1, 1)]:
        mapping = Mapping(world_size=tp * pp * cp, rank=0, tp_size=tp, pp_size=pp, cp_size=cp)
        enabled = tp > 1 and pp == cp == 1
        broadcaster = RequestBroadcaster(MPIDist(mapping), Pause())
        assert (broadcaster.tp_handles is not None) == enabled
        if enabled:
            # The single-round path would bypass the handle dispatch.
            assert not broadcaster._single_round(False)
            assert broadcaster.tp_handles.tensor_attributes == {"py_multimodal_data"}
        assert RequestBroadcaster(Unsupported(mapping), Pause()).tp_handles is None


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_default_and_fallback(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _default_and_fallback_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_metadata_types(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _metadata_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("handles", [False, True], ids=["bytes", "automatic_handles"])
def test_tensor_variants(mpi_pool_executor: Any, handles: bool) -> None:
    _run(mpi_pool_executor, _tensor_variants_worker, handles)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_last_consumer_lifetime(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _last_consumer_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("method", ["dump_to_dict", "from_dict"], ids=["export", "mapping"])
def test_failure_falls_back_to_bytes(mpi_pool_executor: Any, method: str) -> None:
    _run(mpi_pool_executor, _failure_worker, method)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_serialization_abort(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _serialization_abort_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_retirement_cycles(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _retirement_cycles_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("data", [None, {"metadata": [1, "image", None]}])
def test_requests_without_tensor_payload(mpi_pool_executor: Any, data: dict | None) -> None:
    _run(mpi_pool_executor, _no_tensor_payload_worker, data)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_view_failure_after_storage_open(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _view_failure_worker)


def _imported_cuda_roundtrip(ctx: Context) -> None:
    """Rank 0 re-shares a CUDA tensor it imported from another process."""
    source = torch.arange(4096, device="cuda:0")
    handle = None
    if ctx.rank == 1:
        handle = SharedTensorContainer.from_tensor(source).dump_to_dict()
    handle = ctx.comm.bcast(handle, root=1)
    requests = []
    if ctx.rank == 0:
        imported = SharedTensorContainer.from_dict(handle).get_local_view()
        requests = _request(dict(image=imported))
    requests, objects = ctx.broadcaster().broadcast(requests)
    attach_py_objects_to_requests(requests, objects)
    assert torch.equal(requests[0].request.py_multimodal_data["image"].to("cuda:0"), source)
    del requests, objects
    ctx.comm.Barrier()


def _cuda_worker() -> None:
    torch.cuda.set_device(MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count())
    ctx = Context("cuda:0")
    for handles in (False, True):
        _roundtrip(ctx, handles, delay=True)
    expanded = torch.arange(4, device="cuda:0").expand(4, 4)
    requests = _request(dict(image=expanded)) if ctx.rank == 0 else []
    requests, objects = ctx.broadcaster().broadcast(requests)
    assert torch.equal(dict(objects)["py_multimodal_data"][41]["image"], expanded)
    del requests, objects
    _imported_cuda_roundtrip(ctx)
    _failure_falls_back(ctx, "dump_to_dict", failing_rank=0)
    _failure_falls_back(ctx, "from_dict", failing_rank=2)
    _serialization_abort(ctx)
    allocations, fds = [], []
    for cycle in range(20):
        _roundtrip(ctx, True, delay=cycle % 5 == 0)
        # ipc_collect frees rank 0 blocks that peers released. Without it, the
        # caching allocator frees them on its next out-of-memory retry.
        allocations.append(torch.cuda.memory_allocated(0))
        fds.append(len(list(Path("/proc/self/fd").iterdir())))
    assert max(allocations[2:]) <= allocations[1], allocations
    assert max(fds[2:]) <= fds[1], fds


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires a visible CUDA device")
@ON_FOUR_WORKERS
def test_cuda_transport(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _cuda_worker)
