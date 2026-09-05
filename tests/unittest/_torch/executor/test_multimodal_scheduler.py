# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    MM_ENCODER_COMPLETION_REQUEST_ID,
    MM_ENCODER_INPUT_REQUEST_ID,
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    MultimodalEncoderRequestError,
    MultimodalEncoderRequestState,
    get_multimodal_encoder_token_lengths,
    initialize_multimodal_encoder_request,
)
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine,
    _validate_mm_encoder_scheduling_compatibility,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    MultimodalScheduler,
    ScheduledRequests,
    SchedulerOutput,
    SimpleScheduler,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.request import MultimodalEncoderCompletion, MultimodalEncoderInput
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_INPUT_ID_KEY,
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalParams,
    MultimodalRuntimeData,
    strip_mm_encoder_inputs,
)
from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy


class _CapacityScheduler:
    def schedule_request(self, requests):
        return list(requests), [], []


class _RejectMultimodalCapacityScheduler:
    def schedule_request(self, requests):
        fitting = [request for request in requests if request.py_mm_encoder_state is None]
        return fitting, [], []


class _MicroBatchScheduler:
    def __init__(self, *, chunk_size=None, chunk_unit_size=None):
        self.chunk_size = chunk_size
        self.ctx_chunk_config = (
            None if chunk_unit_size is None else SimpleNamespace(chunk_unit_size=chunk_unit_size)
        )

    def schedule(self, requests, inflight_request_ids):
        del inflight_request_ids
        for request in requests:
            request.context_chunk_size = min(
                request.context_remaining_length,
                self.chunk_size or request.context_remaining_length,
            )
        return [], list(requests), []


class _BaseScheduler:
    def __init__(self, *, chunk_size=None, chunk_unit_size=None):
        self.capacity_scheduler = _CapacityScheduler()
        self.micro_batch_scheduler = _MicroBatchScheduler(
            chunk_size=chunk_size, chunk_unit_size=chunk_unit_size
        )

    @property
    def scheduling_state_range(self):
        return (LlmRequestState.CONTEXT_INIT, LlmRequestState.GENERATION_TO_COMPLETE)

    def schedule_request(self, requests, inflight_request_ids):
        fitting, disagg, paused = self.capacity_scheduler.schedule_request(requests)
        encoder, context, generation = self.micro_batch_scheduler.schedule(
            fitting, inflight_request_ids
        )
        return SchedulerOutput(encoder, context, generation, paused, disagg, len(fitting))

    def can_schedule(self, requests):
        return bool(requests)


def _item_cache_keys(request):
    state = request.py_mm_encoder_state
    return [("test_mm", request.request_id, item_idx) for item_idx in range(state.num_items)]


def _scheduler(
    *,
    max_batch_size,
    max_num_tokens,
    cache_capacity=1 << 20,
    base_scheduler=None,
    scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
    retain_cache_entries=False,
    encoder_outputs_ready_immediately=True,
):
    return MultimodalScheduler(
        base_scheduler or _BaseScheduler(),
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        encoder_cache=TensorLRUCache(cache_capacity),
        get_item_cache_keys=_item_cache_keys,
        bytes_per_encoder_embedding=4,
        retain_cache_entries=retain_cache_entries,
        scheduling_policy=scheduling_policy,
        encoder_outputs_ready_immediately=encoder_outputs_ready_immediately,
    )


def _llm_request(
    request_id,
    multimodal_data=None,
    *,
    input_tokens=None,
    multimodal_positions=None,
    multimodal_lengths=None,
):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=input_tokens or [1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        py_multimodal_data=multimodal_data,
        multimodal_positions=multimodal_positions,
        multimodal_lengths=multimodal_lengths,
    )


def _request(request_id, costs):
    positions = [2 * item_idx + 1 for item_idx in range(len(costs))]
    mask = [0] * (2 * len(costs) + 1)
    for position in positions:
        mask[position] = 1
    cumsum = torch.tensor(mask, dtype=torch.int64).cumsum(0)
    request = _llm_request(
        request_id,
        multimodal_data={
            "image": {"pixel_values": torch.empty(len(costs), 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", item_idx) for item_idx in range(len(costs))],
                encoder_token_lengths=costs,
                output_embedding_lengths=[1] * len(costs),
            ),
            "multimodal_embedding_lengths": [1] * len(costs),
            "multimodal_embed_mask_cumsum": cumsum,
        },
        input_tokens=list(range(len(mask))),
        multimodal_positions=positions,
        multimodal_lengths=[1] * len(costs),
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=1 << 30)
    return request


def test_collect_scheduled_batch_stats_includes_multimodal_work():
    request = _request(7, [11, 13])
    scheduled = ScheduledRequests()
    scheduled.scheduled_mm_encoder_items = {request.request_id: [0, 1]}
    scheduled.mm_encoder_blocked_request_ids = [8]
    scheduled.mm_encoder_cache_removals = [("old", 0)]
    cache = TensorLRUCache(1 << 20)
    executor = SimpleNamespace(
        _mm_encoder_item_scheduling_enabled=True,
        active_requests=[request],
        model_engine=SimpleNamespace(
            bytes_per_mm_encoder_embedding=4,
            mm_encoder_cache=cache,
        ),
        _is_stats_dummy_request=PyExecutor._is_stats_dummy_request,
    )

    stats = PyExecutor._collect_scheduled_batch_stats(executor, scheduled)

    assert stats.mm_encoder_stats == {
        "numItems": 2,
        "numInputTokens": 24,
        "numOutputRows": 2,
        "numBlockedRequests": 1,
        "numCacheRemovals": 1,
        "selectedOutputBytes": 8,
        "cacheMaxBytes": 1 << 20,
        "cacheCurrentBytes": 0,
        "cacheReservedBytes": 0,
        "cacheInUseBytes": 0,
        "cacheHits": 0,
        "cacheProducerMisses": 0,
        "cacheReservationHits": 0,
        "cacheEvictions": 0,
    }


def test_mm_encoder_token_lengths_distinguishes_missing_and_invalid_data():
    request = _llm_request(1)

    assert get_multimodal_encoder_token_lengths(request) is None

    request.py_multimodal_data = []
    with pytest.raises(TypeError, match="multimodal_data must be a dict"):
        get_multimodal_encoder_token_lengths(request)


def test_item_scheduling_rejects_raw_payload_without_item_metadata():
    request = _llm_request(
        1,
        multimodal_data={"image": {"pixel_values": torch.empty(1, 1)}},
    )

    with pytest.raises(ValueError, match="requires multimodal_encoder_item_metadata"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_pending_external_encoder_outputs_initialize_item_state():
    request = _llm_request(
        1,
        multimodal_data={
            "multimodal_embedding_lengths": [2, 3],
            "encoder_token_lengths": [8, 12],
        },
        multimodal_positions=[1, 5],
        multimodal_lengths=[2, 3],
    )

    initialize_multimodal_encoder_request(request, max_num_tokens=16)

    state = request.py_mm_encoder_state
    assert state is not None
    assert state.embedding_lengths == [2, 3]
    assert state.encoder_token_lengths == [8, 12]


def test_complete_external_encoder_outputs_keep_whole_request_path():
    request = _llm_request(
        1,
        multimodal_data={
            "multimodal_embedding": torch.ones(5, 4),
            "multimodal_embedding_lengths": [2, 3],
            "encoder_token_lengths": [8, 12],
        },
        multimodal_positions=[1, 5],
        multimodal_lengths=[2, 3],
    )

    initialize_multimodal_encoder_request(request, max_num_tokens=16)

    assert request.py_mm_encoder_state is None


def test_external_encoder_outputs_commit_without_local_encoder():
    cache = TensorLRUCache(1 << 20, name="test")

    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, selected_items):
            raise AssertionError("external outputs must not run the local encoder")

    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    engine.model._multimodal_encoder_cache = cache
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: True)
    outputs = [torch.arange(8, dtype=torch.float32).reshape(2, 4)]
    request = _llm_request(
        1,
        multimodal_data={
            "multimodal_embedding_lengths": [2],
            "encoder_token_lengths": [8],
        },
        multimodal_positions=[1],
        multimodal_lengths=[2],
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=16)
    state = request.py_mm_encoder_state
    assert state is not None
    request.py_multimodal_data["multimodal_embedding"] = outputs
    cache_key = ("mm_transient", request.request_id, 0)
    assert cache.acquire(cache_key, outputs[0].nbytes, retain_after_release=False) is not None
    state.set_item_cache_key(0, cache_key)

    engine.forward_multimodal_encoder_items([request], {request.request_id: [0]})

    cached = cache.get(cache_key)
    torch.testing.assert_close(cached, outputs[0])
    assert cached.data_ptr() != outputs[0].data_ptr()


def test_external_encoder_demand_and_completion_use_existing_reservation():
    cache = TensorLRUCache(1 << 20, name="test")

    class _Model(MultimodalModelMixin):
        pass

    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    engine.model._multimodal_encoder_cache = cache
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: True)
    request = _llm_request(
        1,
        multimodal_data={
            "multimodal_embedding_lengths": [2],
            "encoder_token_lengths": [8],
        },
        multimodal_positions=[1],
        multimodal_lengths=[2],
    )
    request.py_client_id = 17
    initialize_multimodal_encoder_request(request, max_num_tokens=16)
    state = request.py_mm_encoder_state
    assert state is not None
    output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    cache_key = ("mm_transient", request.request_id, 0)
    cache.acquire(cache_key, output.nbytes, retain_after_release=False)
    state.set_item_cache_key(0, cache_key)

    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.model_engine = engine
    executor._mm_encoder_is_local = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor._external_mm_encoder_demands = Queue()
    executor._external_mm_encoder_demand_queue = None
    executor.enable_iter_perf_stats = True
    executor.perf_manager = SimpleNamespace(
        borrow_forward_timing_events=lambda: pytest.fail(
            "P-side external demand must not be timed as local encoder work"
        )
    )
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(rank=0, is_first_pp_rank=True, pp_size=1)
    executor.global_rank = 0
    executor.executor_request_queue = ExecutorRequestQueue(
        dist=executor.dist,
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
    )

    scheduled = ScheduledRequests()
    scheduled.scheduled_mm_encoder_items = {request.request_id: [0]}
    executor._forward_multimodal_encoder_step(scheduled)

    assert scheduled.mm_encoder_gpu_start_event is None
    assert scheduled.mm_encoder_gpu_end_event is None
    assert executor.take_multimodal_encoder_demands() == [(17, [0])]
    executor.dist.rank = 1
    executor._publish_external_mm_encoder_demands({request.request_id: [0]})
    assert executor.take_multimodal_encoder_demands() == []
    executor.dist.rank = 0

    handle = SharedTensorContainer.from_tensor(output).dump_to_dict()
    executor.enqueue_multimodal_encoder_outputs(17, [0], [handle])
    queue_items = [executor.executor_request_queue.request_queue.get_nowait()]
    assert queue_items[0].is_mm_encoder_completion
    assert executor._handle_special_queue_items(queue_items) == []

    cached = cache.get(cache_key)
    torch.testing.assert_close(cached, output)
    assert cached.data_ptr() != output.data_ptr()
    assert "multimodal_embedding" not in request.py_multimodal_data

    duplicate_handle = SharedTensorContainer.from_tensor(
        torch.full_like(output, -1), local=True
    ).dump_to_dict()
    executor._commit_external_mm_encoder_completion((17, [0], [duplicate_handle], None))

    torch.testing.assert_close(cache.get(cache_key), output)
    with pytest.raises(RuntimeError, match="REBUILD_LOCAL tensor missing"):
        SharedTensorContainer.from_dict(duplicate_handle).get_local_view()


def test_stale_external_encoder_completion_consumes_output_handles():
    cache = TensorLRUCache(1 << 20, name="test")
    stale_handle = SharedTensorContainer.from_tensor(torch.ones(2, 4), local=True).dump_to_dict()
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor.model_engine = SimpleNamespace(mm_encoder_cache=cache)

    executor._commit_external_mm_encoder_completion((17, [0], [stale_handle], None))

    with pytest.raises(RuntimeError, match="REBUILD_LOCAL tensor missing"):
        SharedTensorContainer.from_dict(stale_handle).get_local_view()


def test_external_encoder_error_is_scoped_to_its_request():
    failed = _request(1, [4])
    failed.py_client_id = 17
    unrelated = _request(2, [4])
    handled = []
    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed, unrelated]
    executor.model_engine = SimpleNamespace(mm_encoder_cache=TensorLRUCache(1 << 20, name="test"))
    executor._owns_mm_encoder_cache_references = lambda: True
    executor._handle_errors = lambda error, **kwargs: handled.append((error, kwargs))
    completion = RequestQueueItem(
        MM_ENCODER_COMPLETION_REQUEST_ID,
        mm_encoder_completion=(17, [0], [], "encoder peer unavailable"),
    )

    assert executor._handle_special_queue_items([completion]) == []
    assert handled == [
        (
            "External MM encoder failed: encoder peer unavailable",
            {"requests": [failed], "charge_budget": False},
        )
    ]


def test_attention_dp_buffers_owner_encoder_demand_for_rank_state_gather():
    request = _request(1, [1])
    request.py_client_id = 17
    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(rank=1)
    executor._pending_external_mm_encoder_demands = []
    executor._external_mm_encoder_demands = Queue()
    executor._external_mm_encoder_demand_queue = None

    executor._publish_external_mm_encoder_demands({request.request_id: [0]})

    assert executor._pending_external_mm_encoder_demands == [(17, [0])]
    assert executor.take_multimodal_encoder_demands() == []


def test_proxy_sends_encoder_completion_through_request_ingress():
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    proxy.request_queue = Queue()
    output_handle = {"shape": [2, 4]}

    proxy.enqueue_multimodal_encoder_outputs(17, [0], [output_handle])

    completion = proxy.request_queue.get_nowait()
    assert completion == MultimodalEncoderCompletion(17, [0], [output_handle], None)


def test_registered_encoder_input_is_attached_to_item_request():
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    proxy.request_queue = Queue()
    params = MultimodalParams(multimodal_data={"image": {"pixels": torch.ones(2, 3)}})

    proxy.set_multimodal_encoder_input("input", params)

    assert proxy.request_queue.get_nowait() == MultimodalEncoderInput("input", params)

    executor = object.__new__(PyExecutor)
    executor._multimodal_encoder_inputs = {}
    executor._handle_special_queue_items(
        [
            RequestQueueItem(
                MM_ENCODER_INPUT_REQUEST_ID,
                mm_encoder_input=("input", params),
            )
        ]
    )
    request = _llm_request(
        1,
        multimodal_data={
            MULTIMODAL_ENCODER_INPUT_ID_KEY: "input",
            "multimodal_embedding_lengths": [3],
        },
        multimodal_positions=[10],
        multimodal_lengths=[3],
    )

    executor._attach_multimodal_encoder_input(request)

    assert request.py_multimodal_data["image"] is params.multimodal_data["image"]
    assert request.py_multimodal_data["multimodal_embedding_lengths"] == [3]
    assert MULTIMODAL_ENCODER_INPUT_ID_KEY not in request.py_multimodal_data
    executor._handle_special_queue_items(
        [
            RequestQueueItem(
                MM_ENCODER_INPUT_REQUEST_ID,
                mm_encoder_input=("input", None),
            )
        ]
    )
    assert executor._multimodal_encoder_inputs == {}


def test_encoder_input_release_waits_for_earlier_request():
    executor = object.__new__(PyExecutor)
    executor.control_requests = []
    executor.is_shutdown = False
    executor.dist = SimpleNamespace(rank=0)
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.executor_request_queue = SimpleNamespace(get_from_request_queue=lambda _timeout: [])
    executor.request_broadcaster = SimpleNamespace(broadcast=lambda requests: (requests, None))
    executor._multimodal_encoder_inputs = {}
    params = MultimodalParams(multimodal_data={"image": {"pixels": torch.ones(2, 3)}})
    register = RequestQueueItem(
        MM_ENCODER_INPUT_REQUEST_ID,
        mm_encoder_input=("input", params),
    )
    request = _llm_request(
        1,
        multimodal_data={
            MULTIMODAL_ENCODER_INPUT_ID_KEY: "input",
            "multimodal_embedding_lengths": [3],
        },
        multimodal_positions=[10],
        multimodal_lengths=[3],
    )
    request_item = RequestQueueItem(1, request=request)
    release = RequestQueueItem(
        MM_ENCODER_INPUT_REQUEST_ID,
        mm_encoder_input=("input", None),
    )
    executor.request_accumulated = [register, request_item, release]
    waiting_queue = FCFSWaitingQueue()

    executor._fetch_and_enqueue_requests(waiting_queue, total_num_active_requests=1)

    assert executor.request_accumulated == [release]
    assert "input" in executor._multimodal_encoder_inputs
    queued_request = waiting_queue.peek_request().request
    assert queued_request is request
    assert MULTIMODAL_ENCODER_INPUT_ID_KEY not in request.py_multimodal_data
    assert request.py_multimodal_data["image"] is params.multimodal_data["image"]

    executor._fetch_and_enqueue_requests(waiting_queue, total_num_active_requests=1)

    assert executor._multimodal_encoder_inputs == {}
    assert request.py_multimodal_data["image"] is params.multimodal_data["image"]


def test_complete_external_handoff_keeps_direct_path():
    request = _llm_request(
        1,
        multimodal_data={
            "multimodal_embedding": [torch.ones(2, 4)],
            "multimodal_embedding_lengths": [2],
            "encoder_token_lengths": [8],
        },
        multimodal_positions=[1],
        multimodal_lengths=[2],
    )

    initialize_multimodal_encoder_request(request, max_num_tokens=16)

    assert request.py_mm_encoder_state is None


def test_multimodal_scheduler_continues_with_later_proposed_requests():
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=10)
    first = _request(1, [7, 7])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {2: [0]}
    assert output.context_requests == [second]
    assert first.py_mm_encoder_state.item_cache_keys == [None, None]


def test_multimodal_scheduler_encodes_shared_cache_key_once():
    cache = TensorLRUCache(8)
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_batch_size=2,
        max_num_tokens=8,
        encoder_cache=cache,
        get_item_cache_keys=lambda _request: [("stable", 0)],
        bytes_per_encoder_embedding=4,
        retain_cache_entries=True,
        scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
    )
    first = _request(1, [4])
    second = _request(2, [4])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {first.request_id: [0]}
    assert output.context_requests == [first, second]
    assert first.py_mm_encoder_state.item_cache_keys == second.py_mm_encoder_state.item_cache_keys
    assert cache.stats().inflight_deduplications == 1


def test_scheduler_defers_items_beyond_output_byte_budget():
    # Budget hosts exactly one 1-row item (4 bytes): the second request's
    # item must wait even though the token budget would admit it
    # (acquire-before-compute).
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=1 << 20, cache_capacity=4)
    first = _request(1, [3])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [first]


def test_pinned_outputs_block_new_admissions_until_explicit_release():
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=1 << 20, cache_capacity=4)
    holder = _request(1, [3])
    newcomer = _request(2, [3])

    first_output = scheduler.schedule_request([holder], set())
    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    holder_cache_key = holder.py_mm_encoder_state.item_cache_keys[0]
    assert holder_cache_key is not None
    scheduler.encoder_cache.commit(holder_cache_key, torch.ones(1, dtype=torch.float32))

    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [holder]

    assert holder.py_mm_encoder_state.clear_item_cache_key(0) == holder_cache_key
    assert scheduler.encoder_cache.release(holder_cache_key) == holder_cache_key
    holder.py_mm_encoder_state = None
    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items == {2: [0]}


def test_scheduler_gets_only_items_needed_by_the_proposed_chunk():
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=5,
        base_scheduler=_BaseScheduler(chunk_size=3, chunk_unit_size=1),
    )
    request = _request(1, [5, 5])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [request]
    assert request.py_mm_encoder_state.item_cache_keys[0] is not None
    assert request.py_mm_encoder_state.item_cache_keys[1] is None


def test_default_scheduler_batches_all_request_items_when_they_fit():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
    )
    request = _request(1, [1, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == [request]
    assert output.mm_encoder_context_chunk_sizes is None


def test_default_scheduler_accumulates_items_without_llm_chunking():
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=1,
        cache_capacity=8,
        base_scheduler=SimpleScheduler(_CapacityScheduler(), _MicroBatchScheduler()),
    )
    request = _request(1, [1, 1])

    first_output = scheduler.schedule_request([request], set())
    first_cache_key = request.py_mm_encoder_state.item_cache_keys[0]
    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    assert first_output.context_requests == []
    assert first_output.mm_encoder_blocked_request_ids == [1]
    assert first_output.mm_encoder_context_chunk_sizes is None
    assert request.py_mm_encoder_state.item_cache_keys[1] is None

    scheduler.encoder_cache.commit(first_cache_key, torch.ones(1))
    second_output = scheduler.schedule_request([request], set())

    assert second_output.scheduled_mm_encoder_items == {1: [1]}
    assert second_output.context_requests == [request]
    assert second_output.mm_encoder_context_chunk_sizes is None


def test_external_encoder_demand_waits_for_cache_completion():
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=1,
        base_scheduler=_BaseScheduler(chunk_size=3, chunk_unit_size=1),
        encoder_outputs_ready_immediately=False,
    )
    request = _request(1, [1])

    first_output = scheduler.schedule_request([request], set())

    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    assert first_output.context_requests == [request]
    assert request.context_chunk_size == 1
    cache_key = request.py_mm_encoder_state.item_cache_keys[0]
    assert scheduler.encoder_cache.get(cache_key, record_stats=False) is None

    request.context_current_position = 1
    waiting_output = scheduler.schedule_request([request], set())

    assert waiting_output.scheduled_mm_encoder_items is None
    assert waiting_output.context_requests == []
    assert waiting_output.mm_encoder_blocked_request_ids == [1]
    assert request.py_mm_encoder_state.item_cache_keys == [cache_key]

    scheduler.encoder_cache.commit(cache_key, torch.ones(1))
    ready_output = scheduler.schedule_request([request], set())

    assert ready_output.scheduled_mm_encoder_items is None
    assert ready_output.context_requests == [request]


def test_external_encoder_demand_does_not_schedule_future_items_while_pending():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
        scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
        encoder_outputs_ready_immediately=False,
    )
    request = _request(1, [1, 2, 1])

    first_output = scheduler.schedule_request([request], set())
    first_cache_key = request.py_mm_encoder_state.item_cache_keys[0]
    assert scheduler.encoder_cache.stats().producer_misses == 1
    request.context_current_position = 1
    waiting_output = scheduler.schedule_request([request], set())

    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    assert waiting_output.scheduled_mm_encoder_items is None
    assert waiting_output.context_requests == []
    assert waiting_output.mm_encoder_blocked_request_ids == [1]
    assert request.py_mm_encoder_state.item_cache_keys == [
        first_cache_key,
        None,
        None,
    ]
    assert scheduler.encoder_cache.stats().producer_misses == 1

    scheduler.encoder_cache.commit(first_cache_key, torch.ones(1))
    next_output = scheduler.schedule_request([request], set())

    assert next_output.scheduled_mm_encoder_items == {1: [1]}
    assert next_output.context_requests == [request]
    assert request.context_chunk_size == 2
    assert scheduler.encoder_cache.stats().producer_misses == 2


def test_external_encoder_demand_blocks_whole_request_without_llm_chunking():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        base_scheduler=SimpleScheduler(_CapacityScheduler(), _MicroBatchScheduler()),
        encoder_outputs_ready_immediately=False,
    )
    request = _request(1, [1, 1])

    first_output = scheduler.schedule_request([request], set())

    assert first_output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert first_output.context_requests == []
    assert first_output.mm_encoder_blocked_request_ids == [1]
    cache_keys = list(request.py_mm_encoder_state.item_cache_keys)

    for cache_key in cache_keys:
        scheduler.encoder_cache.commit(cache_key, torch.ones(1))
    ready_output = scheduler.schedule_request([request], set())

    assert ready_output.scheduled_mm_encoder_items is None
    assert ready_output.context_requests == [request]


def test_default_scheduler_rejects_unshardable_outputs_without_llm_chunking():
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=1,
        cache_capacity=4,
        base_scheduler=SimpleScheduler(_CapacityScheduler(), _MicroBatchScheduler()),
    )
    request = _request(1, [1, 1])

    with pytest.raises(
        MultimodalEncoderRequestError,
        match="because LLM chunking is disabled",
    ) as exc_info:
        scheduler.schedule_request([request], set())

    assert exc_info.value.request_ids == frozenset({request.request_id})
    assert request.py_mm_encoder_state.item_cache_keys == [None, None]


def test_default_scheduler_reuses_cached_outputs_for_the_whole_request():
    key_calls = []
    scheduler = MultimodalScheduler(
        SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
        max_batch_size=2,
        max_num_tokens=2,
        encoder_cache=TensorLRUCache(8),
        get_item_cache_keys=lambda _request: key_calls.append(True) or None,
        bytes_per_encoder_embedding=4,
        retain_cache_entries=False,
        scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
    )
    request = _request(1, [1, 1])

    first_output = scheduler.schedule_request([request], set())
    for cache_key in request.py_mm_encoder_state.item_cache_keys:
        scheduler.encoder_cache.commit(cache_key, torch.ones(1))
    calls_after_first_plan = len(key_calls)

    second_output = scheduler.schedule_request([request], set())

    assert first_output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert second_output.scheduled_mm_encoder_items is None
    assert second_output.context_requests == [request]
    assert len(key_calls) == calls_after_first_plan


@pytest.mark.parametrize(
    ("max_batch_size", "max_num_tokens"),
    [(1, 2), (2, 1)],
)
def test_default_scheduler_uses_current_chunk_when_all_items_exceed_step_budget(
    max_batch_size, max_num_tokens
):
    scheduler = _scheduler(
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        cache_capacity=8,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
    )
    request = _request(1, [1, 1])

    first_output = scheduler.schedule_request([request], set())

    state = request.py_mm_encoder_state
    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    assert first_output.context_requests == [request]
    assert first_output.mm_encoder_context_chunk_sizes == {1: 3}
    assert state.item_cache_keys[0] is not None
    assert state.item_cache_keys[1] is None


def test_default_scheduler_releases_entries_when_the_whole_request_does_not_fit():
    scheduler = _scheduler(
        max_batch_size=4,
        max_num_tokens=4,
        cache_capacity=12,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
    )
    holder = _request(1, [1])
    newcomer = _request(2, [1, 1, 1])
    first_output = scheduler.schedule_request([holder], set())
    holder_cache_key = holder.py_mm_encoder_state.item_cache_keys[0]
    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    scheduler.encoder_cache.commit(holder_cache_key, torch.ones(1))

    output = scheduler.schedule_request([holder, newcomer], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [holder]
    assert newcomer.py_mm_encoder_state.item_cache_keys == [None, None, None]


def test_default_scheduler_uses_current_chunk_when_all_outputs_do_not_fit():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        cache_capacity=4,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
    )
    request = _request(1, [1, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [request]
    assert output.mm_encoder_context_chunk_sizes == {1: 3}


def test_default_scheduler_returns_to_whole_request_when_remaining_items_fit():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        cache_capacity=8,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
    )
    request = _request(1, [1, 1, 1])

    first_output = scheduler.schedule_request([request], set())
    first_cache_key = request.py_mm_encoder_state.clear_item_cache_key(0)
    scheduler.encoder_cache.commit(first_cache_key, torch.ones(1))
    assert scheduler.encoder_cache.release(first_cache_key) == first_cache_key
    request.context_current_position = 3

    second_output = scheduler.schedule_request([request], set())

    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    assert first_output.mm_encoder_context_chunk_sizes == {1: 3}
    assert second_output.scheduled_mm_encoder_items == {1: [1, 2]}
    assert second_output.context_requests == [request]
    assert second_output.mm_encoder_context_chunk_sizes is None


def test_default_scheduler_keeps_first_request_and_releases_later_request_entries():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        cache_capacity=20,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
    )
    fitting = _request(1, [1, 1])
    oversized = _request(2, [1, 1, 1])

    output = scheduler.schedule_request([fitting, oversized], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == [fitting]
    assert output.mm_encoder_blocked_request_ids == [2]
    assert output.mm_encoder_context_chunk_sizes is None
    assert oversized.py_mm_encoder_state.item_cache_keys == [None, None, None]


def test_default_scheduler_skips_items_covered_by_kv_cache_reuse():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        base_scheduler=SimpleScheduler(
            _CapacityScheduler(), _MicroBatchScheduler(chunk_size=3, chunk_unit_size=1)
        ),
    )
    request = _request(1, [1, 1])
    request.estimated_reusable_tokens = 2

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [1]}
    assert output.context_requests == [request]


def test_eager_policy_uses_leftover_budget_for_future_items():
    base_scheduler = _BaseScheduler(chunk_size=3, chunk_unit_size=1)
    default_request = _request(1, [1, 1])
    default_output = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        base_scheduler=base_scheduler,
    ).schedule_request([default_request], set())

    eager_request = _request(2, [1, 1])
    eager_output = _scheduler(
        max_batch_size=2,
        max_num_tokens=2,
        base_scheduler=_BaseScheduler(chunk_size=3, chunk_unit_size=1),
        scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
    ).schedule_request([eager_request], set())

    assert default_output.scheduled_mm_encoder_items == {1: [0]}
    assert eager_output.scheduled_mm_encoder_items == {2: [0, 1]}


def test_atomic_item_larger_than_token_budget_runs_alone():
    scheduler = _scheduler(
        max_batch_size=2,
        max_num_tokens=5,
        base_scheduler=_BaseScheduler(chunk_unit_size=1),
        scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
    )
    request = _request(1, [7, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [request]
    assert request.context_chunk_size == 3


def test_ready_cache_hit_does_not_consume_encoder_compute_budget():
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=1,
        retain_cache_entries=True,
    )
    first = _request(1, [1])
    second = _request(2, [1])
    first_key = _item_cache_keys(first)[0]
    scheduler.encoder_cache.put(first_key, torch.ones(1))

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {2: [0]}
    assert output.context_requests == [first, second]


def test_current_chunk_drops_a_future_item_to_free_cache_space():
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=4,
        cache_capacity=4,
        base_scheduler=_BaseScheduler(chunk_unit_size=1),
    )
    request = _request(1, [4, 4])
    future_cache_key = _item_cache_keys(request)[1]
    scheduler.encoder_cache.acquire(future_cache_key, 4, retain_after_release=False)
    scheduler.encoder_cache.commit(future_cache_key, torch.ones(1))
    request.py_mm_encoder_state.set_item_cache_key(1, future_cache_key)

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [request]
    assert request.context_chunk_size == 3
    assert output.mm_encoder_cache_removals == [future_cache_key]
    assert request.py_mm_encoder_state.item_cache_keys[1] is None


def test_preemption_releases_all_multimodal_cache_entries():
    class _PausingScheduler(_BaseScheduler):
        def schedule_request(self, requests, inflight_request_ids):
            del inflight_request_ids
            return SchedulerOutput([], [], [], list(requests), [], 0)

    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=4,
        cache_capacity=4,
        base_scheduler=_PausingScheduler(),
    )
    request = _request(1, [4])
    cache_key = _item_cache_keys(request)[0]
    scheduler.encoder_cache.acquire(cache_key, 4, retain_after_release=False)
    scheduler.encoder_cache.commit(cache_key, torch.ones(1))
    request.py_mm_encoder_state.set_item_cache_key(0, cache_key)

    output = scheduler.schedule_request([request], set())

    assert output.paused_requests == [request]
    assert output.mm_encoder_cache_removals == [cache_key]
    assert request.py_mm_encoder_state.item_cache_keys == [None]


def test_scheduler_ends_chunk_before_an_unavailable_item():
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=5,
        base_scheduler=_BaseScheduler(chunk_unit_size=1),
    )
    request = _request(1, [5, 5])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [request]
    assert request.context_chunk_size == 3
    assert output.mm_encoder_context_chunk_sizes == {1: 3}


def test_scheduler_without_chunking_releases_incomplete_item_selection():
    scheduler = _scheduler(max_batch_size=1, max_num_tokens=5)
    request = _request(1, [5, 5])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == []
    assert output.mm_encoder_blocked_request_ids == [1]
    assert request.py_mm_encoder_state.item_cache_keys == [None, None]


def test_request_rejects_one_item_output_larger_than_cache_budget():
    # A video item whose indivisible output can never fit the resident budget
    # fails at admission (failing only that request), with guidance to raise
    # encoder_max_num_tokens.
    request = _llm_request(
        1,
        multimodal_data={
            "video": {"pixel_values_videos": torch.empty(3, 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("video", 0)],
                encoder_token_lengths=[12],
                output_embedding_lengths=[3],
            ),
            "multimodal_embedding_lengths": [3],
        },
    )
    with pytest.raises(ValueError, match="raise encoder_max_num_tokens") as exc_info:
        initialize_multimodal_encoder_request(
            request,
            max_num_tokens=1 << 30,
            max_output_bytes=2 * 4,  # fits 2 rows; the video needs 3
            bytes_per_encoder_embedding=4,
        )
    assert "Multimodal request 1" in str(exc_info.value)
    assert "effective encoder_max_num_tokens is 1073741824" in str(exc_info.value)


def test_request_checks_largest_item_output_not_whole_request_output():
    request = _llm_request(
        1,
        multimodal_data={
            "image": {"pixel_values": torch.empty(2, 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[2, 2],
                output_embedding_lengths=[2, 2],
            ),
            "multimodal_embedding_lengths": [2, 2],
        },
    )

    initialize_multimodal_encoder_request(
        request,
        max_num_tokens=4,
        max_output_bytes=8,
        bytes_per_encoder_embedding=4,
    )

    assert request.py_mm_encoder_state.embedding_lengths == [2, 2]


def test_scheduler_requires_bytes_per_embedding_alongside_budget():
    with pytest.raises(ValueError, match="bytes_per_encoder_embedding"):
        MultimodalScheduler(
            _BaseScheduler(),
            max_batch_size=1,
            max_num_tokens=1,
            encoder_cache=TensorLRUCache(4),
            get_item_cache_keys=_item_cache_keys,
            bytes_per_encoder_embedding=0,
            retain_cache_entries=False,
            scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
        )


def test_multimodal_scheduler_selects_all_items_and_admits_request_when_batch_fits():
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=10)
    request = _request(1, [6, 4])

    output = scheduler.schedule_request([request], set())

    # The encoder step is the single encode site: an in-budget batch simply
    # has every pending item selected, and the request still enters the LLM
    # batch in the same iteration (encode runs before the LLM forward).
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == [request]


def test_multimodal_scheduler_preserves_non_multimodal_requests():
    scheduler = _scheduler(max_batch_size=1, max_num_tokens=1)
    request = _llm_request(1)
    initialize_multimodal_encoder_request(request, max_num_tokens=1)

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_qwen3_output_budget_uses_post_merge_embedding_capacity():
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    engine = object.__new__(PyTorchModelEngine)
    engine.max_num_tokens = 8192
    engine.encoder_max_num_tokens = 65536
    engine.input_processor = processor
    engine._get_mm_encoder_embedding_size_bytes = lambda: 32768

    budget = engine._compute_mm_encoder_output_budget_bytes()

    assert engine.bytes_per_mm_encoder_embedding == 32768
    assert budget == 512 * 1024**2


def test_output_row_bytes_use_config_dtype_without_pp_embedding_weight():
    engine = object.__new__(PyTorchModelEngine)
    engine.model = SimpleNamespace(
        embedding_dim=16384,
        model_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    )

    assert engine._get_mm_encoder_embedding_size_bytes() == 32768


def test_downstream_pp_item_scheduler_does_not_create_encoder_store():
    class _CacheModel(MultimodalModelMixin):
        supports_encoder_cache = True

    model = _CacheModel()
    model.model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(encoder_cache_max_bytes=64)
    )
    model._multimodal_encoder_cache = None
    engine = object.__new__(PyTorchModelEngine)
    engine.model = model
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: False)

    assert engine.mm_encoder_cache is None
    assert model._multimodal_encoder_cache is None


def test_output_budget_requires_processor_embedding_capacity():
    engine = object.__new__(PyTorchModelEngine)
    engine.encoder_max_num_tokens = 65536
    engine.input_processor = SimpleNamespace(get_max_mm_encoder_output_embeddings=lambda *_: None)

    with pytest.raises(ValueError, match="get_max_mm_encoder_output_embeddings"):
        engine._compute_mm_encoder_output_budget_bytes()


def test_eager_compatibility_is_checked_only_for_item_scheduled_models():
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
            encoder_side_stream_max_ahead=0,
        ),
        enable_attention_dp=True,
        cache_transceiver_config=SimpleNamespace(backend="NIXL"),
    )

    _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="attention DP"):
        _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)

    args.enable_attention_dp = False
    with pytest.raises(ValueError, match="disaggregated"):
        _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_side_stream_compatibility_is_checked_only_for_item_scheduled_models():
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
            encoder_side_stream_max_ahead=1,
        ),
        enable_attention_dp=False,
        cache_transceiver_config=None,
    )

    _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="side-stream prefetch"):
        _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_request_rejects_item_above_effective_startup_maximum():
    request = _request(1, [9])
    request.py_multimodal_data["image"] = {"pixel_values": torch.empty(1)}

    with pytest.raises(ValueError, match="exceeding the effective startup maximum 8"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_eager_policy_uses_leftover_budget_outside_current_llm_batch():
    base_scheduler = _BaseScheduler()
    base_scheduler.capacity_scheduler = _RejectMultimodalCapacityScheduler()
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=8,
        base_scheduler=base_scheduler,
        scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
    )
    multimodal_request = _request(1, [8])
    text_request = _llm_request(2)
    initialize_multimodal_encoder_request(text_request, max_num_tokens=8)

    output = scheduler.schedule_request([multimodal_request, text_request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [text_request]


def test_multimodal_scheduler_remote_result_skips_local_cache_policy():
    scheduler = _scheduler(max_batch_size=1, max_num_tokens=4)
    blocked = _request(1, [4])
    text_request = _llm_request(2)

    output = scheduler.schedule_request_with_mm_decisions(
        [blocked, text_request],
        set(),
        blocked_request_ids=[blocked.request_id],
        scheduled_items={blocked.request_id: [0]},
        cache_removals=[("old", 0)],
        context_chunk_sizes={blocked.request_id: 0},
    )

    assert output.context_requests == [text_request]
    assert output.scheduled_mm_encoder_items == {blocked.request_id: [0]}
    assert output.mm_encoder_cache_removals == [("old", 0)]
    assert blocked.context_chunk_size == 0
    assert blocked.py_mm_encoder_state.item_cache_keys == [None]
    assert len(scheduler.encoder_cache) == 0


def test_forward_multimodal_encoder_step_scopes_failure_to_item_owners():
    failed = _request(1, [4])
    follower = _request(4, [4])
    unrelated_mm_context = _request(5, [4])
    unrelated_context = _llm_request(2)
    unrelated_generation = _llm_request(3)
    handled = []

    def fail_encoder(*_, **__):
        raise MultimodalEncoderRequestError(
            "bad MM output", request_ids={failed.request_id, follower.request_id}
        )

    executor = object.__new__(PyExecutor)
    executor.active_requests = [
        failed,
        follower,
        unrelated_mm_context,
        unrelated_context,
        unrelated_generation,
    ]
    executor.enable_attention_dp = False
    executor.enable_iter_perf_stats = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(world_size=1, is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_encoder)
    executor._handle_errors = lambda error_msg, **kwargs: handled.append((error_msg, kwargs))

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests(
        [failed, follower, unrelated_mm_context, unrelated_context]
    )
    scheduled_requests.append_generation_request(unrelated_generation)
    scheduled_requests.scheduled_mm_encoder_items = {failed.request_id: [0]}

    result = executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [unrelated_mm_context, unrelated_context]
    assert scheduled_requests.generation_requests == [unrelated_generation]
    assert scheduled_requests.scheduled_mm_encoder_items is None
    assert result == ("bad MM output", [failed.request_id, follower.request_id])
    assert handled == [
        (
            "bad MM output",
            {"requests": [failed, follower], "charge_budget": False},
        )
    ]


def test_forward_multimodal_encoder_step_skips_empty_cache_delta():
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        run_multimodal_encoder_schedule=lambda *_args, **_kwargs: pytest.fail(
            "empty MM cache deltas must not be replayed"
        )
    )
    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([_llm_request(1)])

    assert executor._forward_multimodal_encoder_step(scheduled_requests) is None


def test_forward_multimodal_encoder_step_contains_model_contract_error():
    failed = _request(1, [4])
    failed.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = ("image", 0)
    unrelated = _llm_request(2)
    handled = []

    def fail_apply(*_, **__):
        raise MultimodalEncoderRequestError(
            "multimodal_encoder_item_metadata must be a MultimodalEncoderItemMetadata"
        )

    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed, unrelated]
    executor.enable_attention_dp = False
    executor.enable_iter_perf_stats = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(world_size=1, is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_apply)
    executor._handle_errors = lambda error_msg, **kwargs: handled.append((error_msg, kwargs))

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([failed, unrelated])
    scheduled_requests.scheduled_mm_encoder_items = {failed.request_id: [0]}

    executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [unrelated]
    assert scheduled_requests.scheduled_mm_encoder_items is None
    assert len(handled) == 1
    assert "must be a MultimodalEncoderItemMetadata" in handled[0][0]
    assert handled[0][1] == {"requests": [failed], "charge_budget": False}


def test_forward_multimodal_encoder_step_contains_stale_schedule():
    unrelated = _llm_request(2)
    handled = []

    def fail_apply(*_, **__):
        raise MultimodalEncoderRequestError("Scheduled MM request 1 is no longer active")

    executor = object.__new__(PyExecutor)
    executor.active_requests = [unrelated]
    executor.enable_attention_dp = False
    executor.enable_iter_perf_stats = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(world_size=1, is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_apply)
    executor._handle_errors = lambda error_msg, **kwargs: handled.append((error_msg, kwargs))

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([unrelated])
    scheduled_requests.scheduled_mm_encoder_items = {1: [0]}

    executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [unrelated]
    assert scheduled_requests.scheduled_mm_encoder_items is None
    assert handled == [
        (
            "Scheduled MM request 1 is no longer active",
            {
                "requests": [],
                "charge_budget": False,
            },
        )
    ]


def test_forward_multimodal_encoder_step_propagates_system_errors():
    failed = _request(1, [4])

    def fail_encoder(*_, **__):
        raise torch.cuda.OutOfMemoryError("encoder OOM")

    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed]
    executor.enable_attention_dp = False
    executor.enable_iter_perf_stats = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_encoder)

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([failed])
    scheduled_requests.scheduled_mm_encoder_items = {failed.request_id: [0]}

    with pytest.raises(torch.cuda.OutOfMemoryError, match="encoder OOM"):
        executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [failed]
    assert scheduled_requests.scheduled_mm_encoder_items == {failed.request_id: [0]}


def test_item_encoder_slices_and_restores_selected_item_order():
    class _Model(MultimodalModelMixin):
        def encode_multimodal_inputs(self, multimodal_params):
            return torch.cat(
                [param.multimodal_data["image"]["pixel_values"] for param in multimodal_params]
            )

    multimodal_param = MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(5).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[2, 3],
                output_embedding_lengths=[2, 3],
            ),
            "multimodal_embedding_lengths": [2, 3],
        }
    )

    model = _Model()
    encoder_inputs = model.prepare_multimodal_encoder_inputs(
        [(multimodal_param, 1), (multimodal_param, 0)]
    )
    outputs = model.forward_multimodal_encoder_items(encoder_inputs)

    assert [output.squeeze(1).tolist() for output in outputs] == [
        [2, 3, 4],
        [0, 1],
    ]


def test_prepare_multimodal_encoder_inputs_slices_before_device_transfer():
    multimodal_param = MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(5).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[2, 3],
                output_embedding_lengths=[2, 3],
            ),
            "multimodal_embedding_lengths": [2, 3],
        }
    )

    encoder_inputs = MultimodalModelMixin.prepare_multimodal_encoder_inputs(
        MultimodalModelMixin(), [(multimodal_param, 1)]
    )

    item_param, embedding_lengths, modality = encoder_inputs[0]
    assert modality == "image"
    assert embedding_lengths == [3]
    assert item_param.multimodal_data["image"]["pixel_values"].squeeze(1).tolist() == [2, 3, 4]
    assert multimodal_param.multimodal_data["image"]["pixel_values"].shape[0] == 5


def test_prepare_multimodal_encoder_inputs_rejects_invalid_metadata_types():
    multimodal_param = MultimodalParams(
        multimodal_data={
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: ("image", 0),
            "multimodal_embedding_lengths": [1],
        }
    )

    with pytest.raises(
        MultimodalEncoderContractError, match="must be a MultimodalEncoderItemMetadata"
    ):
        MultimodalModelMixin().prepare_multimodal_encoder_inputs([(multimodal_param, 0)])


def test_strip_mm_encoder_inputs_preserves_embedding_and_runtime_metadata():
    embedding = torch.empty(3, 4)
    mm_data = {
        "image": {"pixel_values": torch.empty(2, 3)},
        "video": {"pixel_values_videos": torch.empty(2, 3)},
        "multimodal_embedding": embedding,
        "multimodal_embed_mask_cumsum": torch.tensor([0, 1]),
    }

    strip_mm_encoder_inputs(mm_data)

    assert "image" not in mm_data
    assert "video" not in mm_data
    assert mm_data["multimodal_embedding"] is embedding
    assert "multimodal_embed_mask_cumsum" in mm_data


@pytest.mark.parametrize("output_ready", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_terminate_request_releases_multimodal_cache_refs_idempotently(pp_size, output_ready):
    request = _request(1, [4, 4])
    state = request.py_mm_encoder_state
    cache = TensorLRUCache(16)
    cache_key = ("mm_transient", request.request_id, 0)
    cache.acquire(cache_key, 4, retain_after_release=False)
    if output_ready:
        cache.commit(cache_key, torch.ones(1))
    state.set_item_cache_key(0, cache_key)
    freed = []

    executor = object.__new__(PyExecutor)
    executor._mm_encoder_item_scheduling_enabled = True
    executor.enable_attention_dp = False
    executor.global_rank = 0
    executor.model_engine = SimpleNamespace(mm_encoder_cache=cache)
    executor._pending_mm_encoder_cache_removals = []
    executor.resource_manager = SimpleNamespace(free_resources=freed.append)
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_timed_out_ctx_cancelled_ids = {request.py_request_id}
    executor._disagg_timed_out_gen_cancelled_ids = {request.py_request_id}
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0, is_first_pp_rank=True, pp_size=pp_size)
    executor.result_wait_queues = {}

    executor._do_terminate_request(request)
    executor._release_multimodal_resources(request)

    assert freed == [request]
    assert request.py_mm_encoder_state is None
    assert request.py_multimodal_data == {}
    expected_removals = [cache_key] if output_ready and pp_size == 2 else []
    assert executor._pending_mm_encoder_cache_removals == expected_removals
    cache_stats = cache.stats()
    assert cache_stats.current_bytes == 0
    assert cache_stats.reserved_bytes == 0
    assert cache_stats.in_use_bytes == 0
    assert executor._prefetched_request_ids == set()
    assert executor._disagg_timed_out_ctx_cancelled_ids == set()
    assert executor._disagg_timed_out_gen_cancelled_ids == set()


def test_completed_context_chunk_releases_only_consumed_item_refs():
    request = _request(1, [4, 4])
    state = request.py_mm_encoder_state
    cache = TensorLRUCache(8)
    item_cache_keys = [("mm_transient", request.request_id, item_idx) for item_idx in range(2)]
    for item_idx, cache_key in enumerate(item_cache_keys):
        cache.acquire(cache_key, 4, retain_after_release=False)
        cache.commit(cache_key, torch.ones(1))
        state.set_item_cache_key(item_idx, cache_key)

    executor = object.__new__(PyExecutor)
    executor._mm_encoder_item_scheduling_enabled = True
    executor.enable_attention_dp = False
    executor.global_rank = 0
    executor.dist = SimpleNamespace(is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(mm_encoder_cache=cache)
    executor._pending_mm_encoder_cache_removals = []
    request.context_current_position = 3

    executor._release_consumed_mm_item_entries(request)

    assert state.item_cache_keys == [None, item_cache_keys[1]]
    assert executor._pending_mm_encoder_cache_removals == []
    assert cache.get(item_cache_keys[0], record_stats=False) is None
    assert cache.get(item_cache_keys[1], record_stats=False) is not None


def test_weight_invalidation_clears_old_removal_delta_and_rejects_live_refs():
    invalidations = []
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor.model_engine = SimpleNamespace(
        invalidate_multimodal_encoder_cache=lambda: invalidations.append(True)
    )
    executor._pending_mm_encoder_cache_removals = [("old", 0)]

    executor.invalidate_multimodal_encoder_cache()

    assert invalidations == [True]
    assert executor._pending_mm_encoder_cache_removals == []

    request = _request(1, [4])
    request.py_mm_encoder_state.set_item_cache_key(0, ("cache", 0))
    executor.active_requests = [request]
    with pytest.raises(RuntimeError, match="live multimodal cache references"):
        executor.invalidate_multimodal_encoder_cache()
    assert invalidations == [True]


def test_model_engine_invalidation_advances_encoder_version():
    class _Model(MultimodalModelMixin):
        pass

    model = _Model()
    model._multimodal_encoder_cache = TensorLRUCache(16)
    cache_key = ("old", 0)
    model._multimodal_encoder_cache.put(cache_key, torch.ones(1))

    engine = object.__new__(PyTorchModelEngine)
    engine.model = model
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: True)

    engine.invalidate_multimodal_encoder_cache()

    assert engine.mm_encoder_version == 1
    assert model._multimodal_encoder_cache.get(cache_key) is None


def test_executor_stamps_encoder_version_when_admitting_cached_request():
    request = _llm_request(1, multimodal_data={"image": {}})
    executor = object.__new__(PyExecutor)
    executor.waiting_queue = []
    executor.active_requests = []
    executor.model_engine = SimpleNamespace(mm_encoder_cache=object(), mm_encoder_version=7)
    executor._mm_encoder_item_scheduling_enabled = False
    executor._fetch_new_requests = lambda *_: [request]
    executor._validate_request = lambda _: None

    assert executor._fetch_and_activate_new_requests() == [request]
    assert request.py_multimodal_data["mm_encoder_version"] == 7


def test_item_outputs_commit_to_prompt_ordered_cache_entries(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")

    class _Model(MultimodalModelMixin):
        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    engine.model._multimodal_encoder_cache = cache
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: True)
    engine.bytes_per_mm_encoder_embedding = 8
    request = _llm_request(
        1,
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(5).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[2, 3],
                output_embedding_lengths=[2, 3],
            ),
            "multimodal_embedding_lengths": [2, 3],
        },
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=8)
    state = request.py_mm_encoder_state
    item_cache_keys = [
        ("mm_transient", request.request_id, item_idx) for item_idx in range(state.num_items)
    ]
    for item_idx, (cache_key, rows) in enumerate(
        zip(item_cache_keys, state.embedding_lengths, strict=True)
    ):
        cache.acquire(cache_key, rows * 8, retain_after_release=False)
        state.set_item_cache_key(item_idx, cache_key)

    engine.forward_multimodal_encoder_items([request], {1: [0]})

    torch.testing.assert_close(cache.get(item_cache_keys[0]), torch.full((2, 2), 2.0))
    assert "image" in request.py_multimodal_data

    engine.forward_multimodal_encoder_items([request], {1: [1]})

    torch.testing.assert_close(cache.get(item_cache_keys[1]), torch.full((3, 2), 3.0))
    assert "multimodal_embedding" not in request.py_multimodal_data
    assert "image" in request.py_multimodal_data


def test_llm_data_keeps_only_the_item_slices_used_by_this_chunk():
    cache = TensorLRUCache(20)

    class _Model(MultimodalModelMixin):
        @property
        def text_embedding_layer(self):
            return SimpleNamespace(weight=torch.empty(0, 1))

        @property
        def embedding_dim(self):
            return 1

    cumsum = torch.tensor([0, 1, 2, 3, 3, 4, 5, 5], dtype=torch.int64)
    request = _llm_request(
        1,
        multimodal_data={
            "image": {"pixel_values": torch.empty(2, 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[3, 2],
                output_embedding_lengths=[3, 2],
            ),
            "multimodal_embedding_lengths": [3, 2],
            "multimodal_embed_mask_cumsum": cumsum,
        },
        input_tokens=list(range(cumsum.numel())),
        multimodal_positions=[1, 5],
        multimodal_lengths=[3, 2],
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=8)
    state = request.py_mm_encoder_state
    first = torch.tensor([[0.0], [1.0], [2.0]])
    second = torch.tensor([[3.0], [4.0]])
    for item_idx, value in enumerate((first, second)):
        cache_key = ("cache", item_idx)
        cache.acquire(
            cache_key,
            value.numel() * value.element_size(),
            retain_after_release=True,
        )
        cache.commit(cache_key, value)
        state.set_item_cache_key(item_idx, cache_key)

    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    engine.model._multimodal_encoder_cache = cache
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: True)
    runtime = MultimodalRuntimeData(
        past_seen_token_num=2,
        chunk_end_pos=6,
        embed_mask_cumsum=cumsum,
    )

    mm_data = engine._build_multimodal_data_for_llm(request, runtime)

    assert "image" not in mm_data
    assert "image" in request.py_multimodal_data
    segments = mm_data["multimodal_embedding"]
    assert isinstance(segments, tuple)
    assert len(segments) == 2
    torch.testing.assert_close(segments[0], first[1:])
    torch.testing.assert_close(segments[1], second[:1])
    gathered = engine.model._get_or_encode_multimodal_embeddings(
        [MultimodalParams(multimodal_data=mm_data, multimodal_runtime=runtime)]
    )
    assert gathered[0] is segments[0]
    assert gathered[1] is segments[1]


# ---------------------------------------------------------------------------
# MultimodalEncoderRequestState unit behavior
# ---------------------------------------------------------------------------


def test_mm_encoder_state_enforces_lengths_slot_invariant():
    with pytest.raises(ValueError, match="one cache key per item slot"):
        MultimodalEncoderRequestState(
            embedding_lengths=[2],
            encoder_token_lengths=[4],
            item_cache_keys=[None, None],
        )


def test_mm_encoder_state_copies_validated_scheduler_costs_at_admission():
    request = _request(1, [4, 7])

    assert request.py_mm_encoder_state.encoder_token_lengths == [4, 7]

    metadata = request.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    metadata.encoder_token_lengths[0] = 100
    assert request.py_mm_encoder_state.encoder_token_lengths == [4, 7]

    scheduler = _scheduler(max_batch_size=2, max_num_tokens=11)
    output = scheduler.schedule_request([request], set())
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}


def test_mm_encoder_state_finds_items_by_embedding_rows():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 0, 3])

    assert state.embedding_row_offsets == [0, 2, 2, 5]
    assert state.items_overlapping_embedding_rows(0, 2) == [0]
    assert state.items_overlapping_embedding_rows(1, 3) == [0, 2]
    assert state.items_overlapping_embedding_rows(2, 5) == [2]
    assert state.items_overlapping_embedding_rows(5, 5) == []


def test_mm_encoder_state_rejects_replacing_an_item_cache_key():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2])
    state.set_item_cache_key(0, ("cache", 0))

    with pytest.raises(RuntimeError, match="already has a cache key"):
        state.set_item_cache_key(0, ("cache", 1))
