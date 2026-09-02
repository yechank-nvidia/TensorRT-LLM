# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequestState,
    PyResult,
    get_multimodal_embedding_lengths,
)
from tensorrt_llm._torch.pyexecutor.sampler import EarlyStopWithMMResult, MultimodalResult
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.inputs.multimodal import DisaggPrefillMultimodalInputs
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    "req,expected",
    [
        (
            SimpleNamespace(multimodal_lengths=[3, 5], py_multimodal_data=None),
            None,
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[6, 5],
                py_multimodal_data={"multimodal_embedding_lengths": [5, 3]},
            ),
            [5, 3],
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[6, 5],
                py_multimodal_data={
                    "layout_metadata": {
                        "multimodal_embedding_lengths": [6, 4],
                    },
                },
            ),
            None,
        ),
    ],
)
@pytest.mark.cpu_only
def test_multimodal_embedding_lengths_returns_top_level_metadata(req, expected):
    """Getter reads top-level lengths and ignores layout metadata."""
    assert get_multimodal_embedding_lengths(req) == expected


@pytest.mark.parametrize(
    "req,exception,match",
    [
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": [3, 1]},
            ),
            ValueError,
            "length must match",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": [5]},
            ),
            ValueError,
            "exceeds",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": [-1]},
            ),
            ValueError,
            "non-negative",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={
                    "multimodal_embedding_lengths": torch.tensor([4]),
                },
            ),
            TypeError,
            "must be a list",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": (4,)},
            ),
            TypeError,
            "must be a list",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data=["multimodal_embedding_lengths", [4]],
            ),
            TypeError,
            "py_multimodal_data must be a dict",
        ),
    ],
)
@pytest.mark.cpu_only
def test_multimodal_embedding_lengths_rejects_invalid_metadata(req, exception, match):
    """Bad length metadata is rejected by the getter."""
    with pytest.raises(exception, match=match):
        get_multimodal_embedding_lengths(req)


class _FakePyResult:
    def __init__(self):
        self.mm_embeddings = []
        self.multimodal_layout = None
        self.mrope_position = None

    def append_mm_embeddings(self, mm_embedding, mm_embedding_lengths, multimodal_layout=None):
        self.mm_embeddings.append((mm_embedding, mm_embedding_lengths))
        self.multimodal_layout = multimodal_layout

    def set_mrope_position(self, position_ids, position_deltas):
        self.mrope_position = (position_ids, position_deltas)


class _FakeRequest:
    def __init__(
        self,
        multimodal_lengths=None,
        multimodal_positions=None,
        py_multimodal_data=None,
        tokens=None,
    ):
        self.multimodal_lengths = multimodal_lengths
        self.multimodal_positions = multimodal_positions
        self.multimodal_item_run_cu_offsets = None
        self.multimodal_run_positions = None
        self.multimodal_run_lengths = None
        self.py_multimodal_data = py_multimodal_data
        self._tokens = tokens or []
        self.py_result = _FakePyResult()
        self.state = None
        self.finished_reason = None

    def get_tokens(self, beam):
        assert beam == 0
        return self._tokens

    def set_finished_reason(self, reason, beam):
        self.finished_reason = (reason, beam)


@pytest.mark.cpu_only
def test_mm_encoder_sampler_aligns_mixed_batch_by_request_index():
    """Sparse MM encoder outputs attach to the original request index."""
    text_request = _FakeRequest()
    mm_request = _FakeRequest(multimodal_lengths=[4])
    sampler = EarlyStopWithMMResult()
    state = sampler.SampleState(
        requests=[text_request, mm_request],
        data=MultimodalResult(
            mm_embeddings=[torch.ones(4, 2)],
            mm_embedding_request_indices=[1],
            mm_embedding_lengths=[[4]],
            num_context_requests=2,
            extra_data={
                "mrope_position_ids": ["text-pos", "mm-pos"],
                "mrope_position_deltas": ["text-delta", "mm-delta"],
            },
        ),
    )

    sampler.update_requests(state)

    assert text_request.state == LlmRequestState.GENERATION_COMPLETE
    assert mm_request.state == LlmRequestState.GENERATION_COMPLETE
    assert text_request.finished_reason == (FinishReason.LENGTH, 0)
    assert mm_request.finished_reason == (FinishReason.LENGTH, 0)
    assert len(text_request.py_result.mm_embeddings) == 0
    [(mm_embedding, mm_embedding_lengths)] = mm_request.py_result.mm_embeddings
    assert mm_embedding.shape == (4, 2)
    assert mm_embedding_lengths == [4]
    assert text_request.py_result.mrope_position is None
    assert mm_request.py_result.mrope_position == ("mm-pos", "mm-delta")


@pytest.mark.cpu_only
def test_disagg_prefill_reuses_encoder_side_multimodal_layout():
    """Prefill adopts the encoder layout without calling the legacy rebuilder."""

    class _InputProcessor:
        support_mm_disagg = True
        mm_bidirectional_blocks = False

        def build_disagg_prefill_multimodal_inputs(self, inputs, mm_handles):
            raise AssertionError("encoder-provided layout should be reused")

        def get_vocab_size(self):
            raise AssertionError("encoder cumsum should be reused")

        def get_mm_token_ids(self):
            return torch.tensor([99])

        def get_mm_special_token_ids(self):
            return None

    cumsum = torch.tensor([0, 1, 2, 2], dtype=torch.int64)
    layout = DisaggPrefillMultimodalInputs(
        prompt_token_ids=[7, 99, 99, 8],
        multimodal_lengths=[2],
        multimodal_positions=[1],
        multimodal_embedding_lengths=[2],
        encoder_token_lengths=[8],
        multimodal_item_run_cu_offsets=[0, 1],
        multimodal_run_positions=[1],
        multimodal_run_lengths=[2],
        multimodal_embed_mask_cumsum=cumsum,
    )
    disaggregated_params = DisaggregatedParams(
        multimodal_embedding_handles=[{"tensor_size": [2, 4]}],
        multimodal_layout=layout,
    )
    llm = object.__new__(BaseLLM)
    llm.args = SimpleNamespace(backend="pytorch")
    llm._hf_model_config = SimpleNamespace(is_encoder_decoder=False)
    llm.input_processor = _InputProcessor()

    prompt_token_ids, _, multimodal_params, _ = llm._preprocess(
        {"prompt": "unused by the encoder-provided layout"},
        SamplingParams(),
        disaggregated_params,
    )

    assert prompt_token_ids == layout.prompt_token_ids
    assert multimodal_params.multimodal_input.multimodal_positions == [1]
    assert multimodal_params.multimodal_input.multimodal_lengths == [2]
    assert multimodal_params.multimodal_data["multimodal_embedding_lengths"] == [2]
    assert multimodal_params.multimodal_data["encoder_token_lengths"] == [8]
    assert multimodal_params.multimodal_data["multimodal_embed_mask_cumsum"] is cumsum


@pytest.mark.cpu_only
def test_mm_encoder_sampler_carries_embed_cumsum_in_layout():
    """Encoder results snapshot prompt metadata before request cleanup."""
    cumsum = torch.tensor([0, 1, 2, 2], dtype=torch.int64)
    request = _FakeRequest(
        multimodal_lengths=[2],
        multimodal_positions=[1],
        py_multimodal_data={
            "encoder_token_lengths": [8],
            "multimodal_embedding_lengths": [2],
            "multimodal_embed_mask_cumsum": cumsum,
        },
        tokens=[7, 99, 99, 8],
    )
    sampler = EarlyStopWithMMResult()
    scheduled_requests = SimpleNamespace(
        generation_requests=[], context_requests=[request], num_context_requests=1
    )
    state = sampler.sample_async(
        scheduled_requests,
        {
            "mm_embeddings": [torch.ones(2, 4)],
            "mm_embedding_request_indices": [0],
            "mm_embedding_lengths": [[2]],
        },
        [],
    )

    request.py_multimodal_data.clear()
    sampler.update_requests(state)

    assert request.py_result.multimodal_layout is not None
    assert request.py_result.multimodal_layout.encoder_token_lengths == [8]
    assert request.py_result.multimodal_layout.multimodal_embed_mask_cumsum is cumsum


@pytest.mark.cpu_only
def test_py_result_mm_embedding_handles_use_shared_tensor_handles():
    """MM encoder result handles should preserve the producer tensor device."""
    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    layout = DisaggPrefillMultimodalInputs(
        prompt_token_ids=[99, 99, 99, 99],
        multimodal_lengths=[1, 3],
        multimodal_positions=[0, 1],
        multimodal_embedding_lengths=[1, 3],
        multimodal_embed_mask_cumsum=torch.arange(1, 5, dtype=torch.int64),
    )

    result.append_mm_embeddings(source, [1, 3], multimodal_layout=layout)

    handles = result.mm_embedding_handles
    assert handles is not None
    assert [handle["method_key"] for handle in handles] == [2, 2]
    restored = [SharedTensorContainer.from_dict(handle).get_local_view() for handle in handles]
    assert torch.equal(torch.cat(restored, dim=0), source)

    follower = PyResult(prompt_len=1, max_new_tokens=1)
    follower.apply_diff(result.get_diff())
    assert follower.multimodal_layout == layout
    assert follower.multimodal_layout.multimodal_embed_mask_cumsum is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_py_result_cuda_mm_embedding_handles_stay_cuda_backed():
    """CUDA MM encoder outputs should not move to CPU for handoff handles."""
    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)

    result.append_mm_embeddings(source, [2, 2])

    handles = result.mm_embedding_handles
    assert handles is not None
    assert [handle["method_key"] for handle in handles] == [1, 1]


class _FakeScheduledRequests:
    def __init__(self, num_context_requests):
        self.generation_requests = []
        self.context_requests = [_FakeRequest() for _ in range(num_context_requests)]

    @property
    def num_context_requests(self):
        return len(self.context_requests)


@pytest.mark.cpu_only
def test_mm_encoder_sampler_builds_typed_result_from_model_outputs():
    """Sampler converts raw model-output dicts into typed MM results."""
    sampler = EarlyStopWithMMResult()
    model_outputs = {
        "mm_embeddings": [torch.ones(4, 2)],
        "mm_embedding_request_indices": [1],
        "mm_embedding_lengths": [[4]],
        "mrope_position_ids": ["pos"],
    }
    state = sampler.sample_async(
        _FakeScheduledRequests(2),
        model_outputs,
        [],
    )

    assert state.data.mm_embedding_request_indices == [1]
    assert state.data.mm_embedding_lengths == [[4]]
    assert state.data.extra_data == {"mrope_position_ids": ["pos"]}
    assert set(model_outputs) == {
        "mm_embeddings",
        "mm_embedding_request_indices",
        "mm_embedding_lengths",
        "mrope_position_ids",
    }


@pytest.mark.cpu_only
def test_mm_encoder_sampler_rejects_typed_result_batch_mismatch():
    """MM embedding arrays must stay length-aligned with request indices."""
    sampler = EarlyStopWithMMResult()

    with pytest.raises(ValueError, match="batch size"):
        sampler.sample_async(
            _FakeScheduledRequests(2),
            {
                "mm_embeddings": [torch.ones(4, 2)],
                "mm_embedding_request_indices": [1],
                "mm_embedding_lengths": [],
            },
            [],
        )


@pytest.mark.cpu_only
def test_mm_encoder_sampler_rejects_invalid_request_index():
    """MM encoder output cannot target a request outside the scheduled batch."""
    sampler = EarlyStopWithMMResult()

    with pytest.raises(ValueError, match="invalid request index"):
        sampler.sample_async(
            _FakeScheduledRequests(1),
            {
                "mm_embeddings": [torch.ones(4, 2)],
                "mm_embedding_request_indices": [1],
                "mm_embedding_lengths": [[4]],
            },
            [],
        )


@pytest.mark.cpu_only
def test_multimodal_result_rejects_embedding_shape_mismatch():
    """Per-item lengths must sum to the attached embedding rows."""
    with pytest.raises(ValueError, match="shape mismatch"):
        MultimodalResult(
            mm_embeddings=[torch.ones(4, 2)],
            mm_embedding_request_indices=[0],
            mm_embedding_lengths=[[3]],
            num_context_requests=1,
            extra_data={},
        )
