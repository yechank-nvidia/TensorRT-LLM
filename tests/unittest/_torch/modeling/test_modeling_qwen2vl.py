# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed

from tensorrt_llm._torch.models import modeling_qwen2vl
from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5_VisionModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5VLInputProcessorBase,
)


def test_concrete_qwen_vl_models_support_encoder_cache() -> None:
    assert not modeling_qwen2vl.Qwen2VLModelBase.supports_encoder_cache
    assert modeling_qwen2vl.Qwen2VLModel.supports_encoder_cache
    assert modeling_qwen2vl.Qwen2_5_VLModel.supports_encoder_cache


@pytest.mark.parametrize(
    ("modality", "processor_name", "output_name"),
    [
        ("image", "image_processor", "pixel_values"),
        ("video", "video_processor", "pixel_values_videos"),
    ],
)
def test_qwen2_5_vision_processor_artifact_reuse(modality, processor_name, output_name) -> None:
    vision_processor = MagicMock(
        return_value={
            output_name: torch.ones(2, 3),
        }
    )

    class CombinedProcessor:
        def __init__(self):
            self.image_processor = MagicMock()
            self.video_processor = MagicMock()
            setattr(self, processor_name, vision_processor)

        def __call__(self, *, images, videos, **kwargs):
            values = images if images is not None else videos
            return getattr(self, processor_name)(**{f"{modality}s": values})

    processor = object.__new__(Qwen2_5VLInputProcessorBase)
    processor._processor = CombinedProcessor()
    processor._processor_artifacts_in_flight = {}
    processor._processor_artifacts_lock = threading.Lock()
    processor._processor_artifacts_ready = OrderedDict()
    processor._processor_artifacts_ready_bytes = 0
    processor._processor_artifact_cache_max_bytes = 1 << 20
    processor._dtype = torch.bfloat16
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    item = array if modality == "image" else SimpleNamespace(frames=[array])
    mm_data = {modality: [item]}

    artifact_keys = {modality: "same-media"}
    first = processor._preprocess("first", mm_data, {}, artifact_keys)
    second = processor._preprocess("second", mm_data, {}, artifact_keys)

    assert vision_processor.call_count == 1
    torch.testing.assert_close(first[output_name], second[output_name])
    assert first[output_name].dtype == torch.bfloat16
    assert (
        first[output_name].untyped_storage().data_ptr()
        != second[output_name].untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("modalities", [("image",), ("video",), ("image", "video")])
def test_qwen_vision_hashes_select_artifacts(modalities) -> None:
    processor = object.__new__(Qwen2_5VLInputProcessorBase)
    processor.call_with_text_prompt = MagicMock(return_value=([], None))
    inputs = {
        "prompt": "prompt",
        "multi_modal_data": {modality: [object()] for modality in modalities},
    }

    processor._process_with_hashes(
        inputs,
        MagicMock(),
        {modality: [f"{modality}-hash"] for modality in modalities},
        "kwargs-hash",
    )

    artifact_keys = processor.call_with_text_prompt.call_args.kwargs["processor_artifact_keys"]
    assert set(artifact_keys) == set(modalities)
    for modality in modalities:
        assert artifact_keys[modality][1:] == (
            modality,
            "kwargs-hash",
            (f"{modality}-hash",),
        )


def test_qwen2_5_vision_patch_projection_matches_conv3d(monkeypatch) -> None:
    patch_embed = Qwen2_5_VisionPatchEmbed(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=8,
    )
    pixel_values = torch.randn(6, 3 * 2 * 4 * 4)
    expected = patch_embed(pixel_values)

    vision = Qwen2_5_VisionModel.__new__(Qwen2_5_VisionModel)
    torch.nn.Module.__init__(vision)
    vision.patch_embed = patch_embed
    vision.spatial_merge_unit = 1
    vision._rope_position_ids_buffer = None
    vision.full_attn_metadata = object()
    vision.window_attn_metadata = object()
    vision._full_attn_max_seq_len = 6
    vision._window_attn_max_seq_len = 6
    vision.fullatt_block_indexes = []
    vision.blocks = torch.nn.ModuleList()
    vision.merger = torch.nn.Identity()
    vision.get_rotary_pos_emb_window_data = lambda grid: (
        [torch.empty(6, 0)],
        [torch.empty(6, 0)],
        [torch.arange(6)],
        [6],
    )
    vision.prepare_attn_metadata = lambda seq_lens, metadata, **kwargs: metadata
    monkeypatch.setattr(
        modeling_qwen2vl,
        "async_tensor_h2d",
        lambda tensor, *, dtype, device: tensor.to(dtype=dtype, device=device),
    )

    actual = vision(pixel_values, torch.tensor([[1, 2, 3]]))

    torch.testing.assert_close(actual, expected)


def test_qwen2_5_vision_attention_reuses_fused_qkv(monkeypatch) -> None:
    monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", None)

    attention = Qwen2_5_VLVisionAttention.__new__(Qwen2_5_VLVisionAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 4
    attention.q_size = 8
    attention.kv_size = 8
    attention.support_fused_qkv = True
    attention.layer_idx = 0

    fused_qkv = torch.randn(3, 24)
    q, k, _ = fused_qkv.split(8, dim=-1)
    original_q = q.reshape(3, 2, 4).clone()
    original_k = k.reshape(3, 2, 4).clone()
    cos = torch.randn(3, 2)
    sin = torch.randn(3, 2)
    expected_q = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_q.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    expected_k = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_k.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    attention.qkv_proj = lambda hidden_states: fused_qkv
    forwarded = {}

    def forward_impl(**kwargs):
        forwarded.update(kwargs)
        return kwargs["q"]

    attention.forward_impl = forward_impl
    attention.o_proj = lambda output, layer_idx: output

    output = attention.forward(
        torch.empty(3, 8),
        attn_metadata=object(),
        position_embeddings=(cos, sin),
    )

    actual_q, actual_k, _ = fused_qkv.split(8, dim=-1)
    torch.testing.assert_close(actual_q.reshape_as(expected_q), expected_q)
    torch.testing.assert_close(actual_k.reshape_as(expected_k), expected_k)
    assert forwarded["q"] is fused_qkv
    assert forwarded["k"] is None
    assert forwarded["v"] is None
    assert output is fused_qkv


def test_qwen2vl_forward_uses_scheduled_item_segments(monkeypatch) -> None:
    segments = (torch.randn(2, 8), torch.randn(3, 8))
    multimodal_param = SimpleNamespace(multimodal_data={"multimodal_embedding": segments})
    active_embeddings = list(segments)
    find_input_mm_embeds = MagicMock(
        side_effect=AssertionError(
            "scheduled item segments are already limited to the active chunk"
        )
    )
    fuse_input_embeds = MagicMock(return_value=(torch.tensor([1]), torch.empty(5, 8)))
    monkeypatch.setattr(modeling_qwen2vl, "find_input_mm_embeds", find_input_mm_embeds)
    monkeypatch.setattr(modeling_qwen2vl, "fuse_input_embeds", fuse_input_embeds)

    model = SimpleNamespace(
        _get_requests_with_mm_data=lambda params: params,
        _get_or_encode_multimodal_embeddings=MagicMock(return_value=active_embeddings),
        mm_encoder=object(),
        model_config=SimpleNamespace(pretrained_config=SimpleNamespace(disable_fuse_rope=True)),
        llm=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=object()),
            forward=MagicMock(return_value=torch.empty(1, 8)),
        ),
        mm_token_ids=torch.tensor([0]),
    )

    modeling_qwen2vl.Qwen2VLModelBase.forward(
        model,
        SimpleNamespace(num_contexts=1, num_generations=0),
        input_ids=torch.tensor([0]),
        position_ids=torch.tensor([0]),
        multimodal_params=[multimodal_param],
    )

    model._get_or_encode_multimodal_embeddings.assert_called_once_with([multimodal_param])
    find_input_mm_embeds.assert_not_called()
    assert fuse_input_embeds.call_args.args[2] == active_embeddings


def test_qwen2vl_llm_compile_uses_recompile_limit(monkeypatch) -> None:
    eager_model = object()
    compiled_model = object()
    compile_mock = MagicMock(return_value=compiled_model)
    monkeypatch.setattr(torch, "compile", compile_mock)
    model = SimpleNamespace(llm=SimpleNamespace(model=eager_model))

    modeling_qwen2vl.Qwen2VLModelBase.apply_llm_torch_compile(
        model, backend="backend", fullgraph=True, recompile_limit=16
    )

    compile_mock.assert_called_once_with(
        eager_model, backend="backend", fullgraph=True, recompile_limit=16
    )
    assert model.llm.model is compiled_model
