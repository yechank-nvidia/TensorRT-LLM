# Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is based on official VILA: https://github.com/NVlabs/VILA/blob/main/llava/model/multimodal_encoder/

import copy
import itertools
import os
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Type

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams)
from tensorrt_llm.inputs.registry import (MultimodalEncoderItemMetadata,
                                          get_multimodal_encoder_item_metadata)

from ..attention_backend.interface import AttentionMetadata
from .modeling_multimodal_utils import multiscale_forward

# Legacy fallback for encoders that do not yet opt into atomic-item scheduling
# or provide a model-specific item/token-to-segment mapping. Their encoder
# forward is not runtime-budgeted, so configured item/token limits cannot
# safely size fixed ``AttentionMetadata`` per-segment buffers.
#
# TODO: Replace this fallback as the remaining encoders provide model-specific
# conversions from atomic item/token budgets to attention segments. The exact
# conversion requires each encoder's minimum tokens per tile/window/segment, so
# retain this worst-case capacity for models without that contract.
# Runtime-scheduled Qwen and Mistral/Pixtral encoders override
# ``get_encoder_attention_metadata_capacity`` and do not consume this fallback.
_ENCODER_FALLBACK_MAX_NUM_REQUESTS = 8192
_MM_DATA_INPUT_MODALITY_KEYS = frozenset({"audio", "image", "video"})


class MultimodalEncoderContractError(ValueError):
    """A request-local MM encoder input or output contract violation."""


class MultimodalEncoderMixin:
    """Encoder-side counterpart to ``MultimodalModelMixin``.

    Marker + default ``setup_attn_metadata`` for multimodal encoders whose
    ``AttentionMetadata`` is built by ``PyTorchModelEngine`` after model load
    using the runtime encoder token budget (``max_num_tokens``).

    Subclasses set ``metadata_cls`` in their own ``__init__`` (typically from
    ``get_attention_backend(model_config.attn_backend).Metadata``) and either
    use the default ``setup_attn_metadata`` below or override it for custom
    Metadata kwargs (e.g. FlashInfer ``kv_layout``, multi-metadata encoders).
    """
    metadata_cls: Type[AttentionMetadata]
    attn_metadata: Optional[AttentionMetadata] = None

    def encode_multimodal_inputs(
        self,
        multimodal_params: Sequence[MultimodalParams],
    ) -> torch.Tensor:
        """Run model-specific encoder work and return request-ordered rows."""
        raise NotImplementedError

    def prepare_multimodal_encoder_inputs(
        self,
        selected_items: Sequence[tuple[MultimodalParams, int]],
    ) -> list[tuple[MultimodalParams, list[int], str]]:
        """Slice scheduler-selected items before the caller performs H2D.

        Adjacent items from the same request and modality are sliced together,
        while the returned tuples retain the original selection order.
        """
        encoder_inputs: list[tuple[MultimodalParams, list[int], str]] = []
        for (
                multimodal_param,
                run_indices,
                modality,
                item_metadata,
        ) in self._runs_by_request_modality(selected_items):
            item_refs = item_metadata.item_refs
            try:
                residual = self.build_multimodal_encoder_input(
                    multimodal_param,
                    [item_refs[i][1] for i in run_indices],
                    modality=modality,
                )
                self._apply_metadata_slice(residual, multimodal_param,
                                           run_indices)
            except MultimodalEncoderContractError:
                raise
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise MultimodalEncoderContractError(
                    f"Invalid multimodal encoder item input: {error}"
                ) from error
            encoder_inputs.append((
                residual,
                [
                    int(item_metadata.output_embedding_lengths[i])
                    for i in run_indices
                ],
                modality,
            ))
        return encoder_inputs

    @staticmethod
    def _runs_by_request_modality(
        selected_items: Sequence[tuple[MultimodalParams, int]],
    ) -> Iterator[tuple[MultimodalParams, list[int], str,
                        MultimodalEncoderItemMetadata]]:
        """Split selection order into maximal request-and-modality runs."""
        run_param: Optional[MultimodalParams] = None
        run_modality: Optional[str] = None
        run_metadata: Optional[MultimodalEncoderItemMetadata] = None
        run_indices: list[int] = []
        for multimodal_param, item_idx in selected_items:
            try:
                metadata = (run_metadata if multimodal_param is run_param else
                            get_multimodal_encoder_item_metadata(
                                multimodal_param.multimodal_data or {}))
            except (TypeError, ValueError) as error:
                raise MultimodalEncoderContractError(str(error)) from error
            if metadata is None:
                raise MultimodalEncoderContractError(
                    "MM item metadata is required for item encoding")
            if item_idx < 0 or item_idx >= len(metadata.item_refs):
                raise MultimodalEncoderContractError(
                    f"MM item index {item_idx} is out of range for "
                    f"{len(metadata.item_refs)} item(s)")
            modality = metadata.item_refs[item_idx][0]
            if run_indices and (multimodal_param is not run_param
                                or modality != run_modality):
                assert run_param is not None
                assert run_modality is not None
                assert run_metadata is not None
                yield run_param, run_indices, run_modality, run_metadata
                run_indices = []
            run_param = multimodal_param
            run_modality = modality
            run_metadata = metadata
            run_indices.append(item_idx)
        if run_indices:
            assert run_param is not None
            assert run_modality is not None
            assert run_metadata is not None
            yield run_param, run_indices, run_modality, run_metadata

    def forward_multimodal_encoder_items(
        self,
        encoder_inputs: Sequence[tuple[MultimodalParams, list[int], str]],
    ) -> list[torch.Tensor]:
        """Encode prepared inputs and return one tensor per selected item."""
        outputs: list[torch.Tensor] = []
        group_params: list[MultimodalParams] = []
        group_lengths: list[int] = []
        group_modality: Optional[str] = None

        def flush_group() -> None:
            if not group_params:
                return
            embeddings = self._run_multimodal_encoder(group_params)
            expected_length = sum(group_lengths)
            if embeddings.shape[0] != expected_length:
                raise MultimodalEncoderContractError(
                    f"MM encoder output length {embeddings.shape[0]} does not "
                    f"match the {expected_length} rows declared by the "
                    "selected items")
            outputs.extend(torch.split(embeddings, group_lengths, dim=0))
            group_params.clear()
            group_lengths.clear()

        for multimodal_param, embedding_lengths, modality in encoder_inputs:
            if group_modality is not None and modality != group_modality:
                flush_group()
            group_modality = modality
            group_params.append(multimodal_param)
            group_lengths.extend(embedding_lengths)
        flush_group()
        return outputs

    def _run_multimodal_encoder(
        self,
        multimodal_params: Sequence[MultimodalParams],
        **encoder_kwargs: Any,
    ) -> torch.Tensor:
        """Run the local encoder; integrated models may override for DP."""
        return self.encode_multimodal_inputs(multimodal_params,
                                             **encoder_kwargs)

    def build_multimodal_encoder_input(
        self,
        param: MultimodalParams,
        item_indices: Sequence[int],
        modality: Optional[str] = None,
    ) -> MultimodalParams:
        """Return raw and parallel metadata for selected modality items."""
        if modality is None:
            modality = self._encoder_cache_modality(param)
        if modality is None:
            raise NotImplementedError(
                "Default `build_multimodal_encoder_input` cannot infer the "
                "modality of a mixed-modality param. Pass `modality` with "
                "modality-local indices, or override for other layouts.")
        modality_data = param.multimodal_data[modality]
        if not isinstance(modality_data, dict):
            raise TypeError(
                f"multimodal_data[{modality!r}] must be a dict, got "
                f"{type(modality_data).__name__}")

        indices = list(item_indices)
        if not indices:
            raise ValueError("item_indices must not be empty")
        contiguous = indices == list(
            range(indices[0], indices[0] + len(indices)))
        grid_key = {
            "image": "image_grid_thw",
            "video": "video_grid_thw"
        }.get(modality)
        pixel_key = {
            "image": "pixel_values",
            "video": "pixel_values_videos"
        }.get(modality)

        if (grid_key and pixel_key and grid_key in modality_data
                and pixel_key in modality_data):
            grids = modality_data[grid_key]
            n_items = grids.shape[0]
            item_selector: Sequence[int] | slice = indices
            if contiguous and indices[0] >= 0 and indices[-1] < n_items:
                item_selector = slice(indices[0], indices[-1] + 1)
            patch_counts = [
                int(count) for count in torch.prod(grids, dim=1).tolist()
            ]
            row_starts = list(itertools.accumulate(patch_counts, initial=0))
            if isinstance(item_selector, slice):
                pixel_slice = modality_data[pixel_key][
                    row_starts[indices[0]]:row_starts[indices[-1] + 1]]
            else:
                per_item = torch.split(modality_data[pixel_key],
                                       patch_counts,
                                       dim=0)
                pixel_slice = torch.cat([per_item[i] for i in indices], dim=0)
            sliced = {
                pixel_key: pixel_slice,
                grid_key: grids[item_selector],
            }
        elif (
                modality in ("image", "video")
                and isinstance(modality_data.get("pixel_values"), torch.Tensor)
                and modality_data["pixel_values"].ndim >= 2 and
            (not isinstance(
                param.multimodal_data.get("multimodal_embedding_lengths"), list)
             or modality_data["pixel_values"].shape[0] == len(
                 param.multimodal_data["multimodal_embedding_lengths"]))):
            n_items = modality_data["pixel_values"].shape[0]
            item_selector = indices
            if contiguous and indices[0] >= 0 and indices[-1] < n_items:
                item_selector = slice(indices[0], indices[-1] + 1)
            miss_pixel = modality_data["pixel_values"][item_selector]
            image_sizes = modality_data.get("image_sizes")
            if image_sizes is not None:
                miss_sizes = [image_sizes[i] for i in indices]
                if miss_sizes and miss_pixel.dim() >= 4:
                    max_h = max(int(size[0]) for size in miss_sizes)
                    max_w = max(int(size[1]) for size in miss_sizes)
                    miss_pixel = miss_pixel[..., :max_h, :max_w]
                sliced = {
                    "pixel_values": miss_pixel,
                    "image_sizes": miss_sizes,
                }
            else:
                sliced = {"pixel_values": miss_pixel}
        elif modality == "audio" and ("input_features" in modality_data
                                      or "audio_features" in modality_data):
            feature_key = ("input_features" if "input_features" in modality_data
                           else "audio_features")
            n_items = modality_data[feature_key].shape[0]
            item_selector = indices
            if contiguous and indices[0] >= 0 and indices[-1] < n_items:
                item_selector = slice(indices[0], indices[-1] + 1)
            sliced = {feature_key: modality_data[feature_key][item_selector]}
        else:
            raise NotImplementedError(
                "Default `build_multimodal_encoder_input` cannot slice "
                f"{modality} layout with fields {sorted(modality_data)}; "
                "override this method.")

        sliced = {
            **modality_data,
            **sliced,
            **self._slice_per_item_sibling_fields(modality_data, n_items, indices,
                                                  sliced.keys()),
        }
        residual_input = (copy.copy(param.multimodal_input)
                          if param.multimodal_input is not None else None)
        residual_data = {
            key: value
            for key, value in param.multimodal_data.items()
            if key not in _MM_DATA_INPUT_MODALITY_KEYS or key == modality
        }
        residual_data[modality] = sliced
        return MultimodalParams(multimodal_data=residual_data,
                                multimodal_input=residual_input)

    @staticmethod
    def _slice_per_item_sibling_fields(
        modality_data: Dict[str, Any],
        n_items: int,
        item_indices: Sequence[int],
        already_sliced: Iterable[str],
    ) -> Dict[str, Any]:
        """Slice modality fields whose leading axis matches item count."""
        skip = set(already_sliced)
        sliced: Dict[str, Any] = {}
        for key, value in modality_data.items():
            if key in skip:
                continue
            if (isinstance(value, torch.Tensor) and value.dim() > 0
                    and value.shape[0] == n_items):
                sliced[key] = value[item_indices]
            elif isinstance(value, list) and len(value) == n_items:
                sliced[key] = [value[i] for i in item_indices]
        return sliced

    @staticmethod
    def _encoder_cache_modality(param: MultimodalParams) -> Optional[str]:
        """Return the sole raw modality, or None for mixed/no raw input."""
        mm_data = param.multimodal_data
        if mm_data is None:
            return None
        modalities = [
            key for key in _MM_DATA_INPUT_MODALITY_KEYS if key in mm_data
        ]
        return modalities[0] if len(modalities) == 1 else None

    @staticmethod
    def _apply_metadata_slice(
        residual: MultimodalParams,
        source: MultimodalParams,
        item_indices: Sequence[int],
    ) -> None:
        """Slice prompt-ordered item metadata with the raw encoder input."""
        source_lengths = source.multimodal_data["multimodal_embedding_lengths"]
        residual.multimodal_data["multimodal_embedding_lengths"] = [
            source_lengths[i] for i in item_indices
        ]
        source_metadata = get_multimodal_encoder_item_metadata(
            source.multimodal_data)
        if source_metadata is not None:
            residual.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = (
                MultimodalEncoderItemMetadata(
                    item_refs=[
                        source_metadata.item_refs[i] for i in item_indices
                    ],
                    encoder_token_lengths=[
                        source_metadata.encoder_token_lengths[i]
                        for i in item_indices
                    ],
                    output_embedding_lengths=[
                        source_metadata.output_embedding_lengths[i]
                        for i in item_indices
                    ],
                ))
        if (residual.multimodal_input is not None
                and source.multimodal_input is not None):
            source_hashes = source.multimodal_input.multimodal_hashes
            residual.multimodal_input.multimodal_hashes = [
                source_hashes[i] for i in item_indices
            ]

    def get_encoder_attention_metadata_capacity(
            self, max_num_tokens: int) -> dict[str, int]:
        """Map the token budget to model-internal attention sequences.

        Keys name this encoder's attention metadata objects, so they are
        model-specific: this default declares a single `attention` object,
        while a windowed encoder such as Qwen2.5-VL declares `full_attention`
        and `window_attention`. There is no fixed superset to enumerate.

        The default conservatively allows one attention sequence per token.
        Encoders with tighter geometry-aware bounds should override this.
        """
        return {
            "attention": max(max_num_tokens, _ENCODER_FALLBACK_MAX_NUM_REQUESTS)
        }

    def setup_attn_metadata(
        self,
        max_num_tokens: int,
        attention_metadata_capacity: Optional[dict[str, int]] = None,
    ) -> None:
        """Map encoder item/token budgets to attention metadata capacity."""
        capacities = (
            attention_metadata_capacity
            if attention_metadata_capacity is not None else
            self.get_encoder_attention_metadata_capacity(max_num_tokens))
        self.attn_metadata = self.metadata_cls(
            max_num_requests=capacities["attention"],
            max_num_tokens=max_num_tokens,
            kv_cache_manager=None,
        )

    def set_attn_max_seq_len(self, max_seq_len: int) -> None:
        """Set an optional stable per-segment attention capacity.

        Specialized encoders may override this hook. Keeping it separate from
        `setup_attn_metadata` preserves that method's existing override
        contract for external encoders.
        """


class VisionTower(nn.Module):

    def __init__(self, model_name_or_path, config):
        super().__init__()

        assert os.path.exists(
            model_name_or_path
        ), f"Pretrained vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True)
        self.name = vision_tower_cfg.architectures[0].lower()

        if "clip" in self.name:
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path, dtype=config.model_dtype)
        elif "siglip" in self.name:
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                dtype="auto")
        else:
            raise ValueError(f"Unsupported vision tower: {self.name}")

        self.select_layer = getattr(config, "mm_vision_select_layer", -2)
        self.select_feature = getattr(config, "mm_vision_select_feature",
                                      "patch")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(
                f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(
                    image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(
                images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2


class VisionTowerS2(VisionTower):

    def __init__(self, model_name_or_path, config):
        super().__init__(model_name_or_path, config)

        self.scales = list(map(int, config.s2_scales.split(",")))
        self.scales.sort()
        self.max_split_size = config.s2_max_split_size
        self.resize_output_to_scale_idx = getattr(
            config, "s2_resize_output_to_scale_idx", 0)

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device,
                                                         dtype=self.dtype),
                                               output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(
            images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.scales,
                    max_split_size=self.max_split_size,
                    resize_output_to_idx=self.resize_output_to_scale_idx,
                )
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.scales,
                max_split_size=self.max_split_size,
                resize_output_to_idx=self.resize_output_to_scale_idx,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.scales)


class VisionTowerDynamicS2(VisionTowerS2):

    def __init__(self, model_name_or_path, config):
        super().__init__(model_name_or_path, config)

    def forward(self, images):
        assert type(images) is not list

        image_features = self.forward_feature(images)

        return image_features
