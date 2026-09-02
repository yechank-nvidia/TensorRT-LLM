from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

from tqdm import tqdm

from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.inputs import create_input_processor, prompt_inputs
from tensorrt_llm.inputs.data import PromptInputs
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MULTIMODAL_ENCODER_ITEM_MODE_KEY,
    MultimodalInput, MultimodalParams)
from tensorrt_llm.inputs.registry import (MultimodalEncoderItemMetadata,
                                          get_multimodal_encoder_item_metadata)
from tensorrt_llm.sampling_params import SamplingParams

from .llm import BaseLLM, PreprocessedInputs, RequestOutput, _TorchLLM
from .llm_args import TorchLlmArgs
from .mpi_session import external_mpi_comm_available


def _select_multimodal_encoder_items(
    inputs: PreprocessedInputs,
    item_indices: Sequence[int],
) -> PreprocessedInputs:
    """Build one encoder-only request for the selected logical items.

    Raw CPU payloads remain shared with the original preprocessed request. The
    prompt-order metadata is narrowed here so every lower layer sees an
    ordinary, self-consistent request and only needs a boolean item-mode gate.
    """
    params = inputs.multimodal_params
    if params is None or params.multimodal_input is None:
        raise ValueError(
            "Selected MM encoding requires multimodal input metadata")
    data = params.multimodal_data
    item_metadata = get_multimodal_encoder_item_metadata(data)
    if item_metadata is None:
        raise ValueError("Selected MM encoding requires encoder item metadata")

    indices = list(item_indices)
    if not indices:
        raise ValueError("item_indices must not be empty")
    if not all(isinstance(item_idx, int) for item_idx in indices):
        raise TypeError("item_indices must contain only integers")
    if len(indices) != len(set(indices)):
        raise ValueError("item_indices must not contain duplicates")
    if any(item_idx < 0 or item_idx >= len(item_metadata.item_refs)
           for item_idx in indices):
        raise ValueError("item_indices contains an out-of-range item")

    mm_input = params.multimodal_input
    run_offsets = mm_input.multimodal_item_run_cu_offsets
    if run_offsets is None:
        selected_run_offsets = None
        selected_run_positions = None
        selected_run_lengths = None
    else:
        assert mm_input.multimodal_run_positions is not None
        assert mm_input.multimodal_run_lengths is not None
        selected_run_offsets = [0]
        selected_run_positions = []
        selected_run_lengths = []
        for item_idx in indices:
            run_begin = run_offsets[item_idx]
            run_end = run_offsets[item_idx + 1]
            selected_run_positions.extend(
                mm_input.multimodal_run_positions[run_begin:run_end])
            selected_run_lengths.extend(
                mm_input.multimodal_run_lengths[run_begin:run_end])
            selected_run_offsets.append(len(selected_run_positions))

    selected_input = MultimodalInput.from_components(
        [mm_input.multimodal_hashes[item_idx] for item_idx in indices],
        [mm_input.multimodal_positions[item_idx] for item_idx in indices],
        [mm_input.multimodal_lengths[item_idx] for item_idx in indices],
        (None if mm_input.multimodal_uuids is None else
         [mm_input.multimodal_uuids[item_idx] for item_idx in indices]),
        selected_run_offsets,
        selected_run_positions,
        selected_run_lengths,
    )
    selected_metadata = MultimodalEncoderItemMetadata(
        item_refs=[item_metadata.item_refs[item_idx] for item_idx in indices],
        encoder_token_lengths=[
            item_metadata.encoder_token_lengths[item_idx]
            for item_idx in indices
        ],
        output_embedding_lengths=[
            item_metadata.output_embedding_lengths[item_idx]
            for item_idx in indices
        ],
    )
    selected_data = dict(data or {})
    selected_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = selected_metadata
    selected_data["multimodal_embedding_lengths"] = list(
        selected_metadata.output_embedding_lengths)
    selected_data[MULTIMODAL_ENCODER_ITEM_MODE_KEY] = True
    if "encoder_token_lengths" in selected_data:
        selected_data["encoder_token_lengths"] = [
            selected_data["encoder_token_lengths"][item_idx]
            for item_idx in indices
        ]

    selected_order = (None if params.mm_item_order is None else
                      [params.mm_item_order[item_idx] for item_idx in indices])
    return PreprocessedInputs(
        prompt_token_ids=list(inputs.prompt_token_ids),
        multimodal_params=MultimodalParams(
            multimodal_input=selected_input,
            multimodal_data=selected_data,
            mm_item_order=selected_order,
        ),
        encoder_input_token_ids=(None if inputs.encoder_input_token_ids is None
                                 else list(inputs.encoder_input_token_ids)),
    )


class MultimodalEncoder(_TorchLLM):
    """MultimodalEncoder class is the main class for running a multimodal encoder model using PyTorch backend."""

    def __init__(self,
                 model: Union[str, Path],
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: Literal["auto", "float16", "float32",
                                "bfloat16"] = "auto",
                 **kwargs: Any) -> None:

        # Set higher default max_num_tokens for multimodal encoder (16384 vs 8192 default)
        # Vision encoders can handle more tokens than text-only models
        # TODO: Make this adaptive based on model-specific max_mm_token_length (see _test_llm_multimodal_general)
        if 'max_num_tokens' not in kwargs or kwargs['max_num_tokens'] is None:
            kwargs['max_num_tokens'] = 16384

        super().__init__(model,
                         trust_remote_code=trust_remote_code,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype=dtype,
                         **kwargs)

    def _build_model(self):
        BaseLLM._build_model(self)
        assert self._engine_dir is None

        # Tokenizer loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        checkpoint_format = getattr(self.args, "checkpoint_format", None)
        trust_remote_code = self.args.trust_remote_code
        self.input_processor = create_input_processor(
            self._hf_model_dir,
            self.tokenizer,
            checkpoint_format,
            trust_remote_code=trust_remote_code)
        self._tokenizer = self.input_processor.tokenizer

        assert isinstance(self.args, TorchLlmArgs)
        self.args.mm_encoder_only = True

        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=None,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size),
            is_llm_executor=True,  # TODO: check if this is correct or needed
            hf_model_dir=self._hf_model_dir,
            tokenizer=self.tokenizer,
            llm_args=self.args)

    def _validate_mm_args_for_torch_backend(self, kwargs: dict) -> None:
        """Validate that users don't pass LLM-specific arguments when using MultimodalEncoder (PyTorch)."""
        if kwargs.get("encode_only") is True:
            raise ValueError(
                "MultimodalEncoder does not support encode_only=True. "
                "It uses mm_encoder_only execution internally.")

    def _validate_args_for_torch_backend(self, kwargs: dict) -> None:
        """Run multimodal validation inside the tracked constructor boundary."""
        self._validate_mm_args_for_torch_backend(kwargs)
        super()._validate_args_for_torch_backend(kwargs)

    def generate(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        use_tqdm: bool = True,
    ) -> Union[RequestOutput, List[RequestOutput]]:
        """Generate output for the given prompts in the synchronous mode.
        Synchronous generation accepts either single prompt or batched prompts.

        Args:
            inputs (tensorrt_llm.inputs.data.PromptInputs, Sequence[tensorrt_llm.inputs.data.PromptInputs]): The prompt text or token ids.
                It can be single prompt or batched prompts.
        Returns:
            Union[tensorrt_llm.llmapi.RequestOutput, List[tensorrt_llm.llmapi.RequestOutput]]: The output data of the completion request to the LLM.
        """
        unbatched = not isinstance(inputs, list)
        if not unbatched:
            if isinstance(inputs[0], int):
                unbatched = True

        if unbatched:
            inputs = [inputs]

        inputs = [prompt_inputs(i) for i in inputs]

        futures = []
        for request_inputs in inputs:
            future = self.generate_async(request_inputs)
            futures.append(future)

        for future in tqdm(futures,
                           desc="Processed requests",
                           dynamic_ncols=True,
                           disable=not use_tqdm):
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

    @nvtx_range_debug("MM_encoder.generate_async",
                      color="green",
                      category="VisionEncoder")
    def generate_async(
        self,
        inputs: PromptInputs,
        sampling_params: Optional[SamplingParams] = None,
    ) -> RequestOutput:
        """Generate output for the given multimodal request in the asynchronous mode.
        Asynchronous generation accepts single multimodal request only.

        Returns:
            Future that resolves to tensorrt_llm.llmapi.RequestOutput containing mm_embeddings
        """
        result = super().generate_async(inputs, sampling_params)
        # TODO: possible postprocess the result for disaggregated serving
        return result
