import copy
import math
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)

from tensorrt_llm._utils import nvtx_range

from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import register_auto_model


def select_best_resolution(original_size: tuple,
                           possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height,
                                   original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def unpad_image(tensor: torch.Tensor,
                original_size: Tuple[int, int]) -> torch.Tensor:
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


def get_anyres_image_grid_shape(
    image_size: Tuple[int, int],
    grid_pinpoints: Union[str, List[Tuple[int, int]]],
    patch_size: int,
) -> Tuple[int, int]:
    possible_resolutions = grid_pinpoints if isinstance(
        grid_pinpoints, list) else ast.literal_eval(grid_pinpoints)

    original_width, original_height = image_size
    height, width = select_best_resolution((original_height, original_width),
                                           possible_resolutions)
    return width // patch_size, height // patch_size


def reshape_and_unpad_image_features(
    image_feature: torch.Tensor,
    height: int,
    width: int,
    image_size: Tuple[int, int],
    possible_resolutions: List[Tuple[int, int]],
    grid_size: int,
    unpad: bool,
    image_newline: torch.Tensor,
) -> torch.Tensor:
    base_image_feature = image_feature[0]
    image_feature = image_feature[1:]

    assert (
        height * width == base_image_feature.shape[0]
    ), f"height: {height}, width: {width}, base_image_feature.shape[0]: {base_image_feature.shape[0]}"

    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_size, possible_resolutions, grid_size)
    image_feature = image_feature.view(num_patch_height, num_patch_width,
                                       height, width, -1)

    if unpad:
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_size)
        image_feature = torch.cat(
            (
                image_feature,
                image_newline[:, None, None].expand(*image_feature.shape[:-1],
                                                    1).to(image_feature.device),
            ),
            dim=-1,
        )
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    else:
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.flatten(0, 3)
    image_feature = torch.cat((base_image_feature, image_feature), dim=0)

    return image_feature


def anyres_postprocessing(
    image_forward_outs: torch.FloatTensor,
    split_sizes: List[int],
    image_sizes: List[List[int]],
    possible_resolutions: List[Tuple[int, int]],
    is_videos: List[bool],
    patch_size: int,
    grid_size: int,
    image_newline: torch.FloatTensor,
    num_queries_vis_abstractor: int = -1,
    unpad: bool = False,
) -> List[torch.FloatTensor]:
    height = width = grid_size // patch_size

    if num_queries_vis_abstractor > 0:
        assert (num_queries_vis_abstractor**0.5
                ).is_integer(), "n_queries must be square number"
        height = width = int(num_queries_vis_abstractor**0.5)

    image_features = torch.split(image_forward_outs, split_sizes, dim=0)

    # post-processing (unpad, add newline)
    new_image_features = []
    for image_idx, (image_feature,
                    is_video) in enumerate(zip(image_features, is_videos)):
        if image_feature.shape[0] > 1:
            if not is_video:
                image_feature = reshape_and_unpad_image_features(
                    image_feature=image_feature,
                    height=height,
                    width=width,
                    image_size=image_sizes[image_idx],
                    possible_resolutions=possible_resolutions,
                    grid_size=grid_size,  # Pass grid info if needed by helper
                    unpad=unpad,
                    image_newline=image_newline,
                )
            else:
                image_feature = image_feature.flatten(0, 1)
        else:
            image_feature = image_feature[0]
            if unpad and not is_video:
                image_feature = torch.cat(
                    (image_feature, image_newline[None].to(
                        image_feature.device)),
                    dim=0)
        new_image_features.append(image_feature)
    image_features = new_image_features
    return image_features


def adaptive_anyres_postprocessing(
    image_forward_outs: torch.FloatTensor,
    image_sizes: List[List[int]],
    possible_resolutions: List[Tuple[int, int]],
    is_videos: List[bool],
    group_ids: List[List[int]],
    num_queries_vis_abstractors: List[List[int]],
    grid_size: int,
    image_newline: torch.FloatTensor,
    unpad: bool = False,
) -> List[torch.FloatTensor]:
    # post-processing (unpad, add newline)
    new_image_features = []
    for image_idx, (image_feature,
                    is_video) in enumerate(zip(image_forward_outs, is_videos)):
        num_queries_vis_abstractor = num_queries_vis_abstractors[image_idx]
        assert (num_queries_vis_abstractor**0.5
                ).is_integer(), "n_queries must be square number"
        height = width = int(num_queries_vis_abstractor**0.5)

        if image_feature.shape[0] > 1:
            if not is_video:
                image_feature = reshape_and_unpad_image_features(
                    image_feature=image_feature,
                    height=height,
                    width=width,
                    image_size=image_sizes[image_idx],
                    possible_resolutions=possible_resolutions,
                    grid_size=grid_size,
                    unpad=unpad,
                    image_newline=image_newline,
                )
            else:
                image_feature = image_feature.flatten(0, 1)
        else:
            image_feature = image_feature[0]
            if unpad and not is_video:
                image_feature = torch.cat(
                    (image_feature, image_newline[None].to(
                        image_feature.device)),
                    dim=0)
        new_image_features.append(image_feature)

    image_features = [
        torch.cat([new_image_features[group_id] for group_id in group_ids_list],
                  dim=0) for group_ids_list in group_ids
    ]
    return image_features


def compute_adaptive_params(
    pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
    num_queries_vis_abstractors: Optional[List[List[int]]] = None,
    num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
    image_sizes: Optional[List[List[List[int]]]] = None,
    is_videos: Optional[List[bool]] = None,
    first_last_frames_slows: Optional[List[bool]] = None,
) -> Tuple[List[int], List[int], List[List[int]], List[bool], List[List[int]]]:
    # Check if all elements are integers greater than or equal to 0
    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors
    ), "All values in num_queries_vis_abstractors must be integers >= 0."

    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors_slow
    ), "All values in num_queries_vis_abstractors_slow must be integers >= 0."

    assert is_videos is not None

    # Is it the first or last image? (for applying slowfast to video processing)
    is_first_images = []
    is_last_images = []
    for is_video in is_videos:
        for idx, is_video_item in enumerate(is_video):
            if idx == 0:
                is_first_images.append(True)
            else:
                is_first_images.append(False)
            if idx == len(is_video) - 1:
                is_last_images.append(True)
            else:
                is_last_images.append(False)

    num_queries_vis_abstractors = list(chain(*num_queries_vis_abstractors))
    num_queries_vis_abstractors_slow = list(
        chain(*num_queries_vis_abstractors_slow))
    image_sizes = list(chain(*image_sizes))
    is_videos = list(chain(*is_videos))
    first_last_frames_slows = list(chain(*first_last_frames_slows))

    # Use slowfast mode if there's at least one visual token count greater than 0 in num_queries_vis_abstractors_slow
    use_slowfast = any(
        [num_query > 0 for num_query in num_queries_vis_abstractors_slow])
    num_grids = [pixel_value.shape[0] for pixel_value in chain(*pixel_values)]
    num_grids = [0] + num_grids
    group_ids = []

    if use_slowfast:
        new_num_grids = [num_grids[0]]
        new_num_queries = []
        new_image_sizes = []
        new_is_videos = []

        # When using slowfast, split more finely
        # 0th local grid is slow frame, remaining local grids are fast frames
        for (
                num_query,
                num_query_slow,
                num_grid,
                image_size,
                is_video,
                first_last_frames_slow,
                is_first_image,
                is_last_image,
        ) in zip(
                num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow,
                num_grids[1:],
                image_sizes,
                is_videos,
                first_last_frames_slows,
                is_first_images,
                is_last_images,
        ):

            if not first_last_frames_slow and num_query_slow > 0:  # Process all image in slowfast mode
                assert is_video  # slowfast mode is only applied to videos

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                # slow frame (first grid)
                new_num_grids.append(new_num_grids[-1] + 1)
                new_num_queries.append(num_query_slow)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                if num_grid >= 2:
                    # fast frames
                    new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                    new_num_queries.append(num_query)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)
                    this_group_ids.append(this_group_ids[-1] + 1)

                group_ids.append(this_group_ids)
            elif (first_last_frames_slow and num_query_slow > 0
                  and (is_first_image or is_last_image)
                  ):  # Process only first/last image in slowfast mode
                # Case for special treatment of first/last frames in slow mode
                assert is_video  # slowfast mode is only applied to videos

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                if num_grid == 1:
                    # Simply process with slow since there's only one grid
                    new_num_grids.append(new_num_grids[-1] + 1)
                    new_num_queries.append(num_query_slow)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)

                if num_grid >= 2:
                    # Special treatment for first or last grid depending on is_first_image or is_last_image

                    if is_first_image:  # includes both first and last
                        # slow frame (first grid)
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        # fast frames
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    elif is_last_image:
                        # fast frames
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        # slow frame (last grid)
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    else:
                        raise Exception("This case should not be reached.")
                group_ids.append(this_group_ids)
            else:
                # Not in slowfast mode, so reduce all by num_query (fast)
                new_num_grids.append(new_num_grids[-1] + num_grid)
                new_num_queries.append(num_query)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                start_group_id = group_ids[-1][-1] + 1 if group_ids else 0
                group_ids.append([start_group_id])

        num_grids = new_num_grids
        num_queries_vis_abstractors = new_num_queries
        image_sizes = new_image_sizes
        is_videos = new_is_videos
    else:
        num_grids = [sum(num_grids[:i]) for i in range(1, len(num_grids) + 1)]
        group_ids = [[group_id] for group_id in range(len(is_videos))]

    return num_queries_vis_abstractors, num_grids, image_sizes, is_videos, group_ids


# NOTE: This is done in the input processor.
def determine_non_vision_query_lengths(input_ids: torch.LongTensor, pad_id: int,
                                       img_start_id: int) -> List[int]:
    non_vision_query_lengths = []
    batch_size, len_seq = input_ids.size(0), input_ids.size(1)

    for i in range(batch_size):
        temp_idx = (input_ids[i] == pad_id).nonzero()
        eos_idx = temp_idx[0, 0].item() if len(temp_idx) > 0 else len_seq
        num_imgs = (input_ids[i] == img_start_id).sum().item()
        non_vision_query_lengths.append(eos_idx - num_imgs)

    if all([pad_id in input_id for input_id in input_ids.tolist()]):
        non_vision_query_lengths = [
            non_vision_query_length + 1
            for non_vision_query_length in non_vision_query_lengths
        ]

    return non_vision_query_lengths


class HCXVisionInputProcessor(InputProcessor):

    def __init__(self, model_path: str, model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer):

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(model_path,
                                                       trust_remote_code=True,
                                                       use_fast=False)

    @nvtx_range("Inputprocessor _post_process()")
    def _post_process(self, input_ids: torch.Tensor,
                      preprocessed_image: dict[str, any]):
        vision_query_lengths = preprocessed_image.get("vision_query_lengths",
                                                      None)
        non_vision_query_lengths = determine_non_vision_query_lengths(
            input_ids, self.tokenizer.pad_token_id,
            self.model_config.img_start_id)
        batch_size = input_ids.size(0)
        # Slicing is faster than concatenation
        len_inputs_embeds = max([
            sum(vision_query_length) + non_vision_query_length
            for non_vision_query_length, vision_query_length in zip(
                non_vision_query_lengths, vision_query_lengths)
        ])

        len_inputs_embeds = min(self.model_config.decoder_max_length,
                                len_inputs_embeds)

        image_cnts = (input_ids == self.model_config.img_start_id).sum(
            dim=1).tolist()

        fused_input_ids = torch.zeros([batch_size, len_inputs_embeds],
                                      dtype=input_ids.dtype)
        for batch_idx, sample in enumerate(input_ids):
            non_vision_query_length = non_vision_query_lengths[batch_idx]
            # Safely concatenate with visual tokens and then slice
            sample = sample[:non_vision_query_length + image_cnts[batch_idx]]

            mask = (sample == self.model_config.img_start_id)
            img_start_ids = mask.nonzero()
            input_start, temp_start = 0, 0

            for multi_img_idx, img_start_idx in enumerate(img_start_ids):
                # Calculate token length up to the current image starting point
                token_len = img_start_idx - temp_start

                # Copy tokens to inputs_embeds
                fused_input_ids[batch_idx, input_start:input_start +
                                token_len] = input_ids[batch_idx,
                                                       temp_start:temp_start +
                                                       token_len]

                fused_input_ids[
                    batch_idx,
                    input_start + token_len:input_start + token_len +
                    vision_query_lengths[batch_idx][multi_img_idx],
                ] = self.model_config.language_config["vocab_size"] + 1

                # Update starting points for next token processing
                input_start += token_len + vision_query_lengths[batch_idx][
                    multi_img_idx]
                temp_start += token_len + 1  # Increase by 1 to skip the image start token

            # Process tokens after the last image end token
            token_len = min(sample[temp_start:].size(0),
                            fused_input_ids.size(1) - input_start)
            fused_input_ids[batch_idx, input_start:input_start +
                            token_len] = input_ids[batch_idx,
                                                   temp_start:temp_start +
                                                   token_len]

        return fused_input_ids[0]

    @nvtx_range("Inputprocessor _pre_process()")
    def _preprocess(self, text_prompt: dict[str, any], mm_data: dict[str, any],
                    mm_processor_kwargs: Dict[str, Any]):
        images = []
        is_video_list = []

        if mm_data.get("image", None) is not None:
            images = mm_data.get("image", None)
            is_video_list = [False] * len(images)

        preprocessed_image = self.processor(
            images=images,
            is_video_list=is_video_list,
            **mm_processor_kwargs,
        )
        input_ids = self.tokenizer.encode(text_prompt,
                                          add_special_tokens=False,
                                          return_tensors="pt")
        return input_ids, preprocessed_image

    @nvtx_range("Inputprocessor __call__()")
    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})

        input_ids, preprocessed_image = self._preprocess(
            text_prompt, mm_data, mm_processor_kwargs)

        fused_input_ids = self._post_process(input_ids, preprocessed_image)

        # CASE 1: Sending raw image data
        # NOTE: For now, I am using "mm_embeding" in tensor format to send the image data to the model.
        # images = mm_data.get("image", None)
        # for i in range(len(images)):
        #     images[i] = torch.from_numpy(np.array(images[i]))
        # sending_image_data = torch.cat(images, dim = 0)
        # SHAPE: (512, 512, 3) => 786,432

        # CASE 2: Sending preprocessed image data
        sending_image_data = torch.cat(preprocessed_image['pixel_values'][0],
                                       dim=0)
        # SHAPE: (5, 3, 378, 378) => 2,143,260

        return fused_input_ids.to(torch.int32).tolist(), {
            "mm_embedding": sending_image_data,
        }


@register_auto_model("HCXVisionForCausalLM")
@register_input_processor(HCXVisionInputProcessor)
class HCXVisionForCausalLM(PreTrainedModel):

    def __init__(self, model_config: ModelConfig):
        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        self.vision_model_init()

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = PretrainedConfig.from_dict(
            llm_model_config.pretrained_config.language_config)
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.model_dtype = getattr(config, "torch_dtype", torch.float16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True
        self.dummy_pixel_values = torch.randn(5, 3, 378, 378).to('cuda')

    def load_weights(self, weights):

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v.to(self.dtype)
                    assert result[new_k] is not None
            return result

        weights = filter_weights("language_model", weights)
        self.llm.load_weights(weights)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def vision_model_init(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            trust_remote_code=True,
            use_fast=False)
        # Vision related initialization
        config = self.model_config.pretrained_config
        # NOTE: Maybe, we need to do from_config and load_weights to get the vision model like LLM does.
        model = HFAutoModelForCausalLM.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            trust_remote_code=True)
        model.eval()
        device = 'cuda'
        self.vision_model = model.vision_model.to(device)
        self.mm_projector = model.mm_projector.to(device)
        self.image_newline = model.image_newline.to(device)

        self.vision_config = config.vision_config
        self.unpad = config.unpad
        self.use_nth_layer = config.use_nth_layer
        self.anyres = config.anyres

        possible_resolutions = []
        if config.anyres:
            assert config.max_num_grids > 0
            for i in range(1, config.max_num_grids + 1):
                for j in range(1, config.max_num_grids + 1):
                    if i == 1 and j == 1 and not config.use_1x1_grid:
                        continue
                    if i * j <= config.max_num_grids:
                        possible_resolutions.append([i, j])

            possible_resolutions = [[
                ys * config.vision_config["image_size"],
                xs * config.vision_config["image_size"]
            ] for ys, xs in possible_resolutions]

        self.possible_resolutions = possible_resolutions

    @nvtx_range("image_preprocess()")
    def _image_preprocess(self, mm_raw_data: List[Any]):
        preprocessed_image_list = []
        for i in range(len(mm_raw_data)):
            # NOTE: Currently assuming single image inference
            images = mm_raw_data[i]
            is_video_list = [False]
            preprocessed_image = self.processor(
                images=images,
                is_video_list=is_video_list,
            )
            preprocessed_image["pixel_values"] = [[
                pixel_value.pin_memory() for pixel_value in pixel_values_inner
            ] for pixel_values_inner in preprocessed_image["pixel_values"]]
            preprocessed_image["pixel_values"] = [[
                pixel_value.to('cuda', non_blocking=True)
                for pixel_value in pixel_values_inner
            ] for pixel_values_inner in preprocessed_image["pixel_values"]]
            # preprocessed_image["pixel_values"] = [[self.dummy_pixel_values.to('cuda') for pixel_value in pixel_values_inner] for pixel_values_inner in preprocessed_image["pixel_values"]]
            preprocessed_image_list.append(preprocessed_image)

        # Transform the list of dictionaries into a dictionary of lists
        combined_preprocessed = {}
        for key in preprocessed_image_list[0].keys():
            combined_preprocessed[key] = [
                d[key][0] for d in preprocessed_image_list
            ]
        return combined_preprocessed

    @nvtx_range("dummy_image_preprocess()")
    def _dummy_image_preprocess(self, mm_raw_data: List[Any]):
        preprocessed_image = {}

        # Retrieve the preprocessed image from the input processor
        preprocessed_image["pixel_values"] = [[mm_raw]
                                              for mm_raw in mm_raw_data]

        # Use dummy_pixel_values
        # preprocessed_image["pixel_values"] = [[self.dummy_pixel_values] for _ in range(len(mm_raw_data))]

        # extra mm_data that should be retrieved from the request level
        preprocessed_image["image_sizes"] = [[[512, 512]]
                                             for _ in range(len(mm_raw_data))]
        preprocessed_image["is_videos"] = [[False]
                                           for _ in range(len(mm_raw_data))]
        preprocessed_image["vision_query_lengths"] = [
            [423] for _ in range(len(mm_raw_data))
        ]
        preprocessed_image["num_queries_vis_abstractors"] = [
            [81] for _ in range(len(mm_raw_data))
        ]
        preprocessed_image["num_queries_vis_abstractors_slow"] = [
            [0] for _ in range(len(mm_raw_data))
        ]
        preprocessed_image["first_last_frames_slows"] = [
            [False] for _ in range(len(mm_raw_data))
        ]
        return preprocessed_image

    @nvtx_range("get_mm_embeddings()")
    def get_mm_embeddings(self, mm_raw_data: List[Any]) -> torch.Tensor:

        # NOTE: This should be done in the input processor and got the preprocessed_image metadata from request level.
        # preprocessed_image = self._image_preprocess(mm_raw_data)

        # NOTE: For now, using dummy image preprocess to measure the desired inference time.
        preprocessed_image_dummy = self._dummy_image_preprocess(mm_raw_data)
        # preprocessed_image_dummy["pixel_values"] = preprocessed_image["pixel_values"]
        preprocessed_image = preprocessed_image_dummy

        pixel_values = preprocessed_image.get("pixel_values", None)
        image_sizes = preprocessed_image.get("image_sizes", None)
        is_videos = preprocessed_image.get("is_videos", None)
        num_queries_vis_abstractors = preprocessed_image.get(
            "num_queries_vis_abstractors", None)
        num_queries_vis_abstractors_slow = preprocessed_image.get(
            "num_queries_vis_abstractors_slow", None)
        first_last_frames_slows = preprocessed_image.get(
            "first_last_frames_slows", None)

        # Flatten CLIP and connector for feature encoding, then convert back to List of List format
        len_pixel_values = [len(pixel_value) for pixel_value in pixel_values]
        concat_pixel_values = torch.cat(list(chain(*pixel_values)),
                                        dim=0)  # list of list of 4D Tensor
        visual_token_idx = 0 if "siglip" in self.vision_config[
            "model_type"] else 1

        n_chunks = 1
        total_len = concat_pixel_values.size(0)
        # Calculate the size of each chunk based on total data length (divided into 10 chunks)
        chunk_size = math.ceil(total_len / n_chunks) if total_len > 0 else 1
        image_forward_outs_chunks = []

        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            # Current chunk slice (could be an empty tensor if there's no data)
            chunk = concat_pixel_values[start:end].to(self.vision_model.dtype)
            # If the current chunk size is smaller than chunk_size, pad with dummy data
            if chunk.size(0) < chunk_size:
                pad_size = chunk_size - chunk.size(0)
                # Create dummy tensor based on concat_pixel_values shape
                dummy_shape = (pad_size, ) + tuple(
                    concat_pixel_values.shape[1:])
                dummy = torch.zeros(
                    dummy_shape,
                    dtype=concat_pixel_values.dtype,
                    device=concat_pixel_values.device,
                )
                chunk = torch.cat([chunk, dummy], dim=0)

            # Pass the chunk through the vision model (processed according to use_nth_layer)
            if self.use_nth_layer == -1:
                # Replace post_layernorm of the last layer with Identity
                self.vision_model.vision_model.post_layernorm = nn.Identity()
                outs = self.vision_model(chunk)
                outs = outs.last_hidden_state[:, visual_token_idx:]
            else:
                outs = self.vision_model(chunk, output_hidden_states=True)
                outs = outs.hidden_states[self.use_nth_layer][:,
                                                              visual_token_idx:]
            image_forward_outs_chunks.append(outs)

        # Concatenate results from all chunks
        image_forward_outs = torch.cat(image_forward_outs_chunks, dim=0).to(
            image_forward_outs_chunks[0].dtype)

        if num_queries_vis_abstractors is None:
            assert num_queries_vis_abstractors_slow is None
            image_sizes = list(chain(*image_sizes))
            if is_videos is not None:
                is_videos = list(chain(*is_videos))
            group_ids = None
            image_forward_outs = image_forward_outs.to(
                dtype=self.mm_projector.dtype)
            image_forward_outs = self.mm_projector(image_forward_outs)
        else:
            (
                num_queries_vis_abstractors,
                num_grids,
                image_sizes,
                is_videos,
                group_ids,
            ) = compute_adaptive_params(
                pixel_values,
                num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow,
                image_sizes,
                is_videos,
                first_last_frames_slows,
            )

            image_forward_outs = image_forward_outs.to(
                dtype=self.mm_projector.dtype)
            image_forward_outs = self.mm_projector(
                image_forward_outs,
                num_queries_vis_abstractors=num_queries_vis_abstractors,
                num_grids=num_grids,
            )
        if self.anyres:
            split_sizes = [
                pixel_value.shape[0] for pixel_value in chain(*pixel_values)
            ]
            if num_queries_vis_abstractors is None:
                image_features = anyres_postprocessing(
                    image_forward_outs=image_forward_outs,
                    split_sizes=split_sizes,
                    image_sizes=image_sizes,
                    num_queries_vis_abstractor=self.num_queries_vis_abstractor,
                    unpad=self.unpad,
                    is_videos=is_videos,
                    patch_size=self.vision_model.config.patch_size,
                    grid_size=self.vision_model.config.image_size,
                    image_newline=self.image_newline,
                    possible_resolutions=self.possible_resolutions,
                )
            else:
                image_features = adaptive_anyres_postprocessing(
                    image_forward_outs=image_forward_outs,
                    image_sizes=image_sizes,
                    num_queries_vis_abstractors=num_queries_vis_abstractors,
                    unpad=self.unpad,
                    is_videos=is_videos,
                    grid_size=self.vision_model.config.image_size,
                    image_newline=self.image_newline,
                    possible_resolutions=self.possible_resolutions,
                    group_ids=group_ids,
                )
        else:
            if num_queries_vis_abstractors is None:
                image_features = [
                    image_forward_out
                    for image_forward_out in image_forward_outs
                ]
            else:
                image_features = [
                    image_forward_out.unsqueeze(0)
                    for image_forward_out in image_forward_outs
                ]
        image_features = [
            image_features[sum(len_pixel_values[:i]):sum(len_pixel_values[:i +
                                                                          1])]
            for i in range(len(len_pixel_values))
        ]
        mm_embeds = [
            torch.cat(list(chain(image_feature)), dim=0)
            for image_feature in image_features
        ]
        return mm_embeds

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        mm_raw_data = kwargs.get("multi_modal_data", [])

        error_msg = "Number of multimodal features (if provided) should be equal to number of context requests"
        assert mm_raw_data == [] or len(
            mm_raw_data) == num_context_requests, error_msg
        if mm_raw_data != []:
            mm_embeds = self.get_mm_embeddings(mm_raw_data)
        else:
            mm_embeds = []

        input_ids, input_embeds = fuse_input_embeds(self.llm.model.embed_tokens,
                                                    input_ids, mm_embeds)
        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits)

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob
