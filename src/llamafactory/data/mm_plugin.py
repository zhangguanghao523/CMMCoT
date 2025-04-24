import math
import os
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union
import re

import numpy as np
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import is_pillow_available, is_pyav_available
from . import fs

if is_pillow_available():
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = 1000000000
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if TYPE_CHECKING:
    import torch
    from av.stream import Stream
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, ImageObject]
    VideoInput = str


def _get_paligemma_token_type_ids(
    imglens: Sequence[int], seqlens: Sequence[int], processor: "ProcessorMixin"
) -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, sequence_length)
    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seqlen")
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                # image = Image.open(image)
                # image = Image.open(fs.read(os.path.join("oss://mvap-public-data/shufangxun/dataset/llvm/SFT/VQA/", image)))
                image = Image.open(fs.read(image))
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return results

    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        """
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)

        return results

    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128),
                video_fps=getattr(processor, "video_fps", 1.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            seqlens: number of tokens in each sample, shape (batch_size,)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(images, videos)
        return {}


class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen")
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", self.image_token * image_seqlen)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if "image_sizes" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])
        if "pixel_values" in mm_inputs:
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))
        for message in messages:
            content = message["content"]
            while self.image_token in content:
                image_size = next(image_sizes)
                orig_height, orig_width = image_size
                image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
                num_image_tokens += 1
                content = content.replace(self.image_token, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens")
        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        res = self._get_mm_inputs(images, videos, processor)
        return res


class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        num_video_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if "pixel_values" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))
            for message in messages:
                content = message["content"]

                while self.image_token in content:
                    image_size = next(image_sizes)
                    orig_height, orig_width = image_size
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                    num_image_tokens += 1
                    content = content.replace(self.image_token, "{{image}}" * image_seqlen, 1)

                message["content"] = content.replace("{{image}}", self.image_token)

        if "pixel_values_videos" in mm_inputs:
            pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
            height, width = get_image_size(pixel_values_video[0])
            num_frames = pixel_values_video.shape[0]  # frame dim is always after batch dim
            image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
            video_seqlen = image_seqlen // 4 * num_frames  # divide by 4 needed for avg pooling layer

            for message in messages:
                content = message["content"]
                while self.video_token in content:
                    num_video_tokens += 1
                    content = content.replace(self.video_token, "{{video}}", 1)
                message["content"] = content.replace("{{video}}", self.video_token * video_seqlen)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {IMAGE_PLACEHOLDER} tokens")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", "")

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens")

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(images, videos)
        num_images = len(images)
        image_seqlen = num_images * getattr(processor, "image_seqlen")
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        patch_size = getattr(processor, "patch_size")
        image_token = getattr(processor, "image_token")
        image_break_token = getattr(processor, "image_break_token")
        image_end_token = getattr(processor, "image_end_token")

        num_image_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_input_sizes = mm_inputs.get("image_sizes", None)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if image_input_sizes is None:
                    raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens")

                image_size = image_input_sizes[0][num_image_tokens]
                height, width = image_size
                num_height_tokens = height // patch_size
                num_width_tokens = width // patch_size
                replace_tokens = [[image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                replace_tokens[-1] = image_end_token
                replace_str = "".join(replace_tokens)
                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)
                num_image_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if mm_inputs.get("pixel_values"):
            mm_inputs["pixel_values"] = mm_inputs["pixel_values"][0]

        mm_inputs.pop("image_sizes", None)
        return mm_inputs


class Qwen2vlPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self.get_mm_inputs(messages, images, videos, processor) # img 处理完了，qwen2-vl processor也过了
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        # if len(images) != num_image_tokens:
        #     raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens")

        # if len(videos) != num_video_tokens:
        #     raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens")

        return messages, mm_inputs


    @override
    def _regularize_images(self, messages: Sequence[Dict[str, str]], images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """

        # 函数来从gpt的value中提取图像索引和边界框坐标
        def extract_boxes_and_indices(value):
            img_pattern = r'<IMG>(\d+)</IMG>'
            box_pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
            
            # 提取图像索引
            img_indices = re.findall(img_pattern, value)
            img_indices = [int(idx) for idx in img_indices]

            # 提取边界框坐标
            box_coords = re.findall(box_pattern, value)
            box_coords = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in box_coords]
            
            return box_coords, img_indices

        # Load and convert all input images to ImageObject
        images_pil = []
        for image_input in images:
            try:
                if isinstance(image_input, str):
                    image = Image.open(image_input.replace("oss://mvap-public-data/", "/data/oss_bucket_0/")) # oss 挂载
                    # image = Image.open(fs.read(image_input))
                elif isinstance(image_input, bytes):
                    image = Image.open(BytesIO(image_input))
                elif isinstance(image_input, dict):
                    if "bytes" in image_input:
                        image = Image.open(BytesIO(image_input["bytes"]))
                    elif "path" in image_input:
                        image = Image.open(image_input["path"])
                    else:
                        raise ValueError("Image input dictionary must have 'bytes' or 'path' key.")
                else:
                    raise TypeError(f"Unsupported image input type: {type(image_input)}")
            
            except Exception as e:
                print(f"pull fail because: {e}")
                image = Image.new('RGB', (224, 224), (255, 255, 255))
            if image.width==0 or image.height==0:
                image = Image.new('RGB', (224, 224), (255, 255, 255))
            try:
                image = self._preprocess_image(image, **kwargs) # resize to reduce memory
            except Exception as e:
                print(f"An error occurred during image preprocessing: {e}")
                image = Image.new('RGB', (224, 224), (255, 255, 255))
            
            images_pil.append(image)

        # Crop the images based on the boxes specified in the messages
        cropped_images = []
        for message in messages:
            if message['role'] == 'assistant':
                box_coords, img_indices = extract_boxes_and_indices(message['content'])
                for (x1, y1, x2, y2), idx in zip(box_coords, img_indices):
                    try:
                        width, height = images_pil[idx].size
                        x1, y1, x2, y2 = [int(coord / 1000 * (width if i % 2 == 0 else height)) for i, coord in enumerate((x1, y1, x2, y2))]
                        # Ensure the coordinates are valid for cropping
                        if x1 < x2 and y1 < y2:
                            cropped_image = images_pil[idx].crop((x1, y1, x2, y2))
                            cropped_images.append(cropped_image)
                        else:
                            raise ValueError("Invalid crop coordinates.")
                    except ValueError as e:
                        print(f"Skipping cropping due to error: {e}")
                        cropped_images.append(images_pil[idx])  # Use the original image

        # Preprocess all the images (original and cropped)
        results = [self._preprocess_image(image, **kwargs) for image in images_pil + cropped_images]
 
        return results



    @override
    def get_mm_inputs(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:

        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                messages,
                images,
                image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128),
                video_fps=getattr(processor, "video_fps", 1.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs


class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        num_video_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        num_frames = 0
        exist_images = "pixel_values_images" in mm_inputs
        exist_videos = "pixel_values_videos" in mm_inputs
        if exist_videos or exist_images:
            if exist_images:
                height, width = get_image_size(to_numpy_array(mm_inputs.get("pixel_values_images")[0]))
                num_frames = 1
            if exist_videos:
                pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(pixel_values_video[0])
                num_frames = pixel_values_video.shape[0]  # frame dim is always after batch dim
            image_seqlen = (height // processor.patch_size) * (width // processor.patch_size) + 1
            video_seqlen = image_seqlen * num_frames
            if processor.vision_feature_select_strategy == "default":
                image_seqlen -= 1
            for message in messages:
                content = message["content"]
                while self.image_token in content:
                    num_image_tokens += 1
                    content = content.replace(self.image_token, "{{image}}", 1)
                while self.video_token in content:
                    num_video_tokens += 1
                    content = content.replace(self.video_token, "{{video}}", 1)

                content = content.replace("{{image}}", self.image_token * image_seqlen)
                message["content"] = content.replace("{{video}}", self.video_token * video_seqlen)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {self.image_token} tokens")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {self.video_token} tokens")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2_vl": Qwen2vlPlugin,
    "video_llava": VideoLlavaPlugin,
}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return plugin_class(image_token, video_token)
