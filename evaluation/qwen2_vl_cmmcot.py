from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLModel
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import StaticCache, DynamicCache
from qwen_vl_utils import process_vision_info
import re
import torch
import torch.nn as nn
import torch
from fvcore.nn import FlopCountAnalysis

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from evaluation.modeling_qwen2_vl_custom import CustomQwen2VLModel, CustomQwen2VLDecoderLayer



def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CustomGenerationMixin(GenerationMixin):
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        num_new_tokens=1,
    ):
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (attention_mask.shape[0], num_new_tokens)
                        ),
                    ],
                    dim=-1,
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + num_new_tokens
            )
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1,
                past_positions[-1] + num_new_tokens + 1,
                dtype=past_positions.dtype,
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), StaticCache):
            if self.device.type == "cuda":
                logger.warning_once("Using `torch.compile`.")
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # prepare model inputs
            (
                model_inputs,
                origin_input_ids,
                origin_pixel_values,
                origin_image_grid_thw,
            ) = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            num_new_tokens = (
                1
                if origin_input_ids.shape[1] == input_ids.shape[1]
                else (origin_input_ids.shape[1] - input_ids.shape[1] + 1)
            )
            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=num_new_tokens,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([origin_input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs["pixel_values"] = origin_pixel_values
            model_kwargs["image_grid_thw"] = origin_image_grid_thw

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


# Subclassing the model, if necessary
class CustomQwen2VLForConditionalGeneration(
    Qwen2VLForConditionalGeneration, CustomGenerationMixin
):
    def __init__(self, config):
        # Call the parent class's constructor to initialize all basic components
        super().__init__(config)
        # Custom initialization logic
        self.image_inputs = None  # To hold image inputs
        self.processor = None
        self.image_processor = None
        self.image_token = 151655  # Assuming this is a defined constant
        self.vision_start_token = 151652
        self.vision_end_token = 151653
        self.model = CustomQwen2VLModel(config)
        self.image_manager_init = False
        self.image_start_idx = None
        self.new_num_image_tokens = None


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load using the parent class method first
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Process state dict to ensure it matches the model structure
        model_state_dict = model.state_dict()  # Get state dict
        modified_state_dict = cls.modify_state_dict_for_custom_attention(model, model_state_dict)
        
        # Load the modified state_dict
        model.load_state_dict(modified_state_dict, strict=False)
        
        # Ensure any specific custom initialization now, if needed
        for layer_idx, layer in enumerate(model.model.layers):
            if isinstance(layer, CustomQwen2VLDecoderLayer):
                layer.custom_self_attn.load_pretrained_weights(layer.self_attn)
                print(f"Layer {layer_idx}: custom_self_attn loaded weights from self_attn")
        
        return model
    
    @staticmethod
    def modify_state_dict_for_custom_attention(model, state_dict):
        # Add logic to process state dictionary here
        custom_layers_pattern = "custom_self_attn"
        new_state_dict = {}
        for k, v in state_dict.items():
            if custom_layers_pattern in k:
                continue  # Adjust logic based on your requirements
            new_state_dict[k] = v
        return new_state_dict



    def set_image_inputs(self, image_inputs):
        """
        Method to set image inputs outside of the generation call.
        This can be called independently before calling generate.
        """
        self.image_inputs = image_inputs

    def set_processor(self, processor):
        """
        Method to set image inputs outside of the generation call.
        This can be called independently before calling generate.
        """
        self.processor = processor
        self.image_processor = processor.image_processor

    def reset_state(self):
        self.image_manager_init = False
        self.image_start_idx = None
        self.new_num_image_tokens = None

        # Reset any other internal state variables if necessary (e.g., dynamic cache)
        self.model.reset_attention_states()

    def set_kvmanager(self, pixel_values, image_grid_thw, num_input_img):

        input_ids = []
        for i in range(num_input_img):
            input_ids += ([self.vision_start_token]
            + [self.image_token] * (image_grid_thw[i].prod() // 4)
            + [self.vision_end_token])

        input_ids = torch.tensor(input_ids, device=pixel_values.device)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = torch.ones((1, input_ids.shape[1]), device=input_ids.device)

        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=DynamicCache(),
            inputs_embeds=None,
            use_cache=True,
            pixel_values=pixel_values,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            use_custom_attention=True
        )
        return True



    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        assert self.image_inputs is not None, "Image inputs must be provided."
        origin_input_ids = input_ids
        origin_pixel_values = pixel_values
        origin_image_grid_thw = image_grid_thw
        use_custom_attention = False
        
        # print(self.processor.batch_decode(input_ids, skip_special_tokens=False)[0])

        if 151658 in input_ids:
            if not self.image_manager_init:
                self.image_manager_init = self.set_kvmanager(pixel_values, image_grid_thw, image_grid_thw.shape[0])

            use_custom_attention = True
            
            if input_ids[:, -1] == 151658:
                
                # if not self.image_manager_init:
                #     self.image_manager_init = self.set_kvmanager(pixel_values, image_grid_thw, image_grid_thw.shape[0])

                # use_custom_attention = True

                coor_start_ind = torch.where(input_ids == 151648)[1].tolist()[-1]
                decoded_texts = self.processor.batch_decode(
                    input_ids[:, coor_start_ind:], skip_special_tokens=False
                )
                # Check if any decoded text contains the special token </IMG>
                for decoded_text in decoded_texts:
                    # Retrieve coordinates and index
                    match = re.search(
                        r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\><IMG>(\d+)</IMG>",
                        decoded_text,
                    )
                    if match:
                        x1, y1, x2, y2, img_index = map(int, match.groups())

                        # Validate image index
                        if 0 <= img_index < len(self.image_inputs):
                            original_image = self.image_inputs[img_index]
                            try:
                                # Validate and normalize coordinates
                                if (
                                    all(0 <= coord <= 1000 for coord in (x1, y1, x2, y2))
                                    and x1 < x2
                                    and y1 < y2
                                ):
                                    width, height = original_image.size
                                    x1, y1, x2, y2 = [
                                        int(
                                            coord / 1000 * (width if i % 2 == 0 else height)
                                        )
                                        for i, coord in enumerate((x1, y1, x2, y2))
                                    ]
                                    # Crop the image
                                    cropped_image = original_image.crop((x1, y1, x2, y2))
                                else:
                                    raise ValueError(
                                        f"Invalid coordinates ({x1}, {y1}, {x2}, {y2})"
                                    )
                            except Exception as e:
                                print(
                                    f"Error in processing image with index {img_index}: {e}"
                                )
                                cropped_image = original_image  # Fallback to original image

                            # Process cropped image
                            new_image_inputs = self.image_processor(
                                images=cropped_image, return_tensors="pt"
                            )
                            new_pixel_values = torch.tensor(
                                new_image_inputs["pixel_values"], device=input_ids.device
                            )
                            new_image_grid_thw = torch.tensor(
                                new_image_inputs["image_grid_thw"], device=input_ids.device
                            )
                        else:
                            print(
                                f"Image index {img_index} out of range for {len(self.image_inputs)} images."
                            )

                        # Append vision tokens
                        num_image_tokens = new_image_grid_thw.prod() // 4
                        vision_token_ids = (
                            [self.vision_start_token]
                            + [self.image_token] * num_image_tokens
                            + [self.vision_end_token]
                        )
                        vision_token_ids_tensor = torch.tensor(
                            vision_token_ids, device=input_ids.device
                        )

                        input_ids = torch.cat(
                            [input_ids, vision_token_ids_tensor.unsqueeze(0)], dim=1
                        )  # Assuming batch size of 1
                        origin_input_ids = input_ids

                        cache_position = torch.arange(
                            origin_input_ids.shape[1], device=origin_input_ids.device
                        )

                        # Extend the attention mask
                        if attention_mask is not None:
                            # attention_mask = torch.ones((1, origin_input_ids.shape[1]), device=origin_input_ids.device)
                            attention_mask = attention_mask.new_ones(
                                (1, origin_input_ids.shape[1])
                            )

                        # # Reset past_key_values
                        past_key_values = DynamicCache()

                        # Combine with original pixel values if needed
                        origin_pixel_values = torch.cat(
                            [origin_pixel_values, new_pixel_values], dim=0
                        )
                        origin_image_grid_thw = torch.cat(
                            [origin_image_grid_thw, new_image_grid_thw], dim=0
                        )
                        image_start_idx_tmp = (input_ids[0] == self.vision_start_token).nonzero(as_tuple=True)[0][-1].item() if (input_ids[0] == self.vision_start_token).any() else -1
                        self.image_start_idx = torch.cat((self.image_start_idx, torch.tensor(image_start_idx_tmp, device=input_ids.device).unsqueeze(0))) if self.image_start_idx is not None else torch.tensor(image_start_idx_tmp, device=input_ids.device).unsqueeze(0)
                        self.new_num_image_tokens = torch.cat((self.new_num_image_tokens, torch.tensor(num_image_tokens+2, device=input_ids.device).unsqueeze(0))) if self.new_num_image_tokens is not None else torch.tensor(num_image_tokens+2, device=input_ids.device).unsqueeze(0)
            else: # 从第一张小图以后都不能用cache了
                cache_position = torch.arange(
                    origin_input_ids.shape[1], device=origin_input_ids.device
                )

                # Extend the attention mask
                if attention_mask is not None:
                    # attention_mask = torch.ones((1, origin_input_ids.shape[1]), device=origin_input_ids.device)
                    attention_mask = attention_mask.new_ones(
                        (1, origin_input_ids.shape[1])
                    )

                # # Reset past_key_values
                past_key_values = DynamicCache()

        # The rest of your prepare_inputs_for_generation logic
        model_inputs = super().prepare_inputs_for_generation(
            origin_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=origin_pixel_values,  # use the new pixel values
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=origin_image_grid_thw,  # use the new grid thw
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

        if use_custom_attention:
            model_inputs['use_custom_attention'] = use_custom_attention
            model_inputs['image_start_idx'] = self.image_start_idx
            model_inputs['new_num_image_tokens'] = self.new_num_image_tokens


        return (
            model_inputs,
            origin_input_ids,
            origin_pixel_values,
            origin_image_grid_thw,
        )
        # return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_start_idx: Optional[torch.Tensor] = None,  # New parameter
        use_custom_attention: Optional[bool] = None,  # New parameter
        new_num_image_tokens: Optional[torch.Tensor] = None,  # New parameter
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            image_start_idx=image_start_idx,
            use_custom_attention=use_custom_attention,
            new_num_image_tokens=new_num_image_tokens,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


if __name__ == "__main__":

    model_path = "/mnt/workspace/xiaoxi/xiaoxi-all/model_logs/vllm_factory/qwen2_vl-7b/mcot/full_sft_mcot-interleave-noimglabel-singleturn_2024_12_19_23_39_59/checkpoint-1000"
    model = CustomQwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Count and print the number of parameters
    num_parameters = count_model_parameters(model)
    print(f"Number of model parameters: {num_parameters}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/workspace/xiaoxi/code/VoCoT/040.png",
                },
                {
                    "type": "image",
                    "image": "/mnt/workspace/xiaoxi/code/VoCoT/040_1.png",
                },
                {
                    "type": "image",
                    "image": "/mnt/workspace/xiaoxi/code/VoCoT/040_0.png",
                },
                {
                    "type": "text",
                    "text": "Answer the following question:\nIf image 1 is the reference image, which image of the other two is more similar to the reference image? (Assume reference image is image 1 and the other two are image 2 and image 3)",
                },
            ],
        }
    ]



    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # # Set image inputs separately
    model.set_image_inputs(image_inputs)
    model.set_processor(processor)


    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    tensor_inputs = inputs['input_ids']

    # Compute FLOPs using fvcore
    try:
        flops_info = FlopCountAnalysis(model, tensor_inputs)
        print(f"Estimated FLOPs: {flops_info.total()}")  # Total FLOPs for the given inputs
    except Exception as e:
        print(f"Failed to compute FLOPs: {e}")

    generated_ids = model.generate(**inputs, 
                                    max_new_tokens=8192,
                                    temperature=0.8,  # Increased from 0.7
                                    top_k=50,         # Added top_k sampling
                                    top_p=0.95,       # Added nucleus sampling
                                    do_sample=True ,   # Enable sampling
                                    repetition_penalty=1.05)


    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    print(output_text)
