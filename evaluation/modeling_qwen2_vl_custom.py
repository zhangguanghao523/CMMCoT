import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm

import transformers
from transformers.models.qwen2_vl.modeling_qwen2_vl import *
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

class CustomQwen2VLSdpaAttention(Qwen2VLSdpaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.stored_keys = None
        self.stored_values = None
        # self.new_image_query_states_list = None
        self.new_image_query_states_list = []

    def load_pretrained_weights(self, source_attention_module):
        # Assuming the source and target weights and biases are compatible
        self.q_proj.weight.data.copy_(source_attention_module.q_proj.weight.data)
        self.q_proj.bias.data.copy_(source_attention_module.q_proj.bias.data)
        self.k_proj.weight.data.copy_(source_attention_module.k_proj.weight.data)
        self.k_proj.bias.data.copy_(source_attention_module.k_proj.bias.data)
        self.v_proj.weight.data.copy_(source_attention_module.v_proj.weight.data)
        self.v_proj.bias.data.copy_(source_attention_module.v_proj.bias.data)
        self.o_proj.weight.data.copy_(source_attention_module.o_proj.weight.data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_start_idx: Optional[torch.Tensor] = None,
        new_num_image_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        if image_start_idx is not None:
            for i, start_idx in enumerate(image_start_idx):
                end_idx = start_idx + new_num_image_tokens[i]
                
                # Extract the current slice of query states that corresponds to the current image tokens
                image_query_states = query_states[:, :, start_idx:end_idx, :]

                if (self.new_image_query_states_list is [] or 
                    i >= len(self.new_image_query_states_list)):
                    
                    # Compute new image query states using scaled dot product attention
                    new_image_query_states = torch.nn.functional.scaled_dot_product_attention(
                        image_query_states,
                        self.stored_keys,
                        self.stored_values,
                        attn_mask=None,
                        dropout_p=self.attention_dropout if self.training else 0.0
                    )
                    
                    self.new_image_query_states_list.append(new_image_query_states)
                else:
                    # If already present, update the query states with the precomputed values
                    query_states[:, :, start_idx:end_idx, :] = self.new_image_query_states_list[i]


        else:
            self.stored_keys = key_states.clone()
            self.stored_values = value_states.clone()
            # Returning hidden_states unchanged as this is an image-only input phase
            return hidden_states, None, None

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



class CustomQwen2VLDecoderLayer(Qwen2VLDecoderLayer):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Instantiate custom attention
        self.custom_self_attn = CustomQwen2VLSdpaAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_start_idx: Optional[int] = None,
        use_custom_attention: bool = False,
        new_num_image_tokens: Optional[torch.Tensor] = None,  # New parameter
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if use_custom_attention:
            hidden_states, self_attn_weights, present_key_value = self.custom_self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                image_start_idx=image_start_idx,
                new_num_image_tokens=new_num_image_tokens
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CustomQwen2VLModel(Qwen2VLModel):

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([CustomQwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])



    def reset_attention_states(self):
        # Iterate over each layer that uses CustomQwen2VLSdpaAttention
        for layer in self.layers:
            if isinstance(layer.custom_self_attn, CustomQwen2VLSdpaAttention):
                # print("reset")
                layer.custom_self_attn.new_image_query_states_list = []
                layer.custom_self_attn.stored_keys = None
                layer.custom_self_attn.stored_values = None



    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None, 
        past_key_values: Optional[List[torch.FloatTensor]] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        use_cache: Optional[bool] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None, 
        cache_position: Optional[torch.LongTensor] = None, 
        image_start_idx: Optional[int] = None, 
        new_num_image_tokens: Optional[torch.Tensor] = None,  # New parameter
        use_custom_attention: Optional[bool] = None, 
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        use_custom_attention_indices = [0, 4, 8, 12, 16, 20, 24, 27]
        # use_custom_attention = kwargs.pop("use_custom_attention", False)
        use_custom_attention = use_custom_attention
        for i, decoder_layer in enumerate(self.layers):
            # print(f"layers: {i}")
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Determine the use_custom_attention value based on the index
            current_use_custom_attention = use_custom_attention if i in use_custom_attention_indices else False
            # print(f"layers: {i}, use_custom_attention: {current_use_custom_attention}")

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    image_start_idx=image_start_idx,
                    use_custom_attention=use_custom_attention,
                    new_num_image_tokens=new_num_image_tokens
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    image_start_idx=image_start_idx,
                    use_custom_attention=use_custom_attention,
                    new_num_image_tokens=new_num_image_tokens
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
