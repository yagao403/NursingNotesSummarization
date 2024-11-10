# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

# base BART model
# use generate() to iterately generate next tokens
# augment the base model by patient level cross attention
# augment the base model by prompting with temporal information
# apply gumebel softmax to sample next tokens
# generate summary onehot representation
""" PyTorch BART model."""

import math
import warnings
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers import BartConfig, BartModel, BartPreTrainedModel
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.activations import ACT2FN
from transformers.generation import LogitsProcessorList, StoppingCriteriaList, validate_stopping_criteria

from transformers.models.bart.modeling_bart import BartAttention, BartDecoderLayer, BartDecoder, BartLearnedPositionalEmbedding

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput)

from transformers.file_utils import ModelOutput
from transformers.generation import stopping_criteria, logits_process
from transformers.generation.utils import GenerateEncoderDecoderOutput

logger = logging.get_logger(__name__)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@dataclass
class Seq2SeqModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states_patient: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class Seq2SeqLMOutput(ModelOutput):

    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_hidden_states_patient: Optional[Tuple[torch.FloatTensor, ...]] = None

class GenerateEncoderDecoderOutput(ModelOutput):

    sequences: torch.LongTensor = None
    summary_onehot: torch.FloatTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    #logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None

class BartDecoderLayer_PIAug(BartDecoderLayer):
    def __init__(self,config: BartConfig,  patient_info=True):
        super().__init__(config)
        self.patient_info = patient_info
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.activation_fn = ACT2FN[config.activation_function]
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.patientcross_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_patient: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        patient_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states # save the decoder input for later use

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            cross_atten_output, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            cross_atten_output = nn.functional.dropout(cross_atten_output, p=self.dropout, training=self.training)
            #hidden_states = residual + hidden_states
            #hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

            if not self.patient_info:
                encoder_hidden_states_patient = None

            patient_cross_atten_output = None
            # Patient Cross-Attention Block
            if encoder_hidden_states_patient is not None:
                patient_cross_att_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                patient_cross_atten_output, patient_cross_attn_weights, patient_cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states_patient,
                attention_mask=patient_attention_mask, # prepare later
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=patient_cross_att_past_key_value,
                output_attentions=output_attentions,
            )
                patient_cross_atten_output =  nn.functional.dropout(patient_cross_atten_output, p=self.dropout, training=self.training)
                # if encoder_hidden_states_patient is not None, add cross-attn to positions 5,6 of present_key_value tuple
                present_key_value = present_key_value + patient_cross_attn_present_key_value


            hidden_states = residual + cross_atten_output + self.config.lambda2*patient_cross_atten_output



            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights,)
            if encoder_hidden_states_patient is not None:
                outputs += (patient_cross_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BartDecoder_AUG(BartDecoder):

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)


        # if embed_tokens is not None:
        #     self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        if self.config.augmentation_layers == 'all':
            self.layers = nn.ModuleList([BartDecoderLayer_PIAug(config) for _ in range(config.decoder_layers)])
        elif self.config.augmentation_layers == 'first 6':
            self.layers = nn.ModuleList([BartDecoderLayer_PIAug(config) if i < 6 else BartDecoderLayer(config) for i in range(config.decoder_layers)])
        elif self.config.augmentation_layers == 'last 6':
            self.layers = nn.ModuleList([BartDecoderLayer_PIAug(config) if i >= 6 else BartDecoderLayer(config) for i in range(config.decoder_layers)])
        elif self.config.augmentation_layers == 'None':
            self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        else:
            raise ValueError("The augmentation layers should be 'all', 'first 6', 'last 6' or 'None'.")
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # get or set decoder input embeddings
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_patient: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        patient_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        temporal_info_embedding: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_previous_notes = False,
        add_patient_info = False,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
            inputs_embeds = inputs_embeds * self.embed_scale
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            if add_previous_notes and input[:, 0] == 50265:
                if input.shape[1] <= 1:
                    raise ValueError("In such case, the input_ids should have at least two tokens, but it has only one.")
                else:
                    inputs_embeds = self.embed_tokens(input[:, 1:]) * self.embed_scale
                    inputs_embeds = torch.cat([temporal_info_embedding.unsquzeeze(1), inputs_embeds], dim=1) # temporal_info_embedding batch_size * embed_dim
            else:
                inputs_embeds = self.embed_tokens(input) * self.embed_scale


        attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
        # not None -> add_patient_information = True
        if encoder_hidden_states_patient is not None and patient_attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            patient_attention_mask = _prepare_4d_attention_mask(
                    patient_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        # if augment the model with patient embeddings
        # all_patient_cross_attentions = () if (output_attentions and encoder_hidden_states_patient is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:

                layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_patient = encoder_hidden_states_patient,
                        encoder_attention_mask=encoder_attention_mask,
                        patient_attention_mask = patient_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(
                            cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                        ),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0] #layer_outputs: Tuple(hidden_states, self_attn_weights, cross_attn_weights, (patient_cross_attn_weights,) present_key_value)

            if use_cache:
                next_decoder_cache += (layer_outputs[4 if output_attentions else 1],) if len(layer_outputs)>4 else (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],layer_outputs[3],) if len(layer_outputs)>4 else (layer_outputs[2],) # also output patient_cross_attn_weights

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class Bart_QGSumm(BartModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.decoder = BartDecoder_AUG(config, self)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def set_input_embeddings_for_the_starting_token(self, value):
        self.shared = value
        self.decoder.embed_tokens = self.shared

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        patient_attention_mask = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        encoder_outputs_patient: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        temporal_info_embedding: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_previous_notes = False,
        add_patient_info = False,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # remove:
        # if patient_embedding is None:
        #     patient_ids_mapping = {k:v for k, v in zip(patient_ID, patient_input_ids)} #{id: input_ids}
        #     patient_embedding_mask_mapping = {k:v for k, v in zip(patient_ID, patient_embedding_mask)} #{id: attention_mask}
        #     patient_embedding_mapping = {}
        #     for id in patient_ids_mapping.keys():
        #         p_ids = patient_ids_mapping[id]
        #         p_attention_mask = patient_embedding_mask_mapping[id]
        #         patient_embedding_mapping[id] = self.encoder(p_ids, p_attention_mask).last_hidden_state

        #     patient_embedding = []
        #     for id in patient_ID:
        #         patient_embedding.append(patient_embedding_mapping[id])

        #     patient_embedding = torch.stack(patient_embedding)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if add_patient_info:
            if encoder_outputs_patient is None:
                encoder_outputs_patient = self.encoder(
                    input_ids=input_ids,
                    attention_mask=patient_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
            elif return_dict and not isinstance(encoder_outputs_patient, BaseModelOutput):
                encoder_outputs_patient = BaseModelOutput(
                    last_hidden_state=encoder_outputs_patient[0],
                    hidden_states=encoder_outputs_patient[1] if len(encoder_outputs_patient) > 1 else None,
                    attentions=encoder_outputs_patient[2] if len(encoder_outputs_patient) > 2 else None,
                )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_hidden_states_patient=encoder_outputs_patient[0] if add_patient_info else None,
            encoder_attention_mask=attention_mask,
            patient_attention_mask=patient_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            temporal_info_embedding = temporal_info_embedding,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            add_previous_notes = add_previous_notes,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_last_hidden_state_patient=encoder_outputs_patient.last_hidden_state if add_patient_info else None,
        )

class BartForConditionalGeneration_QGSumm(BartPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = Bart_QGSumm(config)

    def get_encoder(self):
        return self.model.get_encoder()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        patient_attention_mask = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        encoder_outputs_patient: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        temporal_info_embedding: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_previous_notes = False,
        add_patient_info = False,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            patient_attention_mask = patient_attention_mask,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            encoder_outputs_patient=encoder_outputs_patient,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            temporal_info_embedding = temporal_info_embedding,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            add_previous_notes = add_previous_notes,
            add_patient_info = add_patient_info,

        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_last_hidden_state_patient=outputs.encoder_last_hidden_state_patient,
        )

    # remove:
    # def prepare_patient_embedding_for_generation(self, patient_ID, patient_input_ids, patient_embedding_mask):
    #     # construct patient embedding
    #     # notes of the same patient have the same patient embedding
    #     patient_ids_mapping = {k:v for k, v in zip(patient_ID, patient_input_ids)} #{id: input_ids}
    #     patient_embedding_mask_mapping = {k:v for k, v in zip(patient_ID, patient_embedding_mask)} #{id: attention_mask}
    #     patient_embedding_mapping = {}
    #     for id in patient_ids_mapping.keys():
    #         p_ids = patient_ids_mapping[id]
    #         p_attention_mask = patient_embedding_mask_mapping[id]
    #         patient_embedding_mapping[id] = self.encoder(p_ids, p_attention_mask).last_hidden_state
    #     patient_embedding = []
    #     for id in patient_ID:
    #         patient_embedding.append(patient_embedding_mapping[id])

    #     patient_embedding = torch.stack(patient_embedding)
    #     return patient_embedding

    def prepare_inputs_for_generation(
        self,
        summary_onehot,
        word_embeds,
        temporal_info_embedding,
        bos_onehot,
        start_onehot,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:] # decoder_input_ids.shape[1] == 1

        if decoder_input_ids.shape[1] > 1:
            if decoder_input_ids[:, 0] == 50265: # used for the first token generation; add previous notes; decoder_input_ids.shape[1] == 3
                inputs_embeds_start = torch.matmul(start_onehot, word_embeds).unsqueeze(1)
                inputs_embeds = torch.matmul(bos_onehot, word_embeds).unsqueeze(1)
                inputs_embeds = torch.cat([inputs_embeds_start, inputs_embeds], dim=1)
                inputs_embeds = torch.cat([temporal_info_embedding.unsqueeze(1), inputs_embeds], dim=1)
            elif decoder_input_ids[:, 0] == 0: # used for the first token generation; not add previous notes; decoder_input_ids.shape[1] == 2
                inputs_embeds_start = torch.matmul(start_onehot, word_embeds).unsqueeze(1)
                inputs_embeds = torch.matmul(bos_onehot, word_embeds).unsqueeze(1)
                inputs_embeds = torch.cat([inputs_embeds_start, inputs_embeds], dim=1)
            else:
                raise ValueError("The first token of decoder_input_ids should be either 50265 or 0.")

        if decoder_input_ids.shape[1] == 1: # used for the following tokens generation
            inputs_embeds = torch.matmul(summary_onehot[:,-1,:-1].unsqueeze(1), word_embeds)

        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_input_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


    def prepare_decoder_input_ids_for_generation(self, add_previous_notes, batch_size, decoder_start_token_id, bos_token_id, device):
        if add_previous_notes:
            decoder_input_ids_star = torch.ones((batch_size,1), dtype=torch.long, device=device)* torch.tensor([50265,decoder_start_token_id,bos_token_id], dtype=torch.long)
        else:
            decoder_input_ids_star = torch.ones((batch_size,1), dtype=torch.long, device=device)* torch.tensor([decoder_start_token_id,bos_token_id], dtype=torch.long)
        return decoder_input_ids_star

    def prob_to_id_by_gumbel_softmax(self, tau, next_tokens_scores):
        onehot = torch.nn.functional.gumbel_softmax(next_tokens_scores, tau=tau, hard=True) # (batch_size, 50264)
        onehot = torch.cat((onehot, torch.zeros(onehot.size(0), 1).to(onehot.device)), dim=1) # (batch_size, 50265)
        #next_tokens = torch.matmul(onehot, torch.arange(onehot.size(1)).to(onehot.device))
        next_tokens = torch.argmax(onehot, dim=-1)
        return next_tokens, onehot # (batch_size, 50265), (batch_size,)

    def generate(
        self,
        add_patient_info: bool = False,
        add_previous_notes: bool = False,
        input_ids: Optional[torch.Tensor] = None,
        patient_input_ids = None,
        patient_attention_mask = None,
        attention_mask: Optional[torch.Tensor] = None,
        temporal_info_embedding = None,
        word_embeds = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
        ):

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if attention_mask is None:
            attention_mask = input_ids.ne(pad_token_id).long

        # if encoder_outputs is None:
        encoder = self.get_encoder()
        encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        if add_patient_info:
            encoder_outputs_patient = encoder(
                input_ids=patient_input_ids,
                attention_mask=patient_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # check shape of patient_embedding and encoder_outputs
            if len(encoder_outputs_patient.last_hidden_state.shape) != 3:
                raise ValueError("patient_embedding must be a 3D tensor")
            if encoder_outputs_patient.last_hidden_state.shape[0] != encoder_outputs.last_hidden_state.shape[0]:
                raise ValueError("The batch size of patient_embedding and encoder_outputs must be the same")

        batch_size = input_ids.shape[0]

        decoder_input_ids = self.prepare_decoder_input_ids_for_generation(add_previous_notes, batch_size=batch_size, decoder_start_token_id=self.config.decoder_start_token_id, bos_token_id=bos_token_id, device=input_ids.device)

        max_length = max_length if max_length is not None else self.config.max_length

        stopping_criteria = self._get_stopping_criteria(max_length=max_length)

        return self.greedy_search(
                input_ids=input_ids,
                patient_attention_mask = patient_attention_mask,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                encoder_outputs_patient = encoder_outputs_patient,
                past_key_values=None,
                temporal_info_embedding = temporal_info_embedding,
                word_embeds=word_embeds,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                use_cache=use_cache,
                add_previous_notes = add_previous_notes,
                add_patient_info = add_patient_info,
                **kwargs,
            )

    def greedy_search(self,
        input_ids: Optional[torch.Tensor] = None,
        patient_attention_mask = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Seq2SeqModelOutput] = None,
        encoder_outputs_patient: Optional[Seq2SeqModelOutput] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        temporal_info_embedding = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        add_previous_notes: bool = False,
        add_patient_info: bool = False,
        use_cache: Optional[bool] = None,):

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        summary_onehot = torch.zeros(decoder_input_ids.shape[0], 1, self.config.vocab_size+1).to(input_ids.device)
        # summary_onehot[:,0,0] = 1 # set the first token to be <BOS>

        pad_onehot = torch.zeros(decoder_input_ids.shape[0], self.config.vocab_size+1).to(pad_token_id.device)
        pad_onehot[:, pad_token_id] = 1
        bos_onehot = torch.zeros(summary_onehot.shape[0], self.config.vocab_size+1).to(summary_onehot.device)
        bos_onehot[:, 0] = 1
        start_onehot = torch.zeros(summary_onehot.shape[0], self.config.vocab_size+1).to(summary_onehot.device)
        start_onehot[:, 2] = 1
        unfinished_sequences = torch.ones(decoder_input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False
        while True:

            # keep only the last token for the next input
            model_inputs = self.prepare_inputs_for_generation(summary_onehot, self.model.shared.weight, temporal_info_embedding, bos_onehot, start_onehot, decoder_input_ids, past_key_values, use_cache)
            outputs = self(
            attention_mask=attention_mask,
            patient_attention_mask=patient_attention_mask,
            decoder_input_ids = None,
            decoder_inputs_embeds = model_inputs["decoder_input_embeds"],
            encoder_outputs=encoder_outputs,
            encoder_outputs_patient = encoder_outputs_patient,
            past_key_values=past_key_values,
            temporal_info_embedding = temporal_info_embedding,
            use_cache=use_cache,
            add_previous_notes = add_previous_notes,
            add_patient_info = add_patient_info,
            )

            next_token_logits = outputs.logits[:, -1, :]
            # next_tokens_scores = logits_processor(decoder_input_ids, next_token_logits) # do we need to process the logits?
            next_tokens_scores = next_token_logits

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # apply gumbel softmax due to the differentiable property
            next_tokens, next_onehot = self.prob_to_id_by_gumbel_softmax(next_tokens_scores) # next_onehot: torch.tensor([batch_size, vocab_size])

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences_mat = unfinished_sequences.view(-1, 1)
                next_onehot = torch.mul(next_onehot,unfinished_sequences_mat) + torch.mul(pad_onehot, 1 - unfinished_sequences_mat)

            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=-1)
            summary_onehot = torch.cat([summary_onehot, next_onehot.unsqueeze(1)], dim=1)

            past_key_values = outputs.past_key_values if use_cache else None

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished:
                break

        summary_onehot = summary_onehot[:,1:,:] # remove the all-zero tensor at the beginning

        # check if the first token is <BOS>, if not, add <BOS> to the beginning

        equal = torch.all(summary_onehot[:, 0, :] == bos_onehot, dim=-1)
        if not equal.all():
            summary_onehot = torch.cat([bos_onehot.unsqueeze(1), summary_onehot], dim=1)

        encoder_attentions = encoder_outputs[2]
        encoder_hidden_states = encoder_outputs[0] if output_hidden_states else None # last layer

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # do not need to output hidden_states for patients
                return GenerateEncoderDecoderOutput(
                    sequences=decoder_input_ids, # required by QGSumm
                    summary_onehot = summary_onehot, # required by QGSumm
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states, # for all layers
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=past_key_values,
                )
