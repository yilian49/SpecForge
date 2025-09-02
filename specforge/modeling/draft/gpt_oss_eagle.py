# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm

from .base import Eagle3DraftModel

logger = logging.getLogger(__name__)


class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # Simplified experts without tensor parallelism for draft model
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty((self.num_experts, self.expert_dim, self.hidden_size))
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )

        self.alpha = 1.702
        self.limit = 7.0

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        
        if self.training:
            next_states = torch.zeros_like(
                hidden_states, dtype=hidden_states.dtype, device=hidden_states.device
            )
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    router_indices, num_classes=num_experts
                )
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hitted = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0
                ).nonzero()
            
            for expert_idx in expert_hitted[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = (
                    current_state @ self.gate_up_proj[expert_idx]
                    + self.gate_up_proj_bias[expert_idx]
                )
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = (
                    gated_output @ self.down_proj[expert_idx]
                    + self.down_proj_bias[expert_idx]
                )
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype)
                )
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = (
                torch.bmm(hidden_states, self.gate_up_proj)
                + self.gate_up_proj_bias[..., None, :]
            )
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.bmm(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(
                num_experts, batch_size, -1, self.hidden_size
            )
            next_states = (
                next_states
                * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[
                    ..., None
                ]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class GptOssTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        router_top_value = torch.nn.functional.softmax(
            router_top_value, dim=1, dtype=router_top_value.dtype
        )
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        return router_scores, router_indices


class GptOssDraftMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(
            hidden_states, router_indices=router_indices, routing_weights=router_scores
        )
        return routed_out, router_scores


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GptOssRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = 10000.0
        
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings,
            device=inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, position_ids):
        if x.shape[-2] > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=x.shape[-2], device=x.device, dtype=x.dtype)
        
        cos = self.cos_cached[:, :, :x.shape[-2], ...].to(dtype=x.dtype)
        sin = self.sin_cached[:, :, :x.shape[-2], ...].to(dtype=x.dtype)
        
        cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
        sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
        
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward_no_sliding_window(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    sinks: torch.Tensor,
    dropout: float = 0.0,
    **kwargs,
):
    """Full attention forward (no sliding window) with sinks mechanism."""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Add sinks mechanism
    sinks_expanded = sinks.reshape(1, -1, 1, 1).expand(
        query.shape[0], -1, query.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks_expanded], dim=-1)

    # Prevent overflow in BF16/FP16
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # Drop the sink scores
    
    attn_weights = F.dropout(scores, p=dropout, training=kwargs.get('training', False))
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GptOssDraftAttention(nn.Module):
    """Multi-headed attention using FULL ATTENTION (no sliding window)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        # NOTE: Using full attention - NO sliding window
        self.sliding_window = None
        
        # Simplified projections without tensor parallelism for draft model
        self.q_proj = nn.Linear(
            config.hidden_size * 2,  # Concatenated input from Eagle pattern
            self.num_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size * 2, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size * 2, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            config.hidden_size, 
            bias=config.attention_bias
        )
        
        self.rotary_emb = GptOssRotaryEmbedding(config)
        # Sinks for full attention (no sliding window complexity)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if cache_hidden is not None:
            # Handle cached attention states (similar to Eagle3)
            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]
            
            cache_k = cache_hidden[0]
            cache_v = cache_hidden[1]
            
            k0 = cache_k[0]
            v0 = cache_v[0]
            
            key_states = repeat_kv(k0, self.num_key_value_groups)
            value_states = repeat_kv(v0, self.num_key_value_groups)
            
            # Use full attention forward (no sliding window)
            attn_output, attn_weights = eager_attention_forward_no_sliding_window(
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sinks=self.sinks,
                training=self.training,
            )
        else:
            # Standard attention without cache
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            attn_output, attn_weights = eager_attention_forward_no_sliding_window(
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sinks=self.sinks,
                training=self.training,
            )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class GptOssDraftDecoderLayer(nn.Module):
    """Single decoder layer using FULL ATTENTION (no sliding window)."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssDraftAttention(config)
        self.mlp = GptOssDraftMLP(config)
        self.input_layernorm = GptOssRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GptOssRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # This layer uses FULL ATTENTION - no sliding window
        self.attention_type = "full_attention"

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        
        # Normalize and concatenate (Eagle pattern)
        hidden_states_norm = self.input_layernorm(hidden_states)
        input_emb_norm = self.input_layernorm(input_emb)
        
        # Concatenate for attention input
        concat_hidden = torch.cat((input_emb_norm, hidden_states_norm), dim=-1)
        
        # Self Attention - FULL ATTENTION (no sliding window)
        attn_output = self.self_attn(
            hidden_states=concat_hidden,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output
        
        # MLP with MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, router_scores = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states


class GptOssForCausalLMEagle3(Eagle3DraftModel):
    """GPT-OSS Draft Model with single layer using FULL ATTENTION (no sliding window)."""
    
    config_class = GptOssConfig

    def __init__(self, config, quant_config=None) -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, 'draft_vocab_size', config.vocab_size)
        
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        
        # Single decoder layer using FULL ATTENTION
        self.decoder_layer = GptOssDraftDecoderLayer(config)
        
        # Projection layer for 3 concatenated hidden states -> single hidden state
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, self.draft_vocab_size, bias=False
        )

        # Create vocab buffers for draft vocabulary mapping
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the draft model."""
        std = getattr(self.config, 'initializer_range', 0.02)
        
        # Initialize embeddings
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
        
        # Initialize linear layers and MoE components
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, GptOssExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
                nn.init.zeros_(module.gate_up_proj_bias)
                nn.init.normal_(module.down_proj, mean=0.0, std=std)
                nn.init.zeros_(module.down_proj_bias)
            elif isinstance(module, GptOssTopKRouter):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.normal_(module.bias, mean=0.0, std=std)
            elif isinstance(module, GptOssDraftAttention):
                nn.init.normal_(module.sinks, mean=0.0, std=std)
            elif isinstance(module, GptOssRMSNorm):
                nn.init.ones_(module.weight)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the input ids."""
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the concatenated hidden states from 3 layers of target model."""
        assert hidden_states.size(-1) == self.config.hidden_size * 3
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the logits of the draft model."""
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[List[torch.Tensor]]],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Single decoder layer backbone using FULL ATTENTION."""
        return self.decoder_layer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        """
        Forward pass of the draft model using FULL ATTENTION.
        
        Args:
            hidden_states: Concatenated hidden states from 3 layers of target model
            inputs_embeds: Input embeddings
            attention_mask: Attention mask
            ttt_length: Test-time training length
        """
        if ttt_length == 1:
            cache_hidden = None
        else:
            cache_hidden = [[], []]

        batch_size, seq_length, _ = hidden_states.size()

        # Create position ids
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )
        attention_mask = self.prepare_decoder_attention_mask(
            attention_mask, hidden_states, batch_size, seq_length, 0
        )

        # Project concatenated hidden states
        projected_hidden = self.project_hidden_states(hidden_states)

        # Pass through single decoder layer with FULL ATTENTION
        output_hidden = self.backbone(
            input_embeds=inputs_embeds,
            hidden_states=projected_hidden,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )

        return output_hidden