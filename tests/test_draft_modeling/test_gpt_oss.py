import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from specforge.modeling.draft.gpt_oss_eagle import (
    GptOssDraftAttention,
    GptOssDraftDecoderLayer,
    GptOssDraftMLP,
    GptOssExperts,
    GptOssForCausalLMEagle3,
    GptOssTopKRouter,
)


class TestGptOssForCausalLMEagle3Loading(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()

        config_dict = {
            "architectures": ["GptOssForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 384,
            "initializer_range": 0.02,
            "intermediate_size": 512,
            "max_position_embeddings": 2048,
            "model_type": "gpt_oss",
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,  # Target model layers
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.28.1",
            "use_cache": True,
            "vocab_size": 1000,
            "draft_vocab_size": 500,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "head_dim": 48,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "layer_types": ["full_attention", "sliding_attention"],
        }

        self.config = GptOssConfig(**config_dict)

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test that the model initializes with correct components."""
        model = GptOssForCausalLMEagle3(self.config)

        # Check that we have a single decoder layer
        self.assertIsInstance(model.decoder_layer, GptOssDraftDecoderLayer)
        self.assertIsInstance(model.decoder_layer.self_attn, GptOssDraftAttention)
        self.assertIsInstance(model.decoder_layer.mlp, GptOssDraftMLP)
        
        # Check MoE components
        self.assertIsInstance(model.decoder_layer.mlp.router, GptOssTopKRouter)
        self.assertIsInstance(model.decoder_layer.mlp.experts, GptOssExperts)
        
        # Check layer normalization
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm
        self.assertIsInstance(model.decoder_layer.input_layernorm, GptOssRMSNorm)
        self.assertIsInstance(model.decoder_layer.post_attention_layernorm, GptOssRMSNorm)
        self.assertIsInstance(model.norm, GptOssRMSNorm)
        
        # Check dimensions
        self.assertEqual(model.decoder_layer.hidden_size, self.config.hidden_size)
        self.assertEqual(model.vocab_size, self.config.vocab_size)
        self.assertEqual(model.draft_vocab_size, self.config.draft_vocab_size)

    def test_attention_type_is_full_attention(self):
        """Test that the draft model uses full attention (no sliding window)."""
        model = GptOssForCausalLMEagle3(self.config)
        
        # Verify attention type
        self.assertEqual(model.decoder_layer.attention_type, "full_attention")
        self.assertIsNone(model.decoder_layer.self_attn.sliding_window)
        
        # Check that we're not using sliding window
        self.assertFalse(hasattr(model.decoder_layer.self_attn, 'sliding_window') and 
                        model.decoder_layer.self_attn.sliding_window is not None)

    def test_moe_components(self):
        """Test MoE router and experts configuration."""
        model = GptOssForCausalLMEagle3(self.config)
        router = model.decoder_layer.mlp.router
        experts = model.decoder_layer.mlp.experts
        
        # Check router configuration
        self.assertEqual(router.num_experts, self.config.num_local_experts)
        self.assertEqual(router.top_k, self.config.num_experts_per_tok)
        self.assertEqual(router.hidden_dim, self.config.hidden_size)
        
        # Check experts configuration
        self.assertEqual(experts.num_experts, self.config.num_local_experts)
        self.assertEqual(experts.hidden_size, self.config.hidden_size)
        self.assertEqual(experts.intermediate_size, self.config.intermediate_size)
        
        # Check parameter shapes
        self.assertEqual(experts.gate_up_proj.shape, 
                        (self.config.num_local_experts, self.config.hidden_size, 2 * self.config.intermediate_size))
        self.assertEqual(experts.down_proj.shape,
                        (self.config.num_local_experts, self.config.intermediate_size, self.config.hidden_size))

    def test_vocab_buffers(self):
        """Test vocabulary mapping buffers."""
        model = GptOssForCausalLMEagle3(self.config)
        
        # Check that vocab buffers exist
        self.assertTrue(hasattr(model, 't2d'))
        self.assertTrue(hasattr(model, 'd2t'))
        
        # Check buffer shapes and types
        self.assertEqual(model.t2d.shape, (self.config.vocab_size,))
        self.assertEqual(model.d2t.shape, (self.config.draft_vocab_size,))
        self.assertEqual(model.t2d.dtype, torch.bool)
        self.assertEqual(model.d2t.dtype, torch.int64)

    def test_abstract_methods(self):
        """Test implementation of Eagle3DraftModel abstract methods."""
        model = GptOssForCausalLMEagle3(self.config)
        batch_size = 2
        seq_len = 10
        
        # Test embed_input_ids
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        embeds = model.embed_input_ids(input_ids)
        self.assertEqual(embeds.shape, (batch_size, seq_len, self.config.hidden_size))
        
        # Test project_hidden_states
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 3)
        projected = model.project_hidden_states(hidden_states)
        self.assertEqual(projected.shape, (batch_size, seq_len, self.config.hidden_size))
        
        # Test compute_logits
        hidden = torch.randn(batch_size, seq_len, self.config.hidden_size)
        logits = model.compute_logits(hidden)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.draft_vocab_size))

    def test_model_forward_pass(self):
        """Test the model's forward pass."""
        model = GptOssForCausalLMEagle3(self.config)
        model.eval()

        batch_size = 2
        seq_len = 10

        # Simulate concatenated hidden states from 3 target model layers
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 3)
        inputs_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        self.assertEqual(outputs.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_forward_with_ttt_cache(self):
        """Test forward pass with test-time training cache."""
        model = GptOssForCausalLMEagle3(self.config)
        model.eval()

        batch_size = 1
        seq_len = 5

        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 3)
        inputs_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size)

        with torch.no_grad():
            # Test with ttt_length > 1 (should create cache)
            outputs_cached = model(
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                ttt_length=3,
            )
            
            # Test with ttt_length = 1 (no cache)
            outputs_no_cache = model(
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                ttt_length=1,
            )

        self.assertEqual(outputs_cached.shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertEqual(outputs_no_cache.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_backbone_method(self):
        """Test the backbone method."""
        model = GptOssForCausalLMEagle3(self.config)
        model.eval()

        batch_size = 2
        seq_len = 8

        input_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = model.prepare_decoder_attention_mask(
            torch.ones(batch_size, seq_len, dtype=torch.bool),
            hidden_states,
            batch_size,
            seq_len,
            0
        )

        with torch.no_grad():
            output = model.backbone(
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                cache_hidden=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_save_pretrained(self):
        """Test the model's save_pretrained functionality."""
        model = GptOssForCausalLMEagle3(self.config)

        self.config.save_pretrained(self.temp_dir)

        model_path = os.path.join(self.temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(os.path.exists(model_path))

    @patch("transformers.modeling_utils.PreTrainedModel.from_pretrained")
    def test_from_pretrained_mock(self, mock_from_pretrained):
        """Test loading model from pretrained (mocked)."""
        mock_model = GptOssForCausalLMEagle3(self.config)
        mock_from_pretrained.return_value = mock_model

        loaded_model = GptOssForCausalLMEagle3.from_pretrained(self.temp_dir)
        mock_from_pretrained.assert_called_once_with(self.temp_dir)
        self.assertIsInstance(loaded_model, GptOssForCausalLMEagle3)

    def test_state_dict_compatibility(self):
        """Test state dict saving and loading."""
        model1 = GptOssForCausalLMEagle3(self.config)
        model2 = GptOssForCausalLMEagle3(self.config)

        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            self.assertTrue(torch.equal(param1, param2))

    def test_config_validation(self):
        """Test model creation with invalid config."""
        # Test missing required attributes
        invalid_config = GptOssConfig(
            vocab_size=1000,
            hidden_size=384,
            num_attention_heads=8,
            # Missing required MoE parameters
        )
        
        with self.assertRaises((AttributeError, TypeError)):
            GptOssForCausalLMEagle3(invalid_config)

    def test_parameter_count(self):
        """Test that the model has reasonable parameter count for a draft model."""
        model = GptOssForCausalLMEagle3(self.config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Draft model should be much smaller than target model
        # Rough check - should be under 10M parameters for this config
        self.assertLess(total_params, 10_000_000)
        self.assertGreater(total_params, 100_000)  # But not trivially small

    def test_attention_weights_shapes(self):
        """Test attention component weight shapes."""
        model = GptOssForCausalLMEagle3(self.config)
        attn = model.decoder_layer.self_attn
        
        # Check projection shapes (concatenated input = hidden_size * 2)
        expected_q_shape = (self.config.hidden_size * 2, self.config.num_attention_heads * attn.head_dim)
        expected_k_shape = (self.config.hidden_size * 2, self.config.num_key_value_heads * attn.head_dim)
        expected_v_shape = (self.config.hidden_size * 2, self.config.num_key_value_heads * attn.head_dim)
        expected_o_shape = (self.config.num_attention_heads * attn.head_dim, self.config.hidden_size)
        
        self.assertEqual(attn.q_proj.weight.shape, expected_q_shape)
        self.assertEqual(attn.k_proj.weight.shape, expected_k_shape)
        self.assertEqual(attn.v_proj.weight.shape, expected_v_shape)
        self.assertEqual(attn.o_proj.weight.shape, expected_o_shape)
        
        # Check sinks parameter
        self.assertEqual(attn.sinks.shape, (self.config.num_attention_heads,))


class TestGptOssComponents(unittest.TestCase):
    """Test individual components of the GPT-OSS draft model."""
    
    def setUp(self):
        self.config = GptOssConfig(
            hidden_size=384,
            intermediate_size=512,
            num_local_experts=4,
            num_experts_per_tok=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_bias=False,
            attention_dropout=0.0,
            rms_norm_eps=1e-6,
        )

    def test_router_forward(self):
        """Test router forward pass."""
        router = GptOssTopKRouter(self.config)
        
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        router_scores, router_indices = router(hidden_states)
        
        # Check output shapes
        self.assertEqual(router_scores.shape, (batch_size * seq_len, self.config.num_local_experts))
        self.assertEqual(router_indices.shape, (batch_size * seq_len, self.config.num_experts_per_tok))
        
        # Check that scores sum to 1 (approximately) for selected experts
        selected_scores = torch.gather(router_scores, 1, router_indices)
        score_sums = selected_scores.sum(dim=1)
        self.assertTrue(torch.allclose(score_sums, torch.ones_like(score_sums), atol=1e-6))

    def test_experts_forward(self):
        """Test experts forward pass."""
        experts = GptOssExperts(self.config)
        router = GptOssTopKRouter(self.config)
        
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        router_scores, router_indices = router(hidden_states)
        output = experts(hidden_states, router_indices, router_scores)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_mlp_forward(self):
        """Test MLP (router + experts) forward pass."""
        mlp = GptOssDraftMLP(self.config)
        
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output, router_scores = mlp(hidden_states)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertEqual(router_scores.shape, (batch_size * seq_len, self.config.num_local_experts))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTest(unittest.makeSuite(TestGptOssForCausalLMEagle3Loading))
    suite.addTest(unittest.makeSuite(TestGptOssComponents))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)