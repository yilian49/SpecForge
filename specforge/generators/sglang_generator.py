"""SGLang-based implementation of hidden states generator."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseHiddenStatesGenerator, GeneratorArgs

# SGLang imports
try:
    from sglang.bench_one_batch import BenchArgs, load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.logits_processor import (
        LogitsProcessor,
        LogitsProcessorOutput,
    )
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils import (
        DeepEPMode,
        configure_logger,
        get_bool_env_var,
        require_mlp_sync,
        require_mlp_tp_gather,
        set_gpu_proc_affinity,
    )
    from transformers import AutoConfig

    SGLANG_AVAILABLE = True
except ImportError as e:
    SGLANG_AVAILABLE = False
    import_error = e


class LogitsProcessorForEAGLE3(nn.Module):
    """Wrapper for LogitsProcessor to capture hidden states for EAGLE3."""

    def __init__(self, logits_processor):
        super().__init__()
        self.logits_processor = logits_processor

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        ret = self.logits_processor.forward(
            input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states
        )
        ret.last_hidden_states = hidden_states
        return ret


def wrap_logits_processors_in_module(module: nn.Module):
    """Wrap LogitsProcessor modules to capture hidden states."""
    for name, submodule in module.named_modules():
        if isinstance(submodule, LogitsProcessor):
            wrapped = LogitsProcessorForEAGLE3(submodule)
            setattr(module, name, wrapped)
            print(f"Wrapped {name} with LogitsProcessorForEAGLE3")


class SglangHiddenStatesGenerator(BaseHiddenStatesGenerator):
    """SGLang implementation of hidden states generator."""

    def __init__(self, args, tp_rank: int = 0):
        if not SGLANG_AVAILABLE:
            raise ImportError(
                f"SGLang is not available. Please install SGLang first. "
                f"Original error: {import_error}"
            )

        # Convert argparse namespace to GeneratorArgs if needed
        if not isinstance(args, GeneratorArgs):
            # Extract the fields we need for the base class
            base_args = GeneratorArgs(
                model_path=args.model_path,
                output_path=args.output_path,
                max_length=args.max_length,
                batch_size=getattr(args, "batch_size", [1]),
                enable_aux_hidden_states=args.enable_aux_hidden_states,
                aux_hidden_states_layers=args.aux_hidden_states_layers,
                seed=getattr(args, "seed", 42),
                tp_size=getattr(args, "tp_size", 1),
                cache_dir=getattr(args, "cache_dir", None),
                trust_remote_code=getattr(args, "trust_remote_code", False),
                profile=getattr(args, "profile", False),
            )
            super().__init__(base_args, tp_rank)
            # Keep the original args for SGLang-specific fields
            self.sglang_args = args
        else:
            super().__init__(args, tp_rank)
            self.sglang_args = args

        self.model_runner = None
        self.server_args = None
        self.port_args = None
        self.bench_args = None

    def initialize_model(self):
        """Initialize SGLang model for hidden states extraction."""
        # Create BenchArgs and ServerArgs from our args
        # Use the original sglang_args which has all the SGLang-specific fields

        self.bench_args = BenchArgs.from_cli_args(self.sglang_args)
        self.server_args = ServerArgs.from_cli_args(self.sglang_args)

        # Configure server args for hidden states extraction
        self.server_args.enable_return_hidden_states = True
        self.server_args.context_length = self.args.max_length
        self.server_args.cuda_graph_max_bs = max(self.bench_args.batch_size)
        self.server_args.cuda_graph_bs = list(self.bench_args.batch_size)

        # Set environment variables and config
        _set_envs_and_config(self.server_args)

        # Initialize port args
        self.port_args = PortArgs.init_new(self.server_args)

        # Set CPU affinity if needed
        if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
            set_gpu_proc_affinity(
                self.server_args.tp_size, self.server_args.nnodes, self.tp_rank
            )

        # Configure logger
        configure_logger(self.server_args, prefix=f" TP{self.tp_rank}")

        # Load model
        self.model_runner, _ = load_model(
            self.server_args, self.port_args, self.tp_rank
        )

        # Wrap logits processors
        wrap_logits_processors_in_module(self.model_runner.model)

        self._print_with_rank(f"Initialized SGLang model: {self.args.model_path}")

        # Setup auxiliary hidden states capture if needed
        if self.args.enable_aux_hidden_states:
            # Auto-detect layers if not specified
            if self.args.aux_hidden_states_layers is None:
                config = AutoConfig.from_pretrained(
                    self.args.model_path,
                    trust_remote_code=self.server_args.trust_remote_code,
                )
                if hasattr(config, "num_hidden_layers"):
                    num_layers = config.num_hidden_layers
                elif hasattr(config, "text_config"):
                    num_layers = config.text_config.num_hidden_layers
                else:
                    raise ValueError(
                        f"Config does not have num_hidden_layers or text_config.num_hidden_layers"
                    )

                # In sglang, when we do set_eagle3_layers_to_capture, we will add 1 to the layer index
                self.args.aux_hidden_states_layers = [
                    2 - 1,  # Early layer
                    num_layers // 2 - 1,  # Middle layer
                    num_layers - 3 - 1,  # Late layer
                ]

                assert (
                    len(self.args.aux_hidden_states_layers) == 3
                ), "aux_hidden_states_layers is expected to be 3 layers"

                self._print_with_rank(
                    f"Capturing aux hidden states layers: {self.args.aux_hidden_states_layers}, "
                    f"num_layers: {num_layers}"
                )

            self._setup_aux_hidden_states_capture()

    def _setup_aux_hidden_states_capture(self):
        """Setup auxiliary hidden states capture for Eagle3."""
        if not hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            raise ValueError(
                f"Model {self.model_runner.model} does not support Eagle3 auxiliary hidden states. "
                f"Please ensure you're using the correct model implementation."
            )

        # Set layers to capture
        self.model_runner.model.set_eagle3_layers_to_capture(
            self.args.aux_hidden_states_layers
        )

        # Verify capture is enabled
        if hasattr(self.model_runner.model, "capture_aux_hidden_states"):
            assert (
                self.model_runner.model.capture_aux_hidden_states
            ), "capture_aux_hidden_states should be True"
        elif hasattr(
            self.model_runner.model.language_model, "capture_aux_hidden_states"
        ):
            assert (
                self.model_runner.model.language_model.capture_aux_hidden_states
            ), "capture_aux_hidden_states should be True"
        else:
            raise ValueError(
                f"Model {self.model_runner.model} does not have capture_aux_hidden_states attribute"
            )

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch):
        """Prepare MLP sync batch if needed."""
        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_cpu_group=self.model_runner.tp_group.cpu_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                enable_two_batch_overlap=self.model_runner.server_args.enable_two_batch_overlap,
                enable_deepep_moe=self.model_runner.server_args.enable_deepep_moe,
                deepep_mode=DeepEPMode[self.model_runner.server_args.deepep_mode],
            )

    @torch.no_grad()
    def extract_hidden_states(
        self, batch_data: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Extract hidden states using SGLang forward pass."""

        # Create sampling params (temperature=0 for deterministic)
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)

        # Create requests from batch data
        reqs = []
        for idx, data in enumerate(batch_data):
            input_ids = data["input_ids"].view(-1).tolist()

            req = Req(
                rid=str(idx),
                origin_input_text="",  # Not needed for hidden states extraction
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
            )

            # Set required attributes for forward pass
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1

            reqs.append(req)

        # Create schedule batch
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=None,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            enable_custom_logit_processor=False,
        )

        # Prepare for extend
        batch.prepare_for_extend()

        # Prepare MLP sync if needed
        self._maybe_prepare_mlp_sync_batch(batch)

        # Get model worker batch
        model_worker_batch = batch.get_model_worker_batch()

        # Create forward batch
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        # Forward pass
        logits_output, _ = self.model_runner.forward(forward_batch)

        # Extract hidden states
        input_lens = [len(req.origin_input_ids) for req in reqs]

        if self.args.enable_aux_hidden_states:
            # Extract both last hidden states and auxiliary hidden states
            assert (
                hasattr(logits_output, "last_hidden_states")
                and logits_output.last_hidden_states is not None
            ), "Please use https://github.com/zyksir/sglang/tree/eagle3-offline"

            hidden_states_list = torch.split(
                logits_output.last_hidden_states, input_lens, dim=0
            )
            aux_hidden_states_list = torch.split(
                logits_output.hidden_states, input_lens, dim=0
            )
        else:
            # Extract only last hidden states
            hidden_states_list = torch.split(
                logits_output.hidden_states, input_lens, dim=0
            )
            aux_hidden_states_list = None

        # Clear pools to free memory
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

        return hidden_states_list, aux_hidden_states_list
