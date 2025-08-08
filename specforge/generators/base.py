import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig


@dataclass
class GeneratorArgs:
    """Common arguments for all generators."""

    model_path: str
    output_path: str
    max_length: int = 2048
    batch_size: List[int] = None
    enable_aux_hidden_states: bool = False
    aux_hidden_states_layers: Optional[List[int]] = None
    seed: int = 42
    tp_size: int = 1
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    profile: bool = False

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = [1]


class BaseHiddenStatesGenerator(ABC):
    """Base class for hidden states generation."""

    def __init__(self, args: GeneratorArgs, tp_rank: int = 0):
        self.args = args
        self.tp_rank = tp_rank
        self.model = None
        self.config = None

        # Initialize aux hidden states layers if needed
        if args.enable_aux_hidden_states and args.aux_hidden_states_layers is None:
            self._auto_detect_aux_layers()

    def _auto_detect_aux_layers(self):
        """Automatically detect auxiliary hidden states layers to capture."""
        config = AutoConfig.from_pretrained(
            self.args.model_path,
            trust_remote_code=self.args.trust_remote_code,
            cache_dir=self.args.cache_dir,
        )

        if hasattr(config, "num_hidden_layers"):
            num_layers = config.num_hidden_layers
        elif hasattr(config, "text_config"):
            num_layers = config.text_config.num_hidden_layers
        else:
            raise ValueError(
                f"Config does not have num_hidden_layers or text_config.num_hidden_layers"
            )

        # Default: capture early, middle, and late layers
        # Note: For SGLang compatibility, we might need to adjust indices by -1
        self.args.aux_hidden_states_layers = [
            1,  # Early layer (2nd layer, 0-indexed)
            num_layers // 2 - 1,  # Middle layer
            num_layers - 3,  # Late layer (3rd from last)
        ]

        assert (
            len(self.args.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers"

        self._print_with_rank(
            f"Auto-detected aux hidden states layers: {self.args.aux_hidden_states_layers}, "
            f"num_layers: {num_layers}"
        )

    def _print_with_rank(self, message: str):
        """Print message with rank information."""
        if dist.is_initialized():
            print(f"[Rank {dist.get_rank()}] {message}")
        else:
            print(message)

    @abstractmethod
    def initialize_model(self):
        """Initialize the model for hidden states generation."""
        pass

    @abstractmethod
    def extract_hidden_states(
        self, batch_data: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Extract hidden states from a batch of data.

        Args:
            batch_data: List of dictionaries containing input_ids and loss_mask

        Returns:
            - List of hidden states tensors
            - Optional list of auxiliary hidden states tensors
        """
        pass

    def _create_output_directory(self, idx: int, group_size: int = 5000) -> str:
        """Create grouped output directory structure."""
        group_start = (idx // group_size) * group_size
        group_end = group_start + group_size
        grouped_subdir = f"rows_{group_start}-{group_end}"

        full_dir = os.path.join(self.args.output_path, grouped_subdir)
        if self.tp_rank == 0 and not os.path.exists(full_dir):
            os.makedirs(full_dir, exist_ok=True)

        return os.path.join(full_dir, f"data_{idx}.ckpt")

    def _save_tensor(
        self,
        hidden_states_cpu: List[Tuple[Any, List[Tuple[Dict, str]]]],
        save_aux_hidden_states: bool,
    ):
        """Save hidden states to disk."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        for idx, (hidden_states, batch_save_info) in enumerate(hidden_states_cpu):
            if idx % world_size != rank:
                continue

            hidden_states_list, aux_hidden_states_list = hidden_states

            if save_aux_hidden_states and aux_hidden_states_list is not None:
                for hidden_state, aux_hidden_state, (data_point, output_file) in zip(
                    hidden_states_list, aux_hidden_states_list, batch_save_info
                ):
                    data_point["hidden_state"] = hidden_state.clone().unsqueeze(0).cpu()
                    data_point["aux_hidden_state"] = (
                        aux_hidden_state.clone().unsqueeze(0).cpu()
                    )

                    # Validate tensors
                    assert not torch.any(
                        torch.isnan(data_point["hidden_state"])
                    ), f"hidden_state is expected to be non-nan"
                    assert not torch.any(
                        torch.isnan(data_point["aux_hidden_state"])
                    ), f"aux_hidden_state is expected to be non-nan"

                    torch.save(data_point, output_file)
            else:
                for hidden_state, (data_point, output_file) in zip(
                    hidden_states_list, batch_save_info
                ):
                    data_point["hidden_state"] = hidden_state.clone().unsqueeze(0).cpu()

                    assert not torch.any(
                        torch.isnan(data_point["hidden_state"])
                    ), f"hidden_state is expected to be non-nan"

                    torch.save(data_point, output_file)

    def generate(self, dataset):
        """Generate hidden states for the entire dataset."""
        MIN_FILE_SIZE = 100 * 1024  # 100KB minimum file size

        # Initialize model
        self.initialize_model()

        # Setup profiler if requested
        profiler = None
        if self.args.profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )
            profiler.start()

        batch_size = self.args.batch_size[0]
        batch_data = []
        batch_save_info = []
        hidden_states_cpu = []
        group_size = 5000

        for idx, row in tqdm(
            enumerate(dataset), total=len(dataset), desc="Generating hidden states"
        ):
            # Create output directory structure
            group_start = (idx // group_size) * group_size
            group_end = group_start + group_size
            grouped_subdir = f"rows_{group_start}-{group_end}"

            if self.tp_rank == 0 and not os.path.exists(
                os.path.join(self.args.output_path, grouped_subdir)
            ):
                os.makedirs(os.path.join(self.args.output_path, grouped_subdir))

            output_file = os.path.join(
                self.args.output_path, grouped_subdir, f"data_{idx}.ckpt"
            )

            # Skip if file already exists
            if (
                os.path.exists(output_file)
                and os.path.getsize(output_file) > MIN_FILE_SIZE
            ):
                continue

            # Prepare data point
            data_point = {
                "input_ids": row["input_ids"].view(-1),
                "loss_mask": row["loss_mask"].view(-1),
            }

            batch_data.append(data_point)
            batch_save_info.append((data_point.copy(), output_file))

            # Process batch when full
            if len(batch_data) == batch_size:
                hidden_states, aux_hidden_states = self.extract_hidden_states(
                    batch_data
                )
                hidden_states_cpu.append(
                    ((hidden_states, aux_hidden_states), batch_save_info[:])
                )

                # Save periodically to manage memory
                if len(hidden_states_cpu) >= 64:
                    torch.cuda.synchronize()
                    self._save_tensor(
                        hidden_states_cpu,
                        save_aux_hidden_states=self.args.enable_aux_hidden_states,
                    )
                    hidden_states_cpu = []
                    torch.cuda.empty_cache()

                batch_data = []
                batch_save_info = []

        # Process remaining batch
        if batch_data:
            hidden_states, aux_hidden_states = self.extract_hidden_states(batch_data)
            hidden_states_cpu.append(
                ((hidden_states, aux_hidden_states), batch_save_info)
            )

        # Save remaining tensors
        if hidden_states_cpu:
            torch.cuda.synchronize()
            self._save_tensor(
                hidden_states_cpu,
                save_aux_hidden_states=self.args.enable_aux_hidden_states,
            )

        # Stop profiler if it was started
        if profiler:
            profiler.stop()
            profiler_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "./profiles")
            os.makedirs(profiler_dir, exist_ok=True)
            profiler.export_chrome_trace(
                os.path.join(
                    profiler_dir,
                    f"hidden_states_gen_rank{self.tp_rank}.trace.json.gz",
                )
            )
            self._print_with_rank(f"Profiling data saved to {profiler_dir}")
