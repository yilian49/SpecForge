#!/usr/bin/env python
"""Script for offline hidden states generation for Eagle3 training with backend selection."""

import argparse
import hashlib
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

# Add parent directory to path to import specforge modules
sys.path.append(str(Path(__file__).parent.parent))

from specforge.data import build_eagle3_dataset
from specforge.generators import GeneratorArgs, create_hidden_states_generator
from specforge.utils import print_on_rank0, print_with_rank, rank_0_priority, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate hidden states for Eagle3 offline training"
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        choices=["huggingface", "sglang"],
        help="Backend to use for hidden states generation",
    )

    # Model and data arguments
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the target model"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to training data (JSON file)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory for hidden states",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory for models and data",
    )

    # Processing arguments
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Chat template to use for preprocessing",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (for debugging)",
    )

    # Hidden states arguments
    parser.add_argument(
        "--enable-aux-hidden-states",
        action="store_true",
        help="Enable auxiliary hidden states extraction",
    )
    parser.add_argument(
        "--aux-hidden-states-layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices for auxiliary hidden states",
    )

    # Performance arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=None,
        help="Batch size(s) for processing",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")

    # Distributed arguments
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (works with both HuggingFace and SGLang backends)",
    )
    parser.add_argument(
        "--dist-timeout", type=int, default=None, help="Distributed timeout in seconds"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models",
    )

    # Backend-specific arguments (SGLang)
    # These will be passed through if using SGLang backend
    if (
        "--backend" in sys.argv
        and sys.argv[sys.argv.index("--backend") + 1] == "sglang"
    ):
        try:
            from sglang.bench_one_batch import BenchArgs
            from sglang.srt.server_args import ServerArgs

            ServerArgs.add_cli_args(parser)
            BenchArgs.add_cli_args(parser)
        except ImportError:
            print("Warning: SGLang not installed, cannot add SGLang-specific arguments")

    args = parser.parse_args()

    # Parse aux_hidden_states_layers if provided
    if args.aux_hidden_states_layers:
        args.aux_hidden_states_layers = [
            int(x.strip()) for x in args.aux_hidden_states_layers.split(",")
        ]

    return args


def initialize_distributed(args):
    """Initialize distributed training if needed."""
    # Get tp_size from args (works for both backends now)
    tp_size = getattr(args, "tp_size", 1)
    tp_rank = 0

    if tp_size > 1:
        # Initialize distributed
        timeout_kwargs = {}
        if args.dist_timeout is not None:
            if args.dist_timeout <= 0:
                raise ValueError(
                    f"--dist-timeout must be positive, got {args.dist_timeout}"
                )
            timeout_kwargs["timeout"] = timedelta(seconds=args.dist_timeout)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", **timeout_kwargs)

        tp_rank = dist.get_rank()
        print_with_rank(
            f"Initialized distributed environment (rank {tp_rank}/{tp_size})"
        )
    else:
        print("Running in single GPU mode")

    return tp_size, tp_rank


def main():
    """Main function for hidden states generation."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Initialize distributed if needed
    tp_size, tp_rank = initialize_distributed(args)

    # Set default output path if not provided
    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        args.output_path = root_path / "cache" / "hidden_states" / args.backend
        args.output_path = str(args.output_path)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    print_on_rank0(f"Output directory: {args.output_path}")

    # Validate dataset path
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset path {args.data_path} does not exist")

    # Load and preprocess dataset
    print_on_rank0(f"Loading dataset from {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path)["train"]

    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print_on_rank0(f"Using {args.num_samples} samples for debugging")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )

    # Build Eagle3 dataset
    print_on_rank0("Building Eagle3 dataset...")
    cache_key = hashlib.md5(args.data_path.encode()).hexdigest()

    with rank_0_priority():
        eagle3_dataset = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
        )

    print_with_rank(f"Built dataset with {len(eagle3_dataset)} samples")

    # Create generator arguments based on backend
    if args.backend == "sglang":
        # For SGLang, we pass the full args object since it has SGLang-specific fields
        # The SGLang generator will extract what it needs
        gen_args = args
    else:
        # For HuggingFace, create GeneratorArgs
        gen_args = GeneratorArgs(
            model_path=args.model_path,
            output_path=args.output_path,
            max_length=args.max_length,
            batch_size=args.batch_size or [1],
            enable_aux_hidden_states=args.enable_aux_hidden_states,
            aux_hidden_states_layers=args.aux_hidden_states_layers,
            seed=args.seed,
            tp_size=args.tp_size,  # Now properly passed for HuggingFace too
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            profile=args.profile,
        )

    # Create generator
    print_on_rank0(f"Creating {args.backend} generator...")
    try:
        if args.backend == "sglang":
            # For SGLang, use the original class directly with the args
            from specforge.generators.sglang_generator import (
                SglangHiddenStatesGenerator,
            )

            generator = SglangHiddenStatesGenerator(args, tp_rank=tp_rank)
        else:
            # For other backends, use the factory function
            generator = create_hidden_states_generator(
                generator_type=args.backend, args=gen_args, tp_rank=tp_rank
            )
    except (ImportError, ValueError) as e:
        print(f"Error creating {args.backend} generator: {e}")
        sys.exit(1)

    # Generate hidden states
    print_on_rank0(f"Starting hidden states generation with {args.backend} backend...")
    generator.generate(eagle3_dataset)

    print_on_rank0(
        f"Hidden states generation complete! Output saved to {args.output_path}"
    )

    # Cleanup distributed if initialized
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
