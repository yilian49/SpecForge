"""HuggingFace Transformers implementation of hidden states generator."""

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .base import BaseHiddenStatesGenerator, GeneratorArgs


class TransformerHiddenStatesGenerator(BaseHiddenStatesGenerator):
    """HuggingFace Transformers implementation of hidden states generator."""

    def __init__(self, args: GeneratorArgs, tp_rank: int = 0):
        super().__init__(args, tp_rank)
        self.tokenizer = None

    def initialize_model(self):
        """Initialize HuggingFace model for hidden states extraction."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=self.args.trust_remote_code,
            cache_dir=self.args.cache_dir,
        )

        # Load config
        self.config = AutoConfig.from_pretrained(
            self.args.model_path,
            trust_remote_code=self.args.trust_remote_code,
            cache_dir=self.args.cache_dir,
        )

        # Configure for hidden states output
        self.config.output_hidden_states = True
        self.config.use_cache = False

        # Load model based on TP size
        if self.args.tp_size > 1:
            # Use distributed model for tensor parallelism
            try:
                from specforge.modeling.auto import AutoDistributedTargetModel

                # To avoid CPU RAM OOM, directly init the model on CUDA
                self.model = AutoDistributedTargetModel.from_pretrained(
                    pretrained_model_name_or_path=self.args.model_path,
                    torch_dtype=torch.bfloat16,
                    cache_dir=self.args.cache_dir,
                    device="cuda",
                    trust_remote_code=self.args.trust_remote_code,
                ).eval()
                self._print_with_rank(
                    f"Initialized distributed HuggingFace model with TP={self.args.tp_size}: {self.args.model_path}"
                )
            except ImportError as e:
                raise ImportError(
                    f"Tensor parallel support requires AutoDistributedTargetModel from specforge.modeling.auto. "
                    f"Error: {e}"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                config=self.config,
                torch_dtype=torch.bfloat16,
                cache_dir=self.args.cache_dir,
                trust_remote_code=self.args.trust_remote_code,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            ).eval()
            self._print_with_rank(
                f"Initialized HuggingFace model: {self.args.model_path}"
            )

        if self.args.enable_aux_hidden_states:
            self._print_with_rank(
                f"Will capture auxiliary hidden states at layers: {self.args.aux_hidden_states_layers}"
            )

    @torch.no_grad()
    def extract_hidden_states(
        self, batch_data: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Extract hidden states using HuggingFace forward pass."""

        # Prepare batch tensors
        input_ids_list = [data["input_ids"] for data in batch_data]
        max_len = max(len(ids) for ids in input_ids_list)

        # Pad sequences (pad on the left for causal LM)
        padded_input_ids = []
        attention_masks = []

        for ids in input_ids_list:
            padding_length = max_len - len(ids)
            if padding_length > 0:
                # Use pad_token_id if available, otherwise use 0
                pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

                # Pad on the left for causal LM
                padded_ids = torch.cat(
                    [
                        torch.full(
                            (padding_length,),
                            pad_token_id,
                            dtype=ids.dtype,
                            device=ids.device,
                        ),
                        ids,
                    ]
                )
                mask = torch.cat(
                    [
                        torch.zeros(
                            padding_length, dtype=torch.long, device=ids.device
                        ),
                        torch.ones(len(ids), dtype=torch.long, device=ids.device),
                    ]
                )
            else:
                padded_ids = ids
                mask = torch.ones(len(ids), dtype=torch.long, device=ids.device)

            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        # Stack into batch tensors
        input_ids = torch.stack(padded_input_ids).cuda()
        attention_mask = torch.stack(attention_masks).cuda()

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states
        all_hidden_states = (
            outputs.hidden_states
        )  # Tuple of tensors (num_layers+1, batch, seq, hidden)

        # Get the last hidden states (from the final layer)
        # Note: hidden_states includes embeddings as layer 0, so last layer is at index -1
        last_hidden_states = all_hidden_states[-1]  # (batch, seq, hidden)

        # Extract only the non-padded portions
        hidden_states_list = []
        input_lens = [len(data["input_ids"]) for data in batch_data]

        for i, length in enumerate(input_lens):
            # Get the actual sequence (excluding padding)
            # Since we padded on the left, take the last 'length' tokens
            hidden_state = last_hidden_states[i, -length:, :]
            hidden_states_list.append(hidden_state)

        # Extract auxiliary hidden states if needed
        aux_hidden_states_list = None
        if self.args.enable_aux_hidden_states and self.args.aux_hidden_states_layers:
            aux_hidden_states_list = []

            for i, length in enumerate(input_lens):
                # Collect hidden states from specified layers
                aux_states = []
                for layer_idx in self.args.aux_hidden_states_layers:
                    # Add 1 to layer_idx because hidden_states includes embeddings at index 0
                    actual_layer_idx = layer_idx + 1
                    if actual_layer_idx < len(all_hidden_states):
                        layer_hidden = all_hidden_states[actual_layer_idx][i]
                        # Extract the actual sequence (last 'length' tokens since we padded left)
                        layer_hidden = layer_hidden[-length:, :]
                        aux_states.append(layer_hidden)
                    else:
                        self._print_with_rank(
                            f"Warning: Layer {layer_idx} not available in model "
                            f"(total layers: {len(all_hidden_states)-1})"
                        )

                # Stack auxiliary hidden states
                if aux_states:
                    # Shape: (num_aux_layers, seq, hidden)
                    aux_hidden = torch.stack(aux_states, dim=0)
                    aux_hidden_states_list.append(aux_hidden)
                else:
                    aux_hidden_states_list.append(None)

        return hidden_states_list, aux_hidden_states_list
