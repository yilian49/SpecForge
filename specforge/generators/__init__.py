"""Hidden states generators for Eagle3 training."""

from .base import BaseHiddenStatesGenerator, GeneratorArgs
from .transformer_generator import TransformerHiddenStatesGenerator

# Import SGLang generator only if available
try:
    from .sglang_generator import SglangHiddenStatesGenerator

    _SGLANG_AVAILABLE = True
except ImportError:
    _SGLANG_AVAILABLE = False


def create_hidden_states_generator(
    generator_type: str, args: GeneratorArgs, tp_rank: int = 0
) -> BaseHiddenStatesGenerator:
    """
    Factory function to create the appropriate hidden states generator.

    Args:
        generator_type: Type of generator ("huggingface" or "sglang")
        args: Generator arguments
        tp_rank: Tensor parallel rank

    Returns:
        Instance of the requested generator

    Raises:
        ValueError: If generator_type is unknown or unavailable
    """

    generators = {
        "huggingface": TransformerHiddenStatesGenerator,
    }

    if _SGLANG_AVAILABLE:
        generators["sglang"] = SglangHiddenStatesGenerator

    if generator_type not in generators:
        available = list(generators.keys())
        raise ValueError(
            f"Unknown or unavailable generator type: {generator_type}. "
            f"Available types: {available}"
        )

    return generators[generator_type](args, tp_rank)


__all__ = [
    "BaseHiddenStatesGenerator",
    "GeneratorArgs",
    "TransformerHiddenStatesGenerator",
    "create_hidden_states_generator",
]

if _SGLANG_AVAILABLE:
    __all__.append("SglangHiddenStatesGenerator")
