"""Model abstraction layer for FYP inference backends."""

from fyp.models.base import GenerationConfig, ModelConfig, ModelRequest, ModelResponse
from fyp.models.ollama_client import OllamaClient
from fyp.models.registry import get_model_config, load_model_config

__all__ = [
    "GenerationConfig",
    "ModelConfig",
    "ModelRequest",
    "ModelResponse",
    "OllamaClient",
    "get_model_config",
    "load_model_config",
]
