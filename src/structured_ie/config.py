"""Default configuration values for structured_ie."""

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for NER and Relation Extraction models."""

    # Default model identifiers from Hugging Face Hub
    default_gliner_model: str = os.getenv("DEFAULT_GLINER_MODEL", "urchade/gliner_multi-v2.1")
    default_glirel_model: str = os.getenv("DEFAULT_GLIREL_MODEL", "jackboyla/glirel-large-v0")

    # Inference parameters
    default_entity_threshold: float = 0.5
    default_relation_threshold: float = 0.5

    # Device
    default_device: str = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_CONFIG = PipelineConfig()
