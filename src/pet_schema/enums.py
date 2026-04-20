from __future__ import annotations

from enum import Enum
from typing import Literal

Modality = Literal["vision", "audio", "sensor", "multimodal"]

SourceType = Literal["youtube", "community", "device", "synthetic"]


class EdgeFormat(str, Enum):
    RKLLM = "rkllm"
    RKNN = "rknn"
    ONNX = "onnx"
    GGUF = "gguf"


class PetSpecies(str, Enum):
    CAT = "cat"
    DOG = "dog"
    OTHER = "other"


class BowlType(str, Enum):
    METAL = "metal"
    CERAMIC = "ceramic"
    PLASTIC = "plastic"
    UNKNOWN = "unknown"


class Lighting(str, Enum):
    BRIGHT = "bright"
    DIM = "dim"
    DARK = "dark"
