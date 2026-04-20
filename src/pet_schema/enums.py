from __future__ import annotations

from enum import StrEnum
from typing import Literal

Modality = Literal["vision", "audio", "sensor", "multimodal"]

SourceType = Literal["youtube", "community", "device", "synthetic"]


class EdgeFormat(StrEnum):
    RKLLM = "rkllm"
    RKNN = "rknn"
    ONNX = "onnx"
    GGUF = "gguf"


class PetSpecies(StrEnum):
    CAT = "cat"
    DOG = "dog"
    OTHER = "other"


class BowlType(StrEnum):
    METAL = "metal"
    CERAMIC = "ceramic"
    PLASTIC = "plastic"
    UNKNOWN = "unknown"


class Lighting(StrEnum):
    BRIGHT = "bright"
    DIM = "dim"
    DARK = "dark"
