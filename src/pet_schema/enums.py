from __future__ import annotations

from enum import StrEnum
from typing import Literal

Modality = Literal["vision", "audio", "sensor", "multimodal"]

SourceType = Literal[
    "youtube",              # public web scraping (fair use / ToS)
    "community",            # user-uploaded with consent
    "device",               # first-party hardware captures
    "synthetic",            # AI-generated
    "academic_dataset",     # research datasets (MIT/Apache/CC licensed)  — NEW v3.1.0
    "commercial_licensed",  # paid / commercially licensed datasets — NEW v3.1.0
]


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
