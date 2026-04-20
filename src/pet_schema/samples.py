from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from pet_schema.enums import BowlType, Lighting, Modality, PetSpecies, SourceType


class SourceInfo(BaseModel):
    """Provenance information for a sample."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_id: str
    license: Optional[str]


class BaseSample(BaseModel):
    """Base contract for all data samples in the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    modality: Modality
    storage_uri: str
    captured_at: datetime
    source: SourceInfo
    pet_species: Optional[PetSpecies] = None
    schema_version: str = "2.0.0"


class VisionSample(BaseSample):
    """A video/image sample with visual quality metrics."""

    modality: Literal["vision"] = "vision"
    frame_width: int
    frame_height: int
    lighting: Lighting
    bowl_type: Optional[BowlType] = None
    blur_score: float
    brightness_score: float


class AudioSample(BaseSample):
    """An audio clip sample with acoustic metadata."""

    modality: Literal["audio"] = "audio"
    duration_s: float
    sample_rate: int
    num_channels: int
    snr_db: Optional[float] = None
    clip_type: Optional[Literal["bark", "meow", "purr", "silence", "ambient"]] = None


class SensorSample(BaseSample):
    """A sensor reading sample (e.g. VOC, temperature, weight)."""

    modality: Literal["sensor"] = "sensor"
    sensor_type: str
    readings: dict[str, float]
    ambient_temp_c: Optional[float] = None
    ambient_humidity: Optional[float] = None


Sample = Annotated[
    Union[VisionSample, AudioSample, SensorSample],
    Discriminator("modality"),
]
