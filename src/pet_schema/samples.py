from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator

from pet_schema.enums import BowlType, Lighting, Modality, PetSpecies, SourceType


class SourceInfo(BaseModel):
    """Provenance information for a sample.

    source_type: provenance category (legal/compliance). See SourceType enum.
    ingester: which pet-data ingester class produced this sample (free-form;
        typically matches BaseSource.ingester_name). Optional for samples not
        produced by a registered ingester (e.g., manually constructed for tests).
        Added v3.1.0 for concept separation (provenance vs implementation identity).
    source_id: identifier within the ingester's domain (e.g., YouTube video_id).
    license: SPDX-like license string or free-form for custom terms.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    ingester: str | None = None
    source_id: str
    license: str | None


class BaseSample(BaseModel):
    """Base contract for all data samples in the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    modality: Modality
    storage_uri: str
    captured_at: datetime
    source: SourceInfo
    pet_species: PetSpecies | None = None
    schema_version: str = "2.0.0"


class VisionSample(BaseSample):
    """A video/image sample with visual quality metrics."""

    modality: Literal["vision"] = "vision"
    frame_width: int
    frame_height: int
    lighting: Lighting
    bowl_type: BowlType | None = None
    blur_score: float
    brightness_score: float


class AudioSample(BaseSample):
    """An audio clip sample with acoustic metadata."""

    modality: Literal["audio"] = "audio"
    duration_s: float
    sample_rate: int
    num_channels: int
    snr_db: float | None = None
    clip_type: Literal["bark", "meow", "purr", "silence", "ambient"] | None = None


class SensorSample(BaseSample):
    """A sensor reading sample (e.g. VOC, temperature, weight)."""

    modality: Literal["sensor"] = "sensor"
    sensor_type: str
    readings: dict[str, float]
    ambient_temp_c: float | None = None
    ambient_humidity: float | None = None


Sample = Annotated[
    VisionSample | AudioSample | SensorSample,
    Discriminator("modality"),
]
