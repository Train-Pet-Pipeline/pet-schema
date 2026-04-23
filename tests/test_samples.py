from datetime import datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from pet_schema.samples import (
    AudioSample,
    BaseSample,
    Sample,
    SensorSample,
    SourceInfo,
    VisionSample,
)


def _src():
    return SourceInfo(source_type="device", source_id="feeder-001", license=None)


def test_vision_sample_roundtrip():
    s = VisionSample(
        sample_id="abc",
        storage_uri="local:///tmp/a.jpg",
        captured_at=datetime(2026, 4, 20),
        source=_src(),
        pet_species=None,
        frame_width=1280,
        frame_height=720,
        lighting="bright",
        bowl_type="metal",
        blur_score=0.1,
        brightness_score=0.7,
    )
    as_dict = s.model_dump(mode="json")
    assert as_dict["modality"] == "vision"
    assert VisionSample.model_validate(as_dict) == s


def test_sample_union_discriminator():
    adapter = TypeAdapter(Sample)
    vision_dict = {
        "sample_id": "v1",
        "modality": "vision",
        "storage_uri": "local:///tmp/a.jpg",
        "captured_at": "2026-04-20T00:00:00",
        "source": {"source_type": "device", "source_id": "feeder-001", "license": None},
        "pet_species": None,
        "frame_width": 1280,
        "frame_height": 720,
        "lighting": "bright",
        "bowl_type": "metal",
        "blur_score": 0.1,
        "brightness_score": 0.7,
    }
    audio_dict = {
        "sample_id": "a1",
        "modality": "audio",
        "storage_uri": "local:///tmp/a.wav",
        "captured_at": "2026-04-20T00:00:00",
        "source": {"source_type": "device", "source_id": "feeder-001", "license": None},
        "pet_species": None,
        "duration_s": 2.3,
        "sample_rate": 16000,
        "num_channels": 1,
        "snr_db": None,
        "clip_type": "bark",
    }
    assert isinstance(adapter.validate_python(vision_dict), VisionSample)
    assert isinstance(adapter.validate_python(audio_dict), AudioSample)


def test_vision_sample_rejects_wrong_modality_literal():
    with pytest.raises(ValidationError):
        VisionSample(
            sample_id="x",
            modality="audio",  # mismatch
            storage_uri="local:///tmp/a.jpg",
            captured_at=datetime(2026, 4, 20),
            source=_src(),
            pet_species=None,
            frame_width=1,
            frame_height=1,
            lighting="bright",
            bowl_type=None,
            blur_score=0.0,
            brightness_score=0.5,
        )


def test_sample_is_frozen():
    s = VisionSample(
        sample_id="abc",
        storage_uri="local://x",
        captured_at=datetime(2026, 4, 20),
        source=_src(),
        pet_species=None,
        frame_width=1,
        frame_height=1,
        lighting="bright",
        bowl_type=None,
        blur_score=0.0,
        brightness_score=0.5,
    )
    with pytest.raises(ValidationError):
        s.sample_id = "mutated"


def test_source_info_extra_forbid():
    with pytest.raises(ValidationError):
        SourceInfo(
            source_type="device",
            source_id="f1",
            license=None,
            surprise="field",  # type: ignore[call-arg]
        )


def test_audio_sample_roundtrip():
    a = AudioSample(
        sample_id="a1",
        storage_uri="local:///tmp/a.wav",
        captured_at=datetime(2026, 4, 20),
        source=_src(),
        pet_species=None,
        duration_s=1.5,
        sample_rate=16000,
        num_channels=1,
        snr_db=None,
        clip_type="meow",
    )
    assert AudioSample.model_validate(a.model_dump(mode="json")) == a


def test_sensor_sample_readings():
    s = SensorSample(
        sample_id="s1",
        storage_uri="local:///tmp/r.json",
        captured_at=datetime(2026, 4, 20),
        source=_src(),
        pet_species=None,
        sensor_type="chem_voc",
        readings={"voc_ppb": 120.0},
        ambient_temp_c=22.5,
        ambient_humidity=45.0,
    )
    assert s.readings["voc_ppb"] == 120.0


def test_base_sample_schema_version_default():
    # spec §3.6 — per-record schema_version stamped on every sample
    s = SensorSample(
        sample_id="s1",
        storage_uri="local:///tmp/r.json",
        captured_at=datetime(2026, 4, 20),
        source=_src(),
        pet_species=None,
        sensor_type="chem_voc",
        readings={},
        ambient_temp_c=None,
        ambient_humidity=None,
    )
    assert s.schema_version == "2.0.0"


def test_base_sample_is_abstract_like_but_ok_to_construct():
    # BaseSample itself is a valid Pydantic model; ensure at least one field exists via
    # introspection
    # (protects against accidental deletion of the class body later)
    assert "modality" in BaseSample.model_fields


def test_source_info_ingester_optional_and_carries_string():
    """SourceInfo.ingester v3.1.0: optional field, defaults None, accepts string, extra still forbidden."""
    # backward compat: constructible without ingester
    si_no_ingester = SourceInfo(source_type="device", source_id="feeder-001", license=None)
    assert si_no_ingester.ingester is None

    # new: constructible with ingester carrying an ingester identity string
    si_with_ingester = SourceInfo(
        source_type="academic_dataset",
        source_id="oxford-003317",
        license="CC-BY-4.0",
        ingester="oxford_pet",
    )
    assert si_with_ingester.ingester == "oxford_pet"

    # extra="forbid" still rejects unknown fields
    with pytest.raises(ValidationError):
        SourceInfo(
            source_type="device",
            source_id="f1",
            license=None,
            unknown_extra="bad",  # type: ignore[call-arg]
        )
