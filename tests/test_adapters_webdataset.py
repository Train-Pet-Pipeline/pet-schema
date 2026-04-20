import json
from datetime import datetime

import pytest

from pet_schema.adapters.webdataset import sample_to_wds_dict
from pet_schema.samples import AudioSample, SensorSample, SourceInfo, VisionSample


def _src() -> SourceInfo:
    return SourceInfo(source_type="device", source_id="f1", license=None)


def test_vision_sample_to_wds_dict():
    s = VisionSample(
        sample_id="abc",
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
    out = sample_to_wds_dict(s)
    assert out["__key__"] == "abc"
    assert out[".jpg"] == "local:///tmp/a.jpg"
    # metadata JSON parses and has expected fields
    meta = json.loads(out[".json"])
    assert meta["sample_id"] == "abc"
    assert meta["modality"] == "vision"


def test_audio_sample_uses_wav_extension():
    s = AudioSample(
        sample_id="a1",
        storage_uri="local:///tmp/a.wav",
        captured_at=datetime(2026, 4, 20),
        source=_src(),
        pet_species=None,
        duration_s=1.0,
        sample_rate=16000,
        num_channels=1,
        snr_db=None,
        clip_type=None,
    )
    out = sample_to_wds_dict(s)
    assert out[".wav"] == "local:///tmp/a.wav"
    assert ".jpg" not in out


def test_sensor_sample_uses_json_only():
    s = SensorSample(
        sample_id="s1",
        storage_uri="local:///tmp/r.json",
        captured_at=datetime(2026, 4, 20),
        source=_src(),
        pet_species=None,
        sensor_type="voc",
        readings={"v": 1.0},
        ambient_temp_c=None,
        ambient_humidity=None,
    )
    out = sample_to_wds_dict(s)
    # sensor has no media extension; storage_uri only appears in metadata
    assert set(out.keys()) == {"__key__", ".json"}


def test_wds_dict_is_pure_no_side_effects():
    s = VisionSample(
        sample_id="p",
        storage_uri="local://p",
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
    out1 = sample_to_wds_dict(s)
    out2 = sample_to_wds_dict(s)
    assert out1 == out2
