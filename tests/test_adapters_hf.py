import pytest

datasets = pytest.importorskip("datasets")

from pet_schema.adapters.hf_features import sample_to_hf_features
from pet_schema.samples import AudioSample, SensorSample, VisionSample


def test_vision_sample_maps_to_features_dict():
    feats = sample_to_hf_features(VisionSample)
    assert isinstance(feats, dict)
    # Core BaseSample fields mapped
    assert "sample_id" in feats
    assert "modality" in feats
    # Vision-specific fields mapped
    assert "frame_width" in feats
    assert "frame_height" in feats
    assert "blur_score" in feats


def test_audio_sample_maps_to_features_dict():
    feats = sample_to_hf_features(AudioSample)
    assert "duration_s" in feats
    assert "sample_rate" in feats


def test_sensor_sample_maps_readings_dict():
    feats = sample_to_hf_features(SensorSample)
    assert "sensor_type" in feats
    # readings is dict[str, float] — we just ensure it's represented
    assert "readings" in feats


def test_features_are_datasets_feature_types():
    feats = sample_to_hf_features(VisionSample)
    # sample_id is str → Value("string")
    sid = feats["sample_id"]
    assert isinstance(sid, datasets.Value)
    assert sid.dtype == "string"


def test_unknown_field_type_raises_typeerror():
    # Build a synthetic Pydantic model with an unsupported type
    from pydantic import BaseModel as _BM

    class Weird(_BM):
        data: bytes   # not supported

    with pytest.raises(TypeError):
        sample_to_hf_features(Weird)
