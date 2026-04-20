"""pet-schema → webdataset ShardWriter dict adapter."""
from __future__ import annotations

from pet_schema.samples import AudioSample, BaseSample, VisionSample


def sample_to_wds_dict(sample: BaseSample) -> dict[str, str]:
    """Convert a BaseSample instance to a webdataset ShardWriter-compatible dict."""
    out: dict[str, str] = {
        "__key__": sample.sample_id,
        ".json": sample.model_dump_json(),
    }
    if isinstance(sample, VisionSample):
        out[".jpg"] = sample.storage_uri
    elif isinstance(sample, AudioSample):
        out[".wav"] = sample.storage_uri
    return out
