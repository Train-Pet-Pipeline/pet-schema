"""Map pet-schema Sample subclasses → HuggingFace `datasets.Features` dict."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel


def _field_type_to_feature(tp: Any) -> Any:
    """Map a Python/Pydantic field annotation to a datasets.Features leaf."""
    from datasets import Sequence, Value

    origin = get_origin(tp)

    # Optional[X] / X | None — unwrap to X
    if origin is Union or origin is UnionType:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _field_type_to_feature(non_none[0])
        raise TypeError(f"unsupported union type: {tp!r}")

    # Literal["x", ...] → string (discriminator fields)
    if origin is Literal:
        return Value("string")

    # Enums → string
    if isinstance(tp, type) and issubclass(tp, Enum):
        return Value("string")

    if tp is str:
        return Value("string")
    if tp is int:
        return Value("int64")
    if tp is float:
        return Value("float64")
    if tp is bool:
        return Value("bool")
    if tp is datetime:
        return Value("timestamp[ns]")

    # list[X] / List[X]
    if origin in (list,):
        (inner,) = get_args(tp)
        return Sequence(_field_type_to_feature(inner))

    # dict[str, X] — represent as Sequence of {key, value} records (HF-friendly)
    if origin in (dict,):
        _, v = get_args(tp)
        return Sequence({"key": Value("string"), "value": _field_type_to_feature(v)})

    # Nested Pydantic model — recurse into its fields
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return sample_to_hf_features(tp)

    raise TypeError(f"unsupported Pydantic field type for HF mapping: {tp!r}")


def sample_to_hf_features(sample_cls: type[BaseModel]) -> dict[str, Any]:
    """Build a datasets.Features-compatible dict for a Pydantic model class."""
    out: dict[str, Any] = {}
    for name, info in sample_cls.model_fields.items():
        out[name] = _field_type_to_feature(info.annotation)
    return out
