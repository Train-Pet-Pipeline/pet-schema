from typing import get_args

from pet_schema.enums import BowlType, EdgeFormat, Lighting, Modality, PetSpecies


def test_edge_format_values():
    assert EdgeFormat.RKLLM.value == "rkllm"
    assert {m.value for m in EdgeFormat} == {"rkllm", "rknn", "onnx", "gguf"}


def test_modality_literal_members():
    assert set(get_args(Modality)) == {"vision", "audio", "sensor", "multimodal"}


def test_petspecies_members_cover_cat_dog():
    assert "cat" in {s.value for s in PetSpecies}
    assert "dog" in {s.value for s in PetSpecies}


def test_bowl_and_lighting_members():
    assert {b.value for b in BowlType} >= {"metal", "ceramic", "plastic", "unknown"}
    assert {lit.value for lit in Lighting} >= {"bright", "dim", "dark"}


def test_source_type_includes_academic_and_commercial():
    """SourceType v3.1.0: two new provenance literals are valid in SourceInfo."""
    from typing import get_args

    from pet_schema.enums import SourceType
    from pet_schema.samples import SourceInfo

    all_values = set(get_args(SourceType))
    assert "academic_dataset" in all_values, "academic_dataset must be a SourceType literal"
    assert "commercial_licensed" in all_values, "commercial_licensed must be a SourceType literal"

    # original 4 literals still valid
    for t in ("youtube", "community", "device", "synthetic"):
        assert t in all_values, f"existing literal {t!r} must still be present"

    # new literals can be used in SourceInfo
    si_academic = SourceInfo(source_type="academic_dataset", source_id="oxford-pet", license="CC-BY-4.0")
    assert si_academic.source_type == "academic_dataset"

    si_commercial = SourceInfo(source_type="commercial_licensed", source_id="dataset-xyz", license="commercial-v1")
    assert si_commercial.source_type == "commercial_licensed"
