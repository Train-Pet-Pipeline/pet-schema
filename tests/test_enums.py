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
    assert {l.value for l in Lighting} >= {"bright", "dim", "dark"}
