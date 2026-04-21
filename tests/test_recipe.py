import networkx as nx
import pytest
from pydantic import ValidationError

from pet_schema.recipe import (
    AblationAxis,
    ArtifactRef,
    ExperimentRecipe,
    RecipeStage,
)


def _stage(name: str, depends_on: list[str] | None = None) -> RecipeStage:
    return RecipeStage(
        name=name,
        component_registry="trainers",
        component_type="pet_train.vlm_sft",
        inputs={},
        config_path=f"trainer/{name}",
        depends_on=depends_on or [],
    )


def _recipe(**overrides) -> ExperimentRecipe:
    base = dict(
        recipe_id="r1",
        description="d",
        scope="single_repo",
        owner_repo="pet-train",
        schema_version="2.0.0",
        stages=[_stage("only")],
        variations=[],
        produces=[],
        default_storage="local",
        required_plugins=[],
    )
    base.update(overrides)
    return ExperimentRecipe(**base)


def test_artifact_ref_accepts_known_ref_types():
    for rt in ["dataset", "model_card", "dvc_path", "recipe_stage_output"]:
        assert ArtifactRef(ref_type=rt, ref_value="x").ref_type == rt


def test_artifact_ref_rejects_unknown_ref_type():
    with pytest.raises(ValidationError):
        ArtifactRef(ref_type="random", ref_value="x")


def test_recipe_stage_rejects_passive_registry():
    for bad in ["metrics", "storage"]:
        with pytest.raises(ValidationError):
            RecipeStage(
                name="x",
                component_registry=bad,  # type: ignore[arg-type]
                component_type="ds",
                inputs={},
                config_path="p",
                depends_on=[],
            )


def test_recipe_stage_accepts_active_registries():
    for good in ["trainers", "evaluators", "converters"]:
        s = RecipeStage(
            name="x",
            component_registry=good,  # type: ignore[arg-type]
            component_type="c",
            inputs={},
            config_path="p",
            depends_on=[],
        )
        assert s.component_registry == good


def test_recipe_stage_accepts_datasets_registry():
    stage = RecipeStage(
        name="calibrate",
        component_registry="datasets",
        component_type="vision_calibration_subset",
        inputs={},
        config_path="configs/smoke/tiny_calibration.yaml",
        depends_on=["train"],
    )
    assert stage.component_registry == "datasets"


def test_recipe_stage_accepts_ota_registry():
    stage = RecipeStage(
        name="deploy",
        component_registry="ota",
        component_type="local_backend",
        inputs={},
        config_path="configs/smoke/tiny_deploy.yaml",
        depends_on=["train"],
    )
    assert stage.component_registry == "ota"


def test_recipe_stage_rejects_unknown_registry():
    with pytest.raises(ValidationError):
        RecipeStage(
            name="x",
            component_registry="bogus",  # type: ignore[arg-type]
            component_type="t",
            inputs={},
            config_path="p",
            depends_on=[],
        )


def test_ablation_axis_values_accept_mixed_types():
    ax = AblationAxis(
        name="lr",
        stage="only",
        hydra_path="trainer.args.lr",
        values=["a", 1, 2.5, True],
    )
    assert ax.values == ["a", 1, 2.5, True]


def test_to_dag_has_depends_on_edge():
    r = ExperimentRecipe(
        recipe_id="r1",
        description="d",
        scope="single_repo",
        owner_repo="pet-train",
        schema_version="2.0.0",
        stages=[_stage("sft"), _stage("dpo", ["sft"])],
        variations=[],
        produces=["m1"],
        default_storage="local",
        required_plugins=["pet_train"],
    )
    g: nx.DiGraph = r.to_dag()
    assert g.has_edge("sft", "dpo")
    assert list(nx.topological_sort(g)) == ["sft", "dpo"]


def test_to_dag_raises_on_cycle():
    with pytest.raises((ValueError, ValidationError), match=r"(?i)cycle"):
        ExperimentRecipe(
            recipe_id="r1",
            description="d",
            scope="single_repo",
            owner_repo="pet-train",
            schema_version="2.0.0",
            stages=[_stage("a", ["b"]), _stage("b", ["a"])],
            variations=[],
            produces=[],
            default_storage="local",
            required_plugins=[],
        )


def test_ablation_axis_must_reference_known_stage():
    with pytest.raises(ValidationError):
        ExperimentRecipe(
            recipe_id="r1",
            description="d",
            scope="single_repo",
            owner_repo="pet-train",
            schema_version="2.0.0",
            stages=[_stage("sft")],
            variations=[
                AblationAxis(
                    name="x",
                    stage="unknown",
                    hydra_path="a.b",
                    values=[1, 2],
                )
            ],
            produces=[],
            default_storage="local",
            required_plugins=[],
        )


def test_scope_accepts_both_values():
    for scope in ["single_repo", "cross_repo"]:
        r = _recipe(scope=scope)
        assert r.scope == scope


def test_scope_rejects_unknown():
    with pytest.raises(ValidationError):
        _recipe(scope="global")


def test_recipe_stage_default_on_failure():
    s = _stage("x")
    assert s.on_failure == "stop"


def test_depends_on_edge_with_three_stages_topo_sort_correct():
    r = ExperimentRecipe(
        recipe_id="r1",
        description="d",
        scope="single_repo",
        owner_repo="pet-train",
        schema_version="2.0.0",
        stages=[
            _stage("sft"),
            _stage("eval", ["sft"]),
            _stage("quant", ["eval"]),
        ],
        variations=[],
        produces=[],
        default_storage="local",
        required_plugins=[],
    )
    order = list(nx.topological_sort(r.to_dag()))
    assert order.index("sft") < order.index("eval") < order.index("quant")
