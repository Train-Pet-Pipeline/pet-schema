"""ExperimentRecipe contract: stage graph, ablation axes, DAG validation.

Defines the schema for experiment orchestration recipes, including
cross-stage dependency DAG construction and cycle detection.
"""
from __future__ import annotations

from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, model_validator


class ArtifactRef(BaseModel):
    """Stage-to-stage / stage-to-dataset references.

    Four ref_types:
    - dataset              -> DATASETS registry key
    - model_card           -> existing ModelCard.id
    - dvc_path             -> arbitrary DVC-tracked path
    - recipe_stage_output  -> name of upstream stage in same recipe
    """

    model_config = ConfigDict(extra="forbid")

    ref_type: Literal["dataset", "model_card", "dvc_path", "recipe_stage_output"]
    ref_value: str


class RecipeStage(BaseModel):
    """A single executable stage in an ExperimentRecipe DAG.

    Five registries may become standalone stages: trainers, evaluators,
    converters, datasets (calibration/subset stages), and ota (deploy stages).
    Passive registries (metrics, storage) are dependency sources only.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    component_registry: Literal["trainers", "evaluators", "converters", "datasets", "ota"]
    component_type: str
    inputs: dict[str, ArtifactRef]
    config_path: str
    depends_on: list[str]
    condition: str | None = None
    on_failure: Literal["stop", "continue", "abort"] = "stop"


class AblationAxis(BaseModel):
    """Declares a sweep axis over a recipe stage's config parameter.

    The ``stage`` field must reference a RecipeStage.name present in the
    enclosing ExperimentRecipe (cross-validated at recipe construction time).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    stage: str
    hydra_path: str
    values: list[str | int | float | bool]
    link_to: str | None = None


class ExperimentRecipe(BaseModel):
    """Top-level experiment orchestration contract.

    Validates that:
    - Every AblationAxis.stage references a known RecipeStage.name.
    - The stage dependency graph is a DAG (no cycles).
    """

    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    description: str
    scope: Literal["single_repo", "cross_repo"]
    owner_repo: str | None = None
    schema_version: str

    stages: list[RecipeStage]
    variations: list[AblationAxis]
    produces: list[str]

    default_storage: str
    required_plugins: list[str]

    @model_validator(mode="after")
    def _cross_validate(self) -> ExperimentRecipe:
        """Cross-validate ablation axes and eagerly detect DAG cycles."""
        stage_names = {s.name for s in self.stages}
        for axis in self.variations:
            if axis.stage not in stage_names:
                raise ValueError(
                    f"AblationAxis {axis.name!r}: stage {axis.stage!r} "
                    f"not found in recipe stages {sorted(stage_names)}"
                )
        # Eagerly build DAG so cycles surface at construction time.
        self.to_dag()
        return self

    def to_dag(self) -> nx.DiGraph:
        """Build and return a directed acyclic graph of the recipe stages.

        Returns:
            nx.DiGraph with one node per stage and one edge per dependency.

        Raises:
            ValueError: if the dependency graph contains a cycle.
        """
        g: nx.DiGraph = nx.DiGraph()
        for stage in self.stages:
            g.add_node(stage.name)
        for stage in self.stages:
            for dep in stage.depends_on:
                g.add_edge(dep, stage.name)
        try:
            cycle = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            return g
        raise ValueError(f"recipe DAG has a cycle: {cycle}")
