"""Inputs, pinned generation settings and progress of a training data preparation."""

import secrets
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, model_validator

from regimex.admin_contracts import (
    GenerationSpec,
    RawGenerationPlan,
    ReferenceGenerationSpec,
    SplitPolicy,
)
from regimex.contracts import MAX_STEPS, Identifier, Seed, StrictModel
from regimex.data_catalog import content_hash
from regimex.directions import (
    DEFAULT_ANALYSIS_PROFILE,
    DIRECTION_POLICY,
    AnalysisConfig,
    AutomaticPolicy,
)
from regimex.market_config import GENERATOR_VERSION, GeneratorVersion
from regimex.operations import OperationState

Stage = Literal["development", "final", "train", "validation", "test"]
STAGES: tuple[Stage, ...] = ("development", "final", "train", "validation", "test")


class PreparationSeeds(StrictModel):
    development: Seed
    final: Seed

    @model_validator(mode="after")
    def independent_seeds(self) -> Self:
        if self.development == self.final:
            raise ValueError("development and final generation seeds must differ")
        return self


class PreparationRequest(StrictModel):
    development: Annotated[int, Field(ge=3, le=1000)]
    final: Annotated[int, Field(ge=3, le=1000)]
    steps: Annotated[int, Field(ge=64, le=MAX_STEPS)]
    browser_seeds: PreparationSeeds | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PreparationPlan(StrictModel):
    request: PreparationRequest
    development_seed: Seed
    final_seed: Seed
    generator_version: GeneratorVersion = GENERATOR_VERSION
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE
    policy: AutomaticPolicy = DIRECTION_POLICY
    development_split: SplitPolicy = Field(
        default_factory=lambda: SplitPolicy(validation_fraction=0.25)
    )
    final_split: SplitPolicy = Field(
        default_factory=lambda: SplitPolicy(train_fraction=0.001, validation_fraction=0.001)
    )

    @classmethod
    def create(cls, request: PreparationRequest) -> PreparationPlan:
        if request.browser_seeds:
            return cls(
                request=request,
                development_seed=request.browser_seeds.development,
                final_seed=request.browser_seeds.final,
            )
        development = secrets.randbits(32)
        final = secrets.randbelow(2**32 - 1)
        return cls(
            request=request, development_seed=development, final_seed=final + (final >= development)
        )

    def next_job(
        self, stage: Stage, outputs: dict[Stage, str]
    ) -> GenerationSpec | ReferenceGenerationSpec:
        if stage == "development" or stage == "final":
            development = stage == "development"
            return GenerationSpec(
                dataset_id="dataset-" + uuid4().hex,
                plan=RawGenerationPlan(
                    seed=self.development_seed if development else self.final_seed,
                    experiment_count=self.request.development
                    if development
                    else self.request.final,
                    step_count=self.request.steps,
                ),
                generator_version=self.generator_version,
                splits=self.development_split if development else self.final_split,
            )
        return ReferenceGenerationSpec(
            reference_set_id="reference-" + uuid4().hex,
            dataset_id=outputs["final" if stage == "test" else "development"],
            partition=stage,
            analysis=self.analysis,
            analysis_profile_id=self.analysis_profile_id,
            policy=self.policy,
        )


class PreparationView(StrictModel):
    preparation_id: UUID
    created_at: AwareDatetime
    request: PreparationRequest
    stage: Stage
    operation_id: UUID
    state: OperationState
    failure: str | None
    outputs: dict[Stage, str]
    benchmark_id: str | None
    selection_closed: bool
    workspace_url: str


class PreparationPage(StrictModel):
    items: tuple[PreparationView, ...]
    limit: int
    offset: int


class PreparationRetry(StrictModel):
    operation_id: UUID


def selection_id(owner_id: str, development: str, final: str, test_reference: str) -> str:
    return "selection-" + content_hash([owner_id, development, final, test_reference])[:32]
