"""Identify related native simulations when their exports enter a dataset."""

from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ConfigDict

from regimex.contracts import Identifier, Seed, StrictModel
from regimex.exports import ExportV2
from regimex.storage import content_hash


class SplitGroup(StrictModel):
    """Source-controlled dependence group, separate from prediction observations.

    A seed is meaningful only within its generator namespace. The namespace must
    remain stable across dataset versions and source aliases. Missing seed means
    an external run, not an observed seed of zero.
    """

    model_config = ConfigDict(revalidate_instances="always")
    namespace: Identifier
    group_id: Identifier
    seed: Seed | None = None


def market_seed_group(seed: int) -> SplitGroup:
    # Asset names changed in generator 4.0. The same stochastic seeds still belong together.
    return SplitGroup(namespace="regimex-exchange-seed/3.0", group_id=f"seed:{seed}", seed=seed)


def sampled_configuration(version: str, seed: int, steps: int, instruments: int) -> str:
    return content_hash(
        {"generator_version": version, "seed": seed, "steps": steps, "instruments": instruments}
    )


def sampled_run_id(seed: int, configuration_id: str) -> str:
    identifier = uuid5(NAMESPACE_URL, f"regimex:single-market:{configuration_id}")
    return f"sample:{seed}:{identifier}"


def native_seed_group(run_id: str, declared: SplitGroup | None) -> SplitGroup:
    seed = declared.seed if declared is not None else None
    if run_id.startswith("sample:"):
        try:
            _, seed_text, suffix = run_id.split(":")
            seed = int(seed_text)
            identifier = UUID(suffix)
        except ValueError as exc:
            raise ValueError("Invalid native sampled run identity.") from exc
        if identifier.version != 5 or run_id != f"sample:{seed}:{identifier}":
            raise ValueError("Invalid native sampled run identity.")
    if seed is None:
        raise ValueError("This native export requires its original seed group for dataset import.")
    group = market_seed_group(seed)
    if declared is not None and declared != group:
        raise ValueError("The declared split group differs from the native market seed group.")
    return group


def imported_seed_group(export: ExportV2, declared: SplitGroup | None) -> SplitGroup | None:
    source = export.source
    if source.source_id != "regimex-exchange":
        return declared
    group = native_seed_group(source.run_id, declared)
    seed = group.seed
    assert seed is not None
    previous_id = str(uuid5(NAMESPACE_URL, f"regimex:single-market:{source.configuration_id}"))
    if source.run_id.startswith("sample:") or source.run_id == previous_id:
        expected = sampled_configuration(
            source.source_version, seed, export.time_grid.step_count, len(export.instruments)
        )
        if expected != source.configuration_id or source.run_id not in (
            previous_id,
            sampled_run_id(seed, expected),
        ):
            raise ValueError("The seed differs from the sampled market configuration.")
    return group
