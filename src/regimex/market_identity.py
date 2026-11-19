"""Identify related native simulations when their exports enter a dataset."""

from uuid import NAMESPACE_URL, uuid5

from regimex.data_catalog import SplitGroup, content_hash
from regimex.exports import ExportV2


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


def imported_seed_group(export: ExportV2, declared: SplitGroup | None) -> SplitGroup | None:
    source = export.source
    if source.source_id != "regimex-exchange":
        return declared
    seed = declared.seed if declared is not None else None
    if source.run_id.startswith("sample:"):
        parts = source.run_id.split(":")
        if len(parts) != 3 or not parts[1].isascii() or not parts[1].isdecimal():
            raise ValueError("Invalid native sampled run identity.")
        seed = int(parts[1])
    if seed is None:
        raise ValueError("This native export requires its original seed group for dataset import.")
    group = market_seed_group(seed)
    if declared is not None and declared != group:
        raise ValueError("The declared split group differs from the native market seed group.")
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
