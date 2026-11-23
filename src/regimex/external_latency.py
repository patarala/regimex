"""Import recorded wall-clock medians for the selected model bundle."""

from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, model_validator

from regimex.contracts import MAX_STEPS, StrictModel
from regimex.external_evaluations import read_evidence_file
from regimex.operations import Sha256

Milliseconds = Annotated[float, Field(ge=0)]


class LatencyAtLength(StrictModel):
    steps: Annotated[int, Field(ge=1, le=MAX_STEPS)]
    predict_ms: Milliseconds | None = None
    predict_decode_ms: Milliseconds


class ExternalLatencyEvidence(StrictModel):
    manifest_sha256: Sha256
    report_sha256: Sha256
    timings: Annotated[tuple[LatencyAtLength, ...], Field(min_length=1, max_length=64)]
    threads: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def increasing_lengths(self) -> Self:
        lengths = [item.steps for item in self.timings]
        if lengths != sorted(set(lengths)):
            raise ValueError("The report must have one timing per measured series length")
        return self


def read_latency_evidence(
    path: Path, *, manifest_sha256: str, selection_file: Path | None = None
) -> ExternalLatencyEvidence:
    report, digest = read_evidence_file(path)
    timings = []
    try:
        if "selection_manifest" in report:
            if report["selection_manifest"] != manifest_sha256 or selection_file is None:
                raise ValueError("Forest timings require the matching bundle and --selection-file")
            selection, _ = read_evidence_file(selection_file)
            if selection["bundle_sha256"] != manifest_sha256:
                raise ValueError("Frozen selection belongs to another bundle")
            candidate = selection["challenger"]
            for length, panel in report["results"].items():
                median = panel["median"][candidate]
                timings.append(
                    LatencyAtLength(
                        steps=int(length),
                        predict_ms=median["predict_wall_ms"],
                        predict_decode_ms=median["predict_decode_wall_ms"],
                    )
                )
        else:
            if report.get("manifest_sha256") != manifest_sha256:
                raise ValueError("Inference timings belong to another bundle")
            for length, medians in report["median_predict_decode_ms"].items():
                timings.append(
                    LatencyAtLength(steps=int(length), predict_decode_ms=medians["blend"])
                )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete latency report for the selected model") from exc
    return ExternalLatencyEvidence(
        manifest_sha256=manifest_sha256,
        report_sha256=digest,
        timings=tuple(sorted(timings, key=lambda item: item.steps)),
        threads=report.get("threads"),
    )
