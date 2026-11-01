"""Observation-only exports with explicit source, model-time, and missingness semantics.

V2 records declarations about a completed source snapshot; it cannot authenticate
those declarations. Import ownership and immutable run identity must also be
enforced by the application. Its digest covers canonical content, not JSON layout.
"""

import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from regimex.contracts import (
    MAX_INPUT_BYTES,
    MAX_STEPS,
    Identifier,
    Instrument,
    Observation,
    Positive,
    Producer,
    StrictModel,
)

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Explanation = Annotated[str, Field(min_length=1, max_length=4096, pattern=r"\S")]
DecimalPlaces = Annotated[int, Field(ge=0, le=18)]
Coverage = Literal["complete", "partial", "unavailable"]
PriceState = Literal["observed", "carried", "no_execution", "unknown"]
QuoteState = Literal["observed", "empty", "unknown"]


class SourceIdentity(StrictModel):
    source_id: Identifier
    source_version: Identifier
    configuration_id: Identifier
    run_id: Identifier
    snapshot_id: Identifier
    state_revision: Identifier
    run_status: Literal["completed"]
    consistency: Literal["atomic_completed_snapshot"]


class ModelTimeGrid(StrictModel):
    """Bin k spans [origin + k * width, min(origin + (k + 1) * width, horizon)).

    Horizon is an absolute coordinate in the declared model-time unit. Every bin
    is exported, including leading, internal, and trailing inactivity. A partial
    final bin is allowed only under the explicit include_partial_bin policy.
    """

    origin: float
    unit: Identifier
    bin_width: Positive
    completed_horizon: float
    step_count: Annotated[int, Field(ge=1, le=MAX_STEPS)]
    interval_bounds: Literal["[start,stop)"]
    terminal_bin: Literal["full_bins_only", "include_partial_bin"]

    @model_validator(mode="after")
    def executed_horizon(self) -> Self:
        # Use exact decimal values without binary rounding or the caller's
        # Decimal context silently discarding a small origin/horizon difference.
        duration = Fraction(str(self.completed_horizon)) - Fraction(str(self.origin))
        width = Fraction(str(self.bin_width))
        stop = width * self.step_count
        if duration <= 0:
            raise ValueError("completed horizon must be after the model-time origin")
        if self.terminal_bin == "full_bins_only" and duration != stop:
            raise ValueError("full bins must cover the entire completed horizon exactly")
        if self.terminal_bin == "include_partial_bin" and not stop - width < duration <= stop:
            raise ValueError("terminal bin must contain the completed horizon without truncation")
        return self


class InstrumentUnits(StrictModel):
    """Asset identifiers on the instrument determine the actual unit names."""

    price_unit: Literal["quote_asset_per_base_asset"]
    volume_unit: Literal["base_asset"]
    price_decimal_places: DecimalPlaces
    volume_decimal_places: DecimalPlaces


def _check_precision(value: float | None, places: int) -> None:
    if value is not None:
        # Inspect digits directly: Decimal.normalize() rounds under the caller's
        # context and could erase the very digits this boundary must reject.
        decimal = Decimal(str(value)).as_tuple()
        decimal_places = max(0, -int(decimal.exponent))
        for digit in reversed(decimal.digits):
            if digit != 0 or decimal_places == 0:
                break
            decimal_places -= 1
        if decimal_places > places:
            raise ValueError("observation exceeds its instrument's declared numeric precision")


class ExportInstrument(Instrument):
    units: InstrumentUnits

    @model_validator(mode="after")
    def initial_price_precision(self) -> Self:
        _check_precision(self.initial_price, self.units.price_decimal_places)
        return self


class ExportObservation(Observation):
    """Null quote sides require explicit unknown versus observed-empty states.

    A carried price is an available price, but is not evidence of an execution.
    Unknown aggregate values remain distinct from observed zeros.
    """

    price_state: PriceState
    bid_state: QuoteState
    ask_state: QuoteState

    @model_validator(mode="after")
    def explicit_missingness(self) -> Self:
        price_present = self.price_state in ("observed", "carried")
        if price_present != (self.last_price is not None):
            raise ValueError("last price must agree with its explicit price state")
        for execution_total in (self.trade_count, self.trade_volume):
            if execution_total is None:
                continue
            if self.price_state == "observed" and execution_total == 0:
                raise ValueError("an observed execution price requires an execution")
            if self.price_state in ("carried", "no_execution") and execution_total != 0:
                raise ValueError("a carried or absent execution price requires no executions")
        for state, price in ((self.bid_state, self.best_bid), (self.ask_state, self.best_ask)):
            if (state == "observed") != (price is not None):
                raise ValueError("quote value must agree with its explicit quote-side state")
        return self


class HistoricalQuoteCapability(StrictModel):
    """Historical quotes refer to bin closes after all events in that bin.

    A present-day book is insufficient. Reconstruction requires an identified,
    complete event history and a documented replay policy; its hash is provenance
    rather than proof that the source actually retained every required event.
    """

    mode: Literal["unavailable", "retained_snapshots", "reconstructed_event_history"]
    coverage: Coverage
    provenance: Explanation
    complete_event_history_sha256: SHA256 | None
    reconstruction_policy: Identifier | None

    @model_validator(mode="after")
    def reconstructible_history(self) -> Self:
        if (self.mode == "unavailable") != (self.coverage == "unavailable"):
            raise ValueError("unavailable quote history must declare unavailable coverage")
        reconstruction = self.mode == "reconstructed_event_history"
        if reconstruction != (self.complete_event_history_sha256 is not None):
            raise ValueError("reconstructed quotes require the complete event-history digest")
        if reconstruction != (self.reconstruction_policy is not None):
            raise ValueError("reconstructed quotes require a documented reconstruction policy")
        return self


class SourceCapabilities(StrictModel):
    """Counts and volumes are per-bin totals, never resting order-book depth.

    Each partial execution counts once; trade volume sums its base quantity once.
    Accepted-order volumes sum submitted quantities, including immediate fills
    and excluding rejected submissions. Missing totals are null, inactivity zero.
    """

    execution_prices: Coverage
    trade_aggregates: Coverage
    accepted_order_aggregates: Coverage
    historical_quotes: HistoricalQuoteCapability
    price_policy: Literal["initial_then_carry", "execution_in_bin_only"]
    trade_count_policy: Literal["one_per_partial_execution"]
    order_count_policy: Literal["accepted_submissions_including_immediate_fills"]
    rejected_order_policy: Literal["excluded"]


class ExportPayloadV2(StrictModel):
    schema_version: Literal["2.1"]
    source: SourceIdentity
    producer: Producer
    time_grid: ModelTimeGrid
    capabilities: SourceCapabilities
    instruments: Annotated[tuple[ExportInstrument, ...], Field(min_length=1, max_length=2)]
    observations: Annotated[
        tuple[ExportObservation, ...], Field(min_length=1, max_length=2 * MAX_STEPS)
    ]

    @model_validator(mode="after")
    def complete_observation_grid(self) -> Self:
        instruments = {instrument.instrument_id: instrument for instrument in self.instruments}
        if len(instruments) != len(self.instruments):
            raise ValueError("instrument identifiers must be unique")
        if len({instrument.quote_asset for instrument in self.instruments}) != 1:
            raise ValueError("instruments must share a quote asset")
        if len(self.observations) != self.time_grid.step_count * len(instruments):
            raise ValueError("observations must cover the entire completed instrument/time grid")
        previous_step = dict.fromkeys(instruments, -1)
        previous_price: dict[str, float | None] = {
            identifier: instrument.initial_price for identifier, instrument in instruments.items()
        }
        for row in self.observations:
            identifier = row.instrument_id
            if identifier not in instruments or row.step >= self.time_grid.step_count:
                raise ValueError("observation references an unknown instrument or step")
            if row.step != previous_step[identifier] + 1:
                raise ValueError("observation steps must be ordered, contiguous, and unique")
            previous_step[identifier] = row.step
            self._validate_capabilities(row)
            units = instruments[identifier].units
            for price in (row.last_price, row.best_bid, row.best_ask):
                _check_precision(price, units.price_decimal_places)
            for volume in (row.trade_volume, row.buy_order_volume, row.sell_order_volume):
                _check_precision(volume, units.volume_decimal_places)
            known_price = previous_price[identifier]
            if (
                row.price_state == "carried"
                and known_price is not None
                and row.last_price != known_price
            ):
                raise ValueError(
                    "carried price differs from the last known execution/initial price"
                )
            # Unknown observations may hide executions. Either known zero total
            # proves inactivity even when the other total and price are lost.
            known_inactive = row.trade_count == 0 or row.trade_volume == 0
            if row.last_price is not None or not known_inactive:
                previous_price[identifier] = row.last_price
        return self

    def _validate_capabilities(self, row: ExportObservation) -> None:
        capability = self.capabilities
        if capability.execution_prices == "unavailable" and row.price_state != "unknown":
            raise ValueError("unavailable execution prices must have unknown price states")
        if capability.execution_prices == "complete" and row.price_state == "unknown":
            raise ValueError("complete execution prices cannot contain unknown prices")
        if capability.price_policy == "execution_in_bin_only" and row.price_state == "carried":
            raise ValueError("execution-in-bin prices cannot be carried forward")
        if capability.price_policy == "initial_then_carry" and row.price_state == "no_execution":
            raise ValueError("initial/carry prices require a price or an explicit unknown state")
        for coverage, values in (
            (capability.trade_aggregates, (row.trade_count, row.trade_volume)),
            (
                capability.accepted_order_aggregates,
                (
                    row.buy_order_count,
                    row.buy_order_volume,
                    row.sell_order_count,
                    row.sell_order_volume,
                ),
            ),
        ):
            if coverage == "unavailable" and any(value is not None for value in values):
                raise ValueError("unavailable aggregate capabilities must contain only nulls")
            if coverage == "complete" and any(value is None for value in values):
                raise ValueError("complete aggregate capabilities cannot contain nulls")
        for state in (row.bid_state, row.ask_state):
            coverage = capability.historical_quotes.coverage
            if coverage == "unavailable" and state != "unknown":
                raise ValueError(
                    "unavailable historical quotes cannot claim an observed/empty side"
                )
            if coverage == "complete" and state == "unknown":
                raise ValueError("complete historical quotes cannot contain unknown sides")


def _content_digest(payload: ExportPayloadV2) -> str:
    content = payload.model_dump(mode="json", exclude={"content_sha256"})
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


class ExportV2(ExportPayloadV2):
    content_sha256: SHA256 = Field(
        description=(
            "SHA-256 of validated JSON-mode content without content_sha256, encoded as UTF-8 "
            "using json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True, "
            "allow_nan=False), with no trailing newline. This is schema-specific canonicalization."
        )
    )

    @model_validator(mode="after")
    def immutable_content(self) -> Self:
        if self.content_sha256 != _content_digest(self):
            raise ValueError("export content SHA-256 mismatch")
        return self


def seal_export(payload: ExportPayloadV2) -> ExportV2:
    """Create a checked immutable snapshot; assigning unique identities is external."""
    return ExportV2(
        **payload.model_dump(exclude={"content_sha256"}), content_sha256=_content_digest(payload)
    )


def load_export_v2(path: Path) -> ExportV2:
    with path.open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    return parse_export_v2(data)


def parse_export_v2(data: bytes) -> ExportV2:
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("export exceeds the 16 MiB input size limit")
    return ExportV2.model_validate_json(data)
