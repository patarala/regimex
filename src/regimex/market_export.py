"""Produce sealed observation exports from the native market simulator."""

from uuid import NAMESPACE_URL, uuid5

from regimex.contracts import Producer
from regimex.data_catalog import content_hash
from regimex.exports import (
    ExportInstrument,
    ExportObservation,
    ExportPayloadV2,
    ExportV2,
    HistoricalQuoteCapability,
    InstrumentUnits,
    ModelTimeGrid,
    SourceCapabilities,
    SourceIdentity,
    decimal_places,
    seal_export,
)
from regimex.market import MarketAudit, simulate
from regimex.market_config import MarketConfig
from regimex.market_sampling import GENERATOR_VERSION, sample_market


def simulate_export(
    market: MarketConfig, *, seed: int, steps: int, configuration_id: str, run_id: str
) -> tuple[ExportV2, MarketAudit]:
    run = simulate(market, seed, steps)
    export = seal_export(
        ExportPayloadV2(
            schema_version="2.1",
            source=SourceIdentity(
                source_id="regimex-exchange",
                source_version=GENERATOR_VERSION,
                configuration_id=configuration_id,
                run_id=run_id,
                snapshot_id=run_id,
                state_revision="completed-1",
                run_status="completed",
                consistency="atomic_completed_snapshot",
            ),
            producer=Producer(name="regimex-exchange", version=GENERATOR_VERSION),
            time_grid=ModelTimeGrid(
                origin=0.0,
                unit="simulation_step",
                bin_width=1.0,
                completed_horizon=float(steps),
                step_count=steps,
                interval_bounds="[start,stop)",
                terminal_bin="full_bins_only",
            ),
            capabilities=SourceCapabilities(
                execution_prices="complete",
                trade_aggregates="complete",
                accepted_order_aggregates="complete",
                historical_quotes=HistoricalQuoteCapability(
                    mode="retained_snapshots",
                    coverage="complete",
                    provenance="Closing resting book after all events in each simulation step.",
                    complete_event_history_sha256=None,
                    reconstruction_policy=None,
                ),
                price_policy="initial_then_carry",
                trade_count_policy="one_per_partial_execution",
                order_count_policy="accepted_submissions_including_immediate_fills",
                rejected_order_policy="excluded",
            ),
            instruments=tuple(
                ExportInstrument(
                    **instrument.model_dump(),
                    units=InstrumentUnits(
                        price_unit="quote_asset_per_base_asset",
                        volume_unit="base_asset",
                        price_decimal_places=decimal_places(market.tick_size),
                        volume_decimal_places=decimal_places(market.lot_size),
                    ),
                )
                for instrument in run.instruments
            ),
            observations=tuple(
                ExportObservation(
                    **row.model_dump(),
                    price_state="observed" if row.trade_count else "carried",
                    bid_state="observed" if row.best_bid is not None else "empty",
                    ask_state="observed" if row.best_ask is not None else "empty",
                )
                for row in run.observations
            ),
        )
    )
    return export, run.audit


def sample_snapshot(*, seed: int, steps: int, instruments: int) -> ExportV2:
    """Simulate one sampled market with an identity reproducible across invocations."""
    market = sample_market(seed, instruments)
    identity = content_hash(
        {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "steps": steps,
            "instruments": instruments,
        }
    )
    run_id = str(uuid5(NAMESPACE_URL, f"regimex:single-market:{identity}"))
    export, _ = simulate_export(
        market, seed=seed, steps=steps, configuration_id=identity, run_id=run_id
    )
    return export
