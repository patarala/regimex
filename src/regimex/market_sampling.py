"""Seeded sampling of participant configurations, independent of annotation policy."""

from typing import Literal

import numpy as np

from regimex.market_config import MarketConfig, ParticipantProfile

GENERATOR_VERSION = "4.0"
SamplingFamily = Literal["inactive", "one_sided", "fixed_price", "heterogeneous"]


def _sample_profile(
    rng: np.random.Generator, count: int, strategy: Literal["random", "trend"]
) -> ParticipantProfile:
    width = float(np.exp(rng.uniform(np.log(2), np.log(400))))
    center = float(rng.uniform(-0.5, 0.5) * width)
    buy, sell, hold = (float(x) for x in rng.dirichlet([2, 2, 1]))
    maximum_lots = int(np.exp(rng.uniform(0, np.log(501))))
    minimum_lots = int(rng.integers(1, maximum_lots + 1))
    return ParticipantProfile(
        count=count,
        strategy=strategy,
        buy_probability=buy,
        sell_probability=sell,
        hold_probability=hold,
        price_offset_bps=(center - width, center + width),
        volume_lots=(minimum_lots, maximum_lots),
        cash=float(10 ** rng.uniform(2, 7)),
        asset_lots=int(10 ** rng.uniform(1, 6)),
        trend_window_trades=int(rng.integers(2, 65)),
        trend_threshold=float(10 ** rng.uniform(-4, -1.5)),
    )


def sample_market(seed: int, instrument_count: int) -> MarketConfig:
    """Sample configurations, not desired outcomes; probabilities describe coverage only."""
    if instrument_count not in (1, 2):
        raise ValueError("one or two instruments are required")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0xC0FFEE])))
    prices = tuple(float(rng.integers(2000, 20001)) / 100 for _ in range(instrument_count))
    choice = rng.random()
    family: SamplingFamily
    if choice < 0.05:
        family = "inactive"
    elif choice < 0.10:
        family = "one_sided"
    elif choice < 0.20:
        family = "fixed_price"
    else:
        family = "heterogeneous"
    profile_count = int(rng.integers(1, 6))
    profiles = [
        _sample_profile(
            rng,
            int(rng.integers(1, 33)),
            "random" if i == 0 or rng.random() < 0.5 else "trend",
        )
        for i in range(profile_count)
    ]
    # The tested participant has its own strategy, resources and size range.
    test_participant = _sample_profile(rng, 1, "random" if rng.random() < 0.5 else "trend")
    configured = [*profiles, test_participant]
    if family == "inactive":
        configured = [
            profile.model_copy(
                update={"buy_probability": 0.0, "sell_probability": 0.0, "hold_probability": 1.0}
            )
            for profile in configured
        ]
    elif family == "one_sided":
        buy = float(rng.integers(2))
        configured = [
            profile.model_copy(
                update={
                    "buy_probability": buy,
                    "sell_probability": 1 - buy,
                    "hold_probability": 0.0,
                }
            )
            for profile in configured
        ]
    elif family == "fixed_price":
        configured = [
            profile.model_copy(update={"price_offset_bps": (0.0, 0.0)}) for profile in configured
        ]
    profiles = configured[:-1]
    profiles = [profiles[int(i)] for i in rng.permutation(len(profiles))]
    return MarketConfig(
        initial_prices=prices, profiles=tuple(profiles), test_participant=configured[-1]
    )
