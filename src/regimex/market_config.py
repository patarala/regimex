"""Exchange and participant settings."""

import math
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from regimex.contracts import Nonnegative, Positive, StrictModel

Probability = Annotated[float, Field(ge=0, le=1)]
Lots = Annotated[int, Field(ge=0, le=10**12)]


class ParticipantProfile(StrictModel):
    count: Annotated[int, Field(ge=1, le=256)] = 16
    strategy: Literal["random", "trend"] = "random"
    buy_probability: Probability = 0.45
    sell_probability: Probability = 0.45
    hold_probability: Probability = 0.1
    price_offset_bps: tuple[float, float] = (-100.0, 100.0)
    volume_lots: tuple[int, int] = (1, 100)
    cash: Nonnegative = 1_000_000.0
    asset_lots: Lots = 100_000
    trend_window_trades: Annotated[int, Field(ge=1, le=4096)] = 20
    trend_threshold: Annotated[float, Field(gt=0, le=1)] = 0.005

    @model_validator(mode="after")
    def valid_profile(self) -> Self:
        if not math.isclose(
            self.buy_probability + self.sell_probability + self.hold_probability,
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("BUY, SELL and HOLD probabilities must sum to one")
        if not -9500 <= self.price_offset_bps[0] <= self.price_offset_bps[1] <= 10000:
            raise ValueError("price offsets must be ordered within [-9500, 10000] bps")
        if not 1 <= self.volume_lots[0] <= self.volume_lots[1] <= 10**6:
            raise ValueError("volume lots must be positive, ordered and at most one million")
        return self


class MarketConfig(StrictModel):
    initial_prices: Annotated[tuple[Positive, ...], Field(min_length=1, max_length=2)] = (100.0,)
    tick_size: Positive = 0.01
    lot_size: Positive = 0.01
    profiles: Annotated[tuple[ParticipantProfile, ...], Field(min_length=1, max_length=16)] = (
        ParticipantProfile(),
    )
    test_participant: ParticipantProfile = ParticipantProfile(count=1)

    def cash_units(self, cash: float) -> int:
        return Fraction(str(cash)) // (Fraction(str(self.tick_size)) * Fraction(str(self.lot_size)))

    def price_ticks(self, price: float) -> int:
        """Convert a configured decimal price without rounding fractional ticks."""
        ticks = Fraction(str(price)) / Fraction(str(self.tick_size))
        if not 1 <= ticks <= 10**12:
            raise ValueError("initial prices must fit the supported tick range")
        if ticks.denominator != 1:
            raise ValueError("initial prices must be exact multiples of tick_size")
        return ticks.numerator

    @model_validator(mode="after")
    def valid_market(self) -> Self:
        if self.test_participant.count != 1:
            raise ValueError("the test participant must have count=1")
        if sum(profile.count for profile in self.profiles) > 256:
            raise ValueError("at most 256 background participants are supported")
        for price in self.initial_prices:
            self.price_ticks(price)
        unit = self.tick_size * self.lot_size
        if not math.isfinite(unit) or unit <= 0:
            raise ValueError("tick and lot sizes must have a finite positive product")
        if any(
            not math.isfinite(p.cash / unit) or p.cash / unit > 10**18
            for p in (*self.profiles, self.test_participant)
        ):
            raise ValueError("cash exceeds the supported integer settlement range")
        return self
