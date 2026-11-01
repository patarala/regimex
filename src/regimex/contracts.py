"""Versioned, strict contracts for aggregate experiment exports and separate annotations."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_STEPS = 4096


Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]
Count = Annotated[int, Field(ge=0, le=2**31 - 1)]
Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Producer(StrictModel):
    name: Identifier
    version: Identifier


class Instrument(StrictModel):
    instrument_id: Identifier
    base_asset: Identifier
    quote_asset: Identifier
    initial_price: Positive

    @model_validator(mode="after")
    def distinct_assets(self) -> Self:
        if self.base_asset == self.quote_asset:
            raise ValueError("base and quote assets must differ")
        return self


class Observation(StrictModel):
    instrument_id: Identifier
    step: Count
    last_price: Positive | None
    trade_count: Count | None
    trade_volume: Nonnegative | None
    buy_order_count: Count | None
    buy_order_volume: Nonnegative | None
    sell_order_count: Count | None
    sell_order_volume: Nonnegative | None
    best_bid: Positive | None
    best_ask: Positive | None

    @model_validator(mode="after")
    def coherent_aggregates(self) -> Self:
        for count, volume in (
            (self.trade_count, self.trade_volume),
            (self.buy_order_count, self.buy_order_volume),
            (self.sell_order_count, self.sell_order_volume),
        ):
            if count is not None and volume is not None and (count == 0) != (volume == 0):
                raise ValueError("observed count and volume must agree about inactivity")
        if (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid > self.best_ask
        ):
            raise ValueError("final best quotes must not be crossed")
        return self
