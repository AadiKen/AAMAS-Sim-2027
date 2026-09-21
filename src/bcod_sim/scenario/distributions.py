"""Closed distribution primitives for reproducible scenario sampling."""

import math
import random
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Fixed(StrictDistribution):
    kind: Literal["fixed"]
    value: float = Field(allow_inf_nan=False)


class Uniform(StrictDistribution):
    kind: Literal["uniform"]
    low: float = Field(allow_inf_nan=False)
    high: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.low >= self.high:
            raise ValueError("Uniform distribution requires low < high")
        return self


class Categorical(StrictDistribution):
    kind: Literal["categorical"]
    values: tuple[float, ...] = Field(min_length=1)
    weights: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def valid(self):
        if not all(math.isfinite(x) for x in self.values):
            raise ValueError("Categorical values must be finite")
        if self.weights is not None and (len(self.weights) != len(self.values) or
                                         not all(math.isfinite(x) and x >= 0 for x in self.weights) or
                                         sum(self.weights) <= 0):
            raise ValueError("Categorical weights are invalid")
        return self


class BoundedNormal(StrictDistribution):
    kind: Literal["normal_bounded"]
    mean: float = Field(allow_inf_nan=False)
    std: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(allow_inf_nan=False)
    high: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.low >= self.high or not self.low <= self.mean <= self.high:
            raise ValueError("Bounded normal requires low <= mean <= high")
        return self


class SampledList(StrictDistribution):
    kind: Literal["sampled_list"]
    values: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def finite(self):
        if not all(math.isfinite(x) for x in self.values):
            raise ValueError("Sample list values must be finite")
        return self


Distribution = Annotated[Fixed | Uniform | Categorical | BoundedNormal | SampledList, Field(discriminator="kind")]


def draw(spec: Distribution, rng: random.Random) -> float:
    if isinstance(spec, Fixed):
        return spec.value
    if isinstance(spec, Uniform):
        return rng.uniform(spec.low, spec.high)
    if isinstance(spec, Categorical):
        return rng.choices(spec.values, weights=spec.weights, k=1)[0]
    if isinstance(spec, SampledList):
        return rng.choice(spec.values)
    # Inverse-CDF sampling avoids retry counts and guarantees bounds.
    from statistics import NormalDist
    normal = NormalDist(spec.mean, spec.std)
    low, high = normal.cdf(spec.low), normal.cdf(spec.high)
    if high <= low:
        raise ValueError("Bounded normal CDF interval is numerically unresolved")
    probability = low + rng.random() * (high - low)
    probability = min(math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), probability))
    return min(spec.high, max(spec.low, normal.inv_cdf(probability)))
