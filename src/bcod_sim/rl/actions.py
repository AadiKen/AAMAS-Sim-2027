"""Hash-stable heterogeneous action contracts."""

from dataclasses import dataclass
import math

from bcod_sim.config.hashing import content_hash
from bcod_sim.core.errors import PhysicalValidationError


@dataclass(frozen=True)
class ActionField:
    name: str
    command_type: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    preprocessing: str = "none"

    def __post_init__(self) -> None:
        size = math.prod(self.shape)
        if (not self.name or not self.command_type or not self.dtype or not self.units or not self.preprocessing or
                any(x <= 0 for x in self.shape) or len(self.minimum) != size or len(self.maximum) != size or
                any(not math.isfinite(x) for x in (*self.minimum, *self.maximum)) or
                any(low >= high for low, high in zip(self.minimum, self.maximum))):
            raise PhysicalValidationError("Invalid action field contract")


@dataclass(frozen=True)
class ActionContract:
    fields: tuple[ActionField, ...]

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if not self.fields or len(names) != len(set(names)):
            raise PhysicalValidationError("Action contract fields must be nonempty and unique")

    @property
    def content_hash(self) -> str:
        return content_hash({"fields": [vars(field) for field in self.fields]})
