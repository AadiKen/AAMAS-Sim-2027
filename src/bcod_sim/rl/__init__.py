"""RL spaces, contracts, adapters, and convenience factories.

Exports are loaded lazily because the physical engine imports RL observation
contracts while RL centralized-state contracts also refer to the engine.
"""

__all__ = ["CentralizedStateContract", "ExperimentBundle", "RLProfile", "make_env", "make_gym_env", "make_parallel_env"]


def __getattr__(name):
    if name == "CentralizedStateContract":
        from .centralized_state import CentralizedStateContract
        return CentralizedStateContract
    if name == "RLProfile":
        from .profile import RLProfile
        return RLProfile
    if name == "ExperimentBundle":
        from .factory import ExperimentBundle
        return ExperimentBundle
    if name in {"make_env", "make_gym_env", "make_parallel_env"}:
        from . import factory
        return getattr(factory, name)
    raise AttributeError(name)
