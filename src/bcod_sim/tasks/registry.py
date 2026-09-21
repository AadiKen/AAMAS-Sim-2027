"""Closed task registry; unknown identities never fall back."""

from pydantic import ValidationError

from bcod_sim.config.registry import Definition
from bcod_sim.core.errors import ConfigSchemaError, UnknownReferenceError
from bcod_sim.tasks.primitives import (CoverageParams, CoverageTask, FormationParams,
                                       FormationTask, TeamWaypointParams, TeamWaypointTask,
                                       WaypointParams, WaypointTask)


_TASKS = {
    "waypoint": (WaypointParams, WaypointTask),
    "waypoint_team": (TeamWaypointParams, TeamWaypointTask),
    "formation": (FormationParams, FormationTask),
    "coverage": (CoverageParams, CoverageTask),
}


def build_task(definition: Definition):
    if definition.kind != "task":
        raise UnknownReferenceError("Definition is not a task")
    kind = definition.payload.get("kind")
    if kind not in _TASKS:
        raise UnknownReferenceError(f"Unknown task primitive: {kind}")
    schema, task_class = _TASKS[kind]
    try:
        params = schema.model_validate(dict(definition.payload))
    except ValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc
    return task_class(params)
