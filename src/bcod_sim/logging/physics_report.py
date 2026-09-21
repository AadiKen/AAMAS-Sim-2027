"""Machine-readable per-fingerprint physics report assembly."""
from typing import Any
from bcod_sim.dynamics.plant6 import Plant6

def physics_report(plant:Plant6,*,fingerprint:str,equilibrium:dict[str,Any],provenance:dict[str,Any],validity:dict[str,Any],actuators:list[dict[str,Any]],known_limitations:list[str])->dict[str,Any]:
    mass=plant.mass.diagnostics()
    serializable={k:(v.tolist() if hasattr(v,"tolist") else v) for k,v in mass.items()}
    return {"plant_fingerprint":fingerprint,"mass_properties":serializable,"equilibrium":equilibrium,
            "hydrostatic_model":plant.hydrostatics.model_name,"damping_model":"full-matrix + nonlinear coefficients",
            "crossflow_model":plant.crossflow.model_name,"actuators":actuators,"validity":validity,
            "provenance":provenance,"force_accounting":"sum of WrenchLedger terms equals tau_total",
            "known_limitations":known_limitations}
