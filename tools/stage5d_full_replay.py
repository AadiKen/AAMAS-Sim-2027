"""Full-field fixed-seed replay of the Stage 5D maximum-stress world."""

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage5d_training_validation import command, stress_env


def normalize(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: normalize(getattr(value, field.name)) for field in fields(value)}
    return value


def run(seed=101, steps=300):
    env = stress_env(seed=seed, max_steps=steps+1)
    env.reset(seed=seed)
    digest = hashlib.sha256()
    contacts = observations = 0
    for index in range(steps):
        actions = {name: command(20. if (index // 11 + ordinal) % 2 else -20.)
                   for ordinal, name in enumerate(env.agents)}
        obs, rewards, terms, truncs, infos = env.step(actions)
        contacts += len({event.contact_id for info in infos.values()
                         for event in info["contact_events"]})
        observations += sum(bool(value) for value in obs.values())
        payload = {"states": env.engine.states, "observations": obs,
                   "rewards": rewards, "terminations": terms,
                   "truncations": truncs, "infos": infos,
                   "events": env.engine.last_frame.contact_events if hasattr(env.engine, "last_frame") else
                   tuple(event for info in infos.values() for event in info["contact_events"])}
        digest.update(json.dumps(normalize(payload), sort_keys=True,
                                 separators=(",", ":"), default=str).encode())
    return {"seed": seed, "steps": steps, "sha256": digest.hexdigest(),
            "contacts": contacts, "nonempty_observations": observations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    first, second = run(), run()
    if first["sha256"] != second["sha256"]:
        raise AssertionError("Full-field stress replay diverged")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "first": first,
                                      "second": second}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
