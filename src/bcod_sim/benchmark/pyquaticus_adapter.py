"""Process-isolated bridge to Pyquaticus's pinned Python 3.10 runtime."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from .core import BenchmarkConfig, Scenario, Truth, VesselReading


class PyquaticusAdapter:
    def __init__(self, python_executable: str, config: BenchmarkConfig = BenchmarkConfig()):
        self.config = config
        worker = Path(__file__).with_name("pyquaticus_worker.py")
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-benchmark-cache")
        self.process = subprocess.Popen([python_executable, "-u", str(worker)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    def _request(self, payload):
        if self.process.poll() is not None:
            raise RuntimeError(f"Pyquaticus worker exited with status {self.process.returncode}")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Pyquaticus worker closed output")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        if payload["op"] == "close":
            return None
        return ({name: VesselReading(**item) for name, item in result["readings"].items()},
                {name: Truth(**item) for name, item in result["truth"].items()})

    def reset(self, scenario: Scenario, seed: int):
        return self._request({"op": "reset", "seed": seed, "scenario": {**asdict(scenario), **asdict(self.config)}})

    def step(self, actions):
        return self._request({"op": "step", "actions": actions})

    def close(self):
        if self.process.poll() is None:
            try:
                self._request({"op": "close"})
            finally:
                self.process.terminate()
                self.process.wait(timeout=5)
