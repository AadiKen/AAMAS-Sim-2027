"""Short, development-only BenchMARL IPPO/MAPPO qualification on BCOD M0.

Algorithms, models, collectors, and optimization are owned by BenchMARL.
This module only freezes the task/scenario contract, evaluates on an isolated
development bank, and writes the report artifacts for the EOD qualification.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models import MlpConfig
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from ..adapters.bcod import BCODV3Backend
from ..config import TaskConfig
from ..parallel_env import NavigationParallelEnv
from .benchmarl_env import wrap_for_benchmarl
from .evaluate import evaluate_m0
from .scenarios import sample_m0
from .task import V3BenchMARLTask


SEED = 11
FRAMES = 2500
ROLLOUT = 500
DEV_COUNT = 20
TASK_CONFIG = TaskConfig(agent_count=2, dynamics="bcod-full",
                         action_mode="high_level", deadline_steps=300)
ALGORITHMS = {"ippo": IppoConfig, "mappo": MappoConfig}


class SeededM0Sampler:
    """Pickle-safe deterministic training-distribution sampler for BenchMARL."""
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, _):
        return sample_m0(self.rng, split="EOD-BCOD-train")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def make_dev_bank():
    rng = np.random.default_rng(911_2026)
    result, seen = [], set()
    while len(result) < DEV_COUNT:
        scenario = sample_m0(rng, split="EOD-BCOD-dev")
        digest = scenario.geometry_hash()
        if digest not in seen:
            seen.add(digest)
            result.append(scenario)
    return tuple(result)


def bank_digest(bank):
    return hashlib.sha256(json.dumps([x.geometry_hash() for x in bank],
                                     separators=(",", ":")).encode()).hexdigest()


def actor_policy(actor):
    """Adapt the public BenchMARL actor module to the independent evaluator."""
    class Adapter:
        ids = ("vessel_0", "vessel_1")

        def actions(self, observations):
            matrix = np.stack([observations[k] for k in self.ids])
            try:
                device = next(actor.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            td = TensorDict({"agents": TensorDict(
                {"observation": torch.as_tensor(matrix, dtype=torch.float32, device=device)},
                batch_size=[2])}, batch_size=[])
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                result = actor(td)
                self.policy_std_mean = (float(result["agents", "scale"].detach().mean().cpu())
                                        if ("agents", "scale") in result.keys(True, True)
                                        else None)
                actions = result["agents", "action"].detach().cpu().numpy()
            return {key: actions[i].astype(np.float32) for i, key in enumerate(self.ids)}
    return Adapter()


def eval_bank(actor, bank, trace_dir=None):
    policy = actor_policy(actor)
    report = evaluate_m0(policy, bank, config=TASK_CONFIG, trace_dir=trace_dir)
    report["mean_fleet_reward"] = float(np.mean([e["fleet_reward"] for e in report["episodes"]]))
    report["policy_std_mean"] = getattr(policy, "policy_std_mean", None)
    return report


def check_contract(out: Path, dev_bank):
    """Instantiate and step the physical ParallelEnv; never access legacy banks."""
    seed = 42117
    scenario_rng = np.random.default_rng(seed)
    scenario = sample_m0(scenario_rng, split="contract-check")
    env = NavigationParallelEnv(TASK_CONFIG, scenario=scenario)
    try:
        obs_a, _ = env.reset(seed=seed)
        initial = {k: v.copy() for k, v in obs_a.items()}
        env.close()
        env = NavigationParallelEnv(TASK_CONFIG, scenario=scenario)
        obs_b, _ = env.reset(seed=seed)
        deterministic = all(np.array_equal(initial[k], obs_b[k]) for k in initial)
        backend_class = type(env.backend).__name__
        engine_class = type(env.backend.engine).__name__ if env.backend.engine else None
        physical_plant_mode = ("planar3" if env.backend._physical.reduced_fidelity else "full6") \
            if backend_class == "BCODV3Backend" else None
        pre = {k: (env._frame.readings[k].x_m, env._frame.readings[k].y_m)
               for k in env.agents}
        actions = {k: np.array([0.25, 0.15 if i == 0 else -0.1], dtype=np.float32)
                   for i, k in enumerate(env.agents)}
        _, rewards, terms, truncs, infos = env.step(actions)
        moved = any(math.dist(pre[k], (env._frame.readings[k].x_m,
                                       env._frame.readings[k].y_m)) > 1e-5 for k in pre)
        physical_commands = {k: list(infos[k]["physical_command"]) for k in infos}
        simultaneous = set(infos) == set(actions) and all(
            isinstance(rewards[k], float) for k in rewards)
        # The wrapper specs expose both actor observation and, for MAPPO, global state.
        mappo_task = V3BenchMARLTask(TASK_CONFIG, scenario_sampler=lambda rng: scenario,
                                     centralized_state=True)
        ippo_task = V3BenchMARLTask(TASK_CONFIG, scenario_sampler=lambda rng: scenario,
                                    centralized_state=False)
        m_env = mappo_task.get_env_fun(1, True, seed, "cpu")()
        i_env = ippo_task.get_env_fun(1, True, seed, "cpu")()
        try:
            actor_shape = list(mappo_task.observation_spec(m_env)["agents", "observation"].shape)
            state_shape = list(mappo_task.state_spec(m_env)["state"].shape)
            ippo_state = ippo_task.state_spec(i_env)
        finally:
            m_env.close(); i_env.close()
        contract = {
            "status": "passed" if backend_class == "BCODV3Backend" and moved and deterministic else "failed",
            "backend": backend_class, "physical_engine": engine_class,
            "dynamics": TASK_CONFIG.dynamics,
            "physical_plant_mode": physical_plant_mode,
            "simulator_physics_device": "CPU",
            "kinematic_backend_instantiated": backend_class == "KinematicBackend",
            "actor_observation_shape_per_agent": actor_shape[1:],
            "actor_group_observation_shape": actor_shape,
            "mappo_centralized_state_shape": state_shape,
            "ippo_has_centralized_state": ippo_state is not None,
            "actions": {"shape_per_agent": [2], "simultaneous": simultaneous,
                        "physical_commands_after_decode": physical_commands,
                        "physical_state_moved": moved},
            "seeded_reset_deterministic": deterministic,
            "task_reward_termination": {
                "reward": "NavigationTask per-agent potential-shaping + goal bonus + collision penalty + step penalty",
                "terminal_success": "fleet reaches all goals",
                "terminal_collision": "physical or task collision; fleet terminal",
                "truncation": "deadline steps; fleet truncation"},
            "task_config": asdict(TASK_CONFIG),
            "task_config_sha256": hashlib.sha256(json.dumps(asdict(TASK_CONFIG), sort_keys=True).encode()).hexdigest(),
            "dev_scenarios": len(dev_bank), "dev_bank_sha256": bank_digest(dev_bank),
            "sealed_final_bank_accessed": False,
        }
        return contract
    finally:
        env.close()


def make_experiment(out: Path, algo_key: str, seed: int, frames: int, rollout: int,
                    envs: int, minibatch: int, iters: int, callback: Callback,
                    sampling_device="cpu", train_device="cpu"):
    centralized = algo_key == "mappo"
    sampler = SeededM0Sampler(seed)
    task = V3BenchMARLTask(TASK_CONFIG, scenario_sampler=sampler,
                           centralized_state=centralized,
                           task_name=f"M0-{TASK_CONFIG.dynamics}")
    algorithm = ALGORITHMS[algo_key].get_from_yaml()
    model = MlpConfig.get_from_yaml()
    config = replace(ExperimentConfig.get_from_yaml(),
        sampling_device=sampling_device, train_device=train_device, buffer_device=train_device,
        gamma=TASK_CONFIG.gamma, max_n_frames=frames,
        on_policy_collected_frames_per_batch=rollout,
        on_policy_n_envs_per_worker=envs,
        on_policy_n_minibatch_iters=iters,
        on_policy_minibatch_size=minibatch,
        render=False, evaluation=False, loggers=["csv"], create_json=False,
        checkpoint_interval=rollout, checkpoint_at_end=True,
        keep_checkpoints_num=None, save_folder=str(out / "benchmarl"))
    (out / "benchmarl").mkdir(parents=True, exist_ok=True)
    exp = Experiment(task=task, algorithm_config=algorithm, model_config=model,
                     critic_model_config=MlpConfig.get_from_yaml(), seed=seed,
                     config=config, callbacks=[callback])
    return exp, algorithm, model, config


class EODCallback(Callback):
    def __init__(self, out, algo_key, dev_bank, interval):
        super().__init__()
        self.out = Path(out); self.algo_key = algo_key
        self.dev_bank = dev_bank; self.interval = interval
        self.started = time.monotonic(); self.last_eval = None

    def write_metrics(self, step, report, checkpoint):
        wall = time.monotonic() - self.started
        row = {"algorithm": self.algo_key.upper(), "environment_steps": int(step),
               "wall_seconds": wall, "fleet_return": report["mean_fleet_reward"],
               "success_rate": report["fleet_success_count"] / report["scenario_count"],
               "collision_rate": report["collision_count"] / report["scenario_count"],
               "timeout_rate": report["deadline_count"] / report["scenario_count"],
               "steps_per_second": step / wall if wall else None,
               "policy_std_mean": report.get("policy_std_mean")}
        with (self.out / "development.csv").open("a") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0: writer.writeheader()
            writer.writerow(row)
        if checkpoint:
            torch.save(self.experiment.state_dict(), self.out / "checkpoints" / f"step-{step}.pt")
            torch.save(self.experiment.policy.state_dict(), self.out / "checkpoints" / f"actor-step-{step}.pt")
        self.last_eval = step

    def evaluate(self, step, checkpoint=False):
        report = eval_bank(self.experiment.policy, self.dev_bank,
                           trace_dir=self.out / "dev_traces" / f"step-{step}")
        write_json(self.out / "dev_reports" / f"step-{step}.json", report)
        self.write_metrics(step, report, checkpoint)
        metric = (report["fleet_success_count"], -report["collision_count"],
                  -report["deadline_count"],
                  -(report["mean_success_completion_steps"] or 1e9))
        best_path = self.out / "checkpoints" / "best-dev.pt"
        if not hasattr(self, "best") or metric > self.best:
            self.best = metric
            torch.save(self.experiment.policy.state_dict(), self.out / "checkpoints" / "best-dev-actor.pt")
            torch.save(self.experiment.state_dict(), self.out / "checkpoints" / "best-dev.pt")
            write_json(self.out / "best_dev.json", {"step": step, "score": metric})

    def on_train_end(self, training_td, group):
        if group != "agents": return
        step = int(self.experiment.total_frames)
        if step % self.interval == 0:
            self.evaluate(step, checkpoint=True)
        row = {"step": step, "iter": self.experiment.n_iters_performed}
        for key in training_td.keys(True, True):
            value = training_td.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                row[str(key)] = float(value.detach().cpu())
        row["wall_seconds"] = time.monotonic() - self.started
        with (self.out / "training.csv").open("a") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0: writer.writeheader()
            writer.writerow(row)


def train_one(root: Path, algo_key: str, seed=SEED, frames=FRAMES, rollout=ROLLOUT,
              envs=1, minibatch=250, iters=5, sampling_device="cpu", train_device="cpu"):
    out = root / algo_key
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if p.name != "checkpoints" or any(p.iterdir())]
    if existing:
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {out}")
    (out / "checkpoints").mkdir(exist_ok=True)
    bank = make_dev_bank()
    callback = EODCallback(out, algo_key, bank, rollout)
    exp, algorithm, model, config = make_experiment(
        out, algo_key, seed, frames, rollout, envs, minibatch, iters, callback,
        sampling_device=sampling_device, train_device=train_device)
    import benchmarl, torchrl, tensordict, pettingzoo
    versions = {m.__name__: m.__version__ for m in (torch, torchrl, tensordict, benchmarl, pettingzoo)}
    manifest = {"algorithm": "BENCHMARL_" + algo_key.upper(), "seed": seed,
        "frames_requested": frames, "rollout_frames": rollout, "environment_count": envs,
        "training_distribution": "sample_m0, version m0-two-vessel-crossing-v1, split EOD-BCOD-train",
        "development_bank_sha256": bank_digest(bank), "development_scenarios": len(bank),
        "task_config": asdict(TASK_CONFIG), "centralized_state": algo_key == "mappo",
        "versions": versions, "algorithm_config": asdict(algorithm),
        "model_config": asdict(model), "experiment_config": asdict(config),
        "sampling_device": sampling_device, "training_device": train_device,
        "simulator_physics_device": "CPU", "final_bank_accessed": False}
    write_json(out / "manifest.json", manifest)
    torch.save(exp.policy.state_dict(), out / "checkpoints" / "initial-actor.pt")
    torch.save(exp.state_dict(), out / "checkpoints" / "initial-experiment.pt")
    start = time.monotonic()
    callback.evaluate(0, checkpoint=False)
    status = "complete"
    try:
        exp.run()
    except BaseException as error:
        status = "failed"
        write_json(out / "failure.json", {"type": type(error).__name__, "error": str(error),
                                           "frames": exp.total_frames,
                                           "wall_seconds": time.monotonic() - start})
        raise
    finally:
        if callback.last_eval != exp.total_frames:
            callback.evaluate(int(exp.total_frames), checkpoint=True)
        torch.save(exp.policy.state_dict(), out / "checkpoints" / "final-actor.pt")
        total_wall = time.monotonic() - start
        write_json(out / "status.json", {"status": status, "environment_steps": int(exp.total_frames),
                                          "wall_seconds": total_wall,
                                          "steps_per_second": exp.total_frames/max(total_wall, 1e-9),
                                          "iterations": exp.n_iters_performed})
    _write_scalar_training_csv(out)
    return {"algorithm": algo_key, "status": status, "frames": exp.total_frames,
            "wall_seconds": time.monotonic()-start}


def _write_scalar_training_csv(out: Path):
    """Join raw BenchMARL scalar logs into a compact iteration CSV."""
    scalar_dirs = list((out / "benchmarl").glob("*/**/scalars"))
    if not scalar_dirs:
        return
    scalars = scalar_dirs[0]
    def values(pattern):
        files = list(scalars.glob(pattern))
        if not files: return {}
        result = {}
        with files[0].open() as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    result[int(float(row[0]))] = float(row[1])
        return result
    names = {
        "loss_policy": "train_agents_loss_objective.csv",
        "loss_critic": "train_agents_loss_critic.csv",
        "entropy": "train_agents_entropy.csv",
        "loss_entropy": "train_agents_loss_entropy.csv",
        "kl_approx": "train_agents_kl_approx.csv",
        "explained_variance": "train_agents_explained_variance.csv",
        "episode_return_mean": "collection_reward_episode_reward_mean.csv",
        "episode_return_min": "collection_reward_episode_reward_min.csv",
        "episode_return_max": "collection_reward_episode_reward_max.csv",
        "collection_seconds": "timers_collection_time.csv",
        "training_seconds": "timers_training_time.csv",
    }
    metrics = {key: values(filename) for key, filename in names.items()}
    frames_by_iter = values("counters_total_frames.csv")
    rows = []
    for iteration, frame_count in frames_by_iter.items():
        row = {"iteration": iteration, "environment_steps": int(frame_count)}
        for name, series in metrics.items():
            if iteration in series: row[name] = series[iteration]
        rows.append(row)
    if rows:
        with (out / "training.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--rollout", type=int, default=ROLLOUT)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--minibatch", type=int, default=250)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--sampling-device", default="cpu")
    parser.add_argument("--train-device", default="cpu")
    args = parser.parse_args(argv)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(train_one(root, args.algorithm, args.seed, args.frames, args.rollout,
                               args.envs, args.minibatch, args.iters,
                               args.sampling_device, args.train_device), indent=2))


if __name__ == "__main__":
    main()
