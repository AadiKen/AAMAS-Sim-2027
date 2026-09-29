# RL quickstart

Use `make_env()` when building a custom research training workflow. It reads a self-contained experiment bundle and returns Gymnasium for one RL agent or PettingZoo Parallel for a fleet.

## Generic SARL

```python
from bcod_sim.rl import make_env

env = make_env("examples/single_vessel_rl.yaml")
obs, info = env.reset(seed=11)
obs, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

## Generic MARL

```python
from bcod_sim.rl import make_env

env = make_env("examples/fleet_rl.yaml")

obs, infos = env.reset(seed=11)

actions = {
    agent: env.action_space(agent).sample()
    for agent in env.agents
}

obs, rewards, terms, truncs, infos = env.step(actions)
state = env.state()
env.close()
```

Use `bcod benchmark ...` for the frozen paper training and evaluation harness.

## Paper SARL

```bash
bcod benchmark check configs/paper/sarl-s0.yaml

bcod benchmark train configs/paper/sarl-s0.yaml \
  --seed 11
```

## Paper MARL

```bash
bcod benchmark check configs/paper/marl-m0.yaml

bcod benchmark train configs/paper/marl-m0.yaml \
  --seed 11
```

Install the optional MARL dependencies with `pip install 'bcod-sim[marl]'` before running paper MARL training.
