"""Read-only action and episode accounting for the shared RL audit."""
import math
import numpy as np
from .core import NAMES, BenchmarkConfig


def action_summary(rows):
    if not rows:
        return {}
    means = np.asarray([r['actor_mean'] for r in rows]).reshape(-1, 2)
    actions = np.asarray([r['sampled'] for r in rows]).reshape(-1, 2)
    return {'actor_mean_surge':float(means[:,0].mean()), 'actor_mean_yaw':float(means[:,1].mean()),
            'actor_mean_abs_yaw':float(abs(means[:,1]).mean()),
            'sampled_surge':float(actions[:,0].mean()), 'sampled_yaw':float(actions[:,1].mean()),
            'surge_std':float(actions[:,0].std()), 'yaw_std':float(actions[:,1].std()),
            'mean_abs_yaw':float(abs(actions[:,1]).mean()),
            'near_limit_fraction':(abs(actions)>.95).mean(axis=0).tolist(),
            'negative_surge_fraction':float((actions[:,0]<0).mean()),
            'log_std':rows[-1]['log_std'], 'raw_std':rows[-1]['raw_std']}


class EpisodeDiagnostics:
    def __init__(self, gamma=None):
        if gamma is None:
            gamma = BenchmarkConfig().gamma
        self.gamma=gamma
        self.rewards=[]
        self.components={n:dict.fromkeys(('potential_shaping','goal','collision','step'),0.) for n in NAMES}
        self.previous_reached=dict.fromkeys(NAMES,False)
        self.left_goal=set(); self.post_goal_collisions=set()

    def add(self, reward, info, env):
        self.rewards.append([reward[n] for n in NAMES])
        for i,n in enumerate(NAMES):
            for key,value in info['reward_components'][n].items():self.components[n][key]+=value
            if self.previous_reached[n]:
                p=env.truth[n]
                if math.dist((p.x_m,p.y_m),env.scenario.goals[i])>env.config.goal_radius_m:self.left_goal.add(n)
                if n in info['colliding_agents']:self.post_goal_collisions.add(n)
        self.previous_reached=info['per_agent_success'].copy()

    def result(self, info):
        values=np.asarray(self.rewards)
        returns=values.sum(axis=0)
        discounted=(values*self.gamma**np.arange(len(values))[:,None]).sum(axis=0)
        streak=[]
        for i in range(4):
            count=0
            for reward in values[-2::-1,i]:
                if reward<=0:break
                count+=1
            streak.append(count)
        return {**info,'per_agent_return':dict(zip(NAMES,returns.tolist())),
                'per_agent_discounted_return':dict(zip(NAMES,discounted.tolist())),
                'fleet_return':float(returns.sum()),'discounted_fleet_return':float(discounted.sum()),
                'terminal_rewards':dict(zip(NAMES,values[-1].tolist())),
                'reward_decomposition':self.components,
                'positive_return_agents':int((returns>0).sum()),
                'positive_discounted_return_agents':int((discounted>0).sum()),
                'preterminal_rewards_last_10':values[-11:-1].tolist(),
                'positive_reward_streak_before_terminal':dict(zip(NAMES,streak)),
                'reached_then_left':sorted(self.left_goal),
                'post_goal_collisions':sorted(self.post_goal_collisions)}
