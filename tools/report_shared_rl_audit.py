"""Aggregate completed audit artifacts and render measured curves."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/shared-rl-matplotlib')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('artifacts/shared-rl-audit')

def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

def diagnostics(directory):
    updates=lines(directory/'updates.jsonl');episodes=lines(directory/'episodes.jsonl')
    if not updates:return {}
    # Derive bootstrap coverage from actual episode boundaries and true termination,
    # avoiding ambiguity of a bootstrap value numerically equal to zero.
    fractions=[]
    for u in updates:
        end=u['environment_steps'];start=end-u['rollout_records']
        boundaries=[r for r in episodes if start<r['environment_steps']<=end]
        cursor=start;boot=0
        for r in boundaries:
            if not r['collision_count'] and not r['fleet_success']:boot+=r['environment_steps']-cursor
            cursor=r['environment_steps']
        boot+=end-cursor;fractions.append(boot/u['rollout_records'])
    ratios=[u['critic_body_grad_norm']/max(u['actor_body_grad_norm'],1e-12) for u in updates]
    collision_updates=[u for u in updates if any(u['environment_steps']-u['rollout_records']<e['environment_steps']<=u['environment_steps'] and e['collision_count'] for e in episodes)]
    return {'updates':len(updates),'episodes':len(episodes),'median_critic_actor_body_grad_ratio':float(np.median(ratios)),
        'fraction_critic_body_larger':float(np.mean(np.asarray(ratios)>1)),
        'collision_update_median_grad_ratio':float(np.median([u['critic_body_grad_norm']/max(u['actor_body_grad_norm'],1e-12) for u in collision_updates])) if collision_updates else None,
        'mean_clip_scale':float(np.mean([u['clip_scale'] for u in updates])),
        'mean_bootstrap_fraction':float(np.mean(fractions)),
        'updates_with_boundary_fraction':float(np.mean([u['episode_boundaries']>0 for u in updates])),
        'mean_complete_episodes_per_update':float(np.mean([sum(u['environment_steps']-u['rollout_records']<=e['environment_steps']-e['steps'] and e['environment_steps']<=u['environment_steps'] for e in episodes) for u in updates])),
        'mean_episode_length':float(np.mean([e['steps'] for e in episodes])) if episodes else None,
        'median_episode_length':float(np.median([e['steps'] for e in episodes])) if episodes else None,
        'return_target_variance_range':[min(u['std_return_target']**2 for u in updates),max(u['std_return_target']**2 for u in updates)],
        'advantage_variance_range':[min(u['std_advantage']**2 for u in updates),max(u['std_advantage']**2 for u in updates)]}

history=read(root/'historical-training.json');probe=read(root/'historical-action-probe.json')
evals=sorted([read(p) for p in root.glob('bcod-eval-*/summary-first30.json')],key=lambda x:x['environment_steps'])
fig,axes=plt.subplots(2,2,figsize=(12,8))
axes[0,0].plot([r['end'] for r in history['windows']],[r['collision_rate'] for r in history['windows']],label='stochastic training, 5k windows',marker='o')
if evals:axes[0,0].plot([r['environment_steps'] for r in evals],[r['collision_rate'] for r in evals],label='deterministic, fixed 30-scenario subset',marker='o')
axes[0,0].set_ylabel('Collision fraction');axes[0,0].legend()
axes[0,1].plot([r['steps'] for r in probe],[r['mean_deterministic_surge'] for r in probe],label='mean deterministic surge')
axes[0,1].plot([r['steps'] for r in probe],[r['mean_abs_deterministic_yaw'] for r in probe],label='mean absolute deterministic yaw')
axes[0,1].set_title('Fixed reset-observation probe');axes[0,1].legend()
for i,name in enumerate(('surge raw std','yaw raw std')):axes[1,0].plot([r['steps'] for r in probe],[r['raw_std'][i] for r in probe],label=name)
axes[1,0].legend()
for name in ('progress','collision','step'):
 axes[1,1].plot([r['end'] for r in history['windows']],[r['mean_reward_components'][name] for r in history['windows']],label=name,marker='o')
axes[1,1].set_ylabel('Mean fleet reward component');axes[1,1].legend()
for ax in axes.flat:ax.set_xlabel('Environment steps');ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(root/'learning-curves.png',dpi=160);plt.close(fig)
all_diagnostics={p.name:diagnostics(p) for p in root.iterdir() if p.is_dir() and (p/'updates.jsonl').exists()}
(root/'diagnostics-summary.json').write_text(json.dumps(all_diagnostics,indent=2))
fig,axes=plt.subplots(2,2,figsize=(12,8))
for name in ('fleet-original','fleet-penalty'):
 p=root/name
 updates=lines(p/'updates.jsonl')
 if not updates:continue
 x=[u['environment_steps'] for u in updates]
 axes[0,0].plot(x,[u['critic_body_grad_norm']/max(u['actor_body_grad_norm'],1e-12) for u in updates],label=name)
 axes[0,1].plot(x,[u['std_return_target']**2 for u in updates],label=name)
 axes[1,0].plot(x,[u['actor_mean_surge'] for u in updates],label=name+' surge')
 axes[1,0].plot(x,[u['actor_mean_abs_yaw'] for u in updates],label=name+' |yaw|',linestyle='--')
 if (p/'evaluations.json').exists():
  e=read(p/'evaluations.json');axes[1,1].plot([r['environment_steps'] for r in e],[r['collision_rate'] for r in e],label=name,marker='o')
for ax,title in zip(axes.flat,('Critic / actor shared-body gradient norm','Return-target variance','Training-state deterministic action mean','Fixed-bank deterministic collision fraction')):
 ax.set_title(title);ax.set_xlabel('Environment steps');ax.grid(alpha=.3)
 if ax.lines:ax.legend()
fig.tight_layout();fig.savefig(root/'ablation-diagnostics.png',dpi=160);plt.close(fig)
print(json.dumps(all_diagnostics,indent=2))

collision_rows=[]
for path in root.rglob('*.jsonl'):
 if path.name=='collision-episodes.jsonl':continue
 for row in lines(path):
  if row.get('collision_count',0) and 'per_agent_return' in row:
   collision_rows.append({'source':str(path.relative_to(root)),**row})
(root/'collision-episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in collision_rows))
