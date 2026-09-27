# Active audit execution

All work is for the latest shared-RL audit attachment. No simulator physics or production reward weights changed.

Native BCOD fixed nominal bank: seeds 700000–700099 (persisted in nominal-100). Historical input: Downloads/benchmark-bcod-50k-corrected, latest checkpoint 38250, seed11. Model-only evaluation of these checkpoints is NOT training or resume.

Running jobs:
- initial BCOD deterministic 100: session24716, /private/tmp/rl-bcod-eval.log
- final BCOD deterministic100: session4269, /private/tmp/rl-bcod-final.log
- intermediate native BCOD100 each plus initial stochastic100: session61168, /private/tmp/rl_eval_checkpoints.py, /private/tmp/rl-checkpoints.log; pool4, steps1000,2500,5000,10000,15000,20000,25000,30000,35000,0(stochastic)
- final BCOD stochastic100: session68451, /private/tmp/rl-bcod-final-stochastic.log
- frozen BCOD latest-policy gradient probe1500 native steps, LR0: session41503, /private/tmp/rl-gradient-probe.log
- Pyquaticus ladder A5000: session33728 completed or finishing; output ladder-A-v2 (first ladder-A attempt failed due production minimum obstacle validation before any rollout)
- ladder B,C,D,E5000 sequential: session96679 /private/tmp/rl_ladder_rest.py, logs /private/tmp/rl-ladder-{kind}.log
- Py full F5000 original: session53979 /private/tmp/rl-fleet-original.log
- Py full F5000 all-agent-penalty diagnostic ablation: session80491 /private/tmp/rl-fleet-penalty.log
- Py observation/truth geometric baselines20 each on A–F: session26715 /private/tmp/rl_baselines.py, /private/tmp/rl-baselines.log

Use report_shared_rl_audit.py to aggregate completed artifacts and plot. User expects docs/shared_rl_learning_audit.md with 13 specified sections and exactly one primary A–E classification at end. Strong evidence points to B reward/objective misalignment; additional partial-observability and learner sample efficiency limitations are not measured causal percentages.

Known results: 54 benchmark tests passed before latest per-agent gradient diagnostics/test addition. Re-run relevant tests after final edits. No full suite needed for this task unless concrete risk.
Historical 118 episodes through38250,103collisions,51positivefleet collision episodes. Hypothetical all-agent penalty makes0/103positive on same historical trajectories. Mean episode321.11steps median270. Updates with boundaries69.93%; wholly contained completeepisodes/update.07843.
Same-geometry reward example 72m goal: crash196.48 total113.802 discounted vs slow safe detour338.518 total22.937 discounted. Crash all-agentcounterfactual46.48 total82.841discounted stillbeatsdetour. No gamma change made.
Fixed-reset400-observation checkpoint probe: deterministic surge -.01239→.08581; abs yaw .06195→.03712; raw std .4966→.4963/.4958. Historical stats are probes, separate from rollout stats.
A/B5000 ladder no deterministic success; geometric observation baseline solvesA/B/C. FinitialPy100det zero collisions andzero success.
Production changes: runner._return_targets extracted, detached rewards/bootstrap; _update diagnostics optional, actor/critic/grad norms/MCentropy; actiondiagnostics optional; trainupdates.jsonl perupdate; EpisodeDiagnostics in train/eval. core adds info colliding_agents/reward_components with unchanged numericreward, opt-in diagnostic_allow_sparse to allow0–5 obstacles in sanitytasks only (default production6–10 unchanged).
Loaded experiments predate latest per-agent gradient-contribution diagnostic addition; frozen native probe uses it. Loaded initial A/F processes predate bootstrap_fraction correction; report generator recomputes exact episode-based fractions from update/episode logs. Don’t claim old logs contained peragent/discounted rewards; these require new instrumented rollouts.
