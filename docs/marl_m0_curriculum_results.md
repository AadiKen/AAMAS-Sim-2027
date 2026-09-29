# M0 high-level MAPPO curriculum v1 results

**Classification: FAIL at the curriculum promotion gate.** No seed completed M0-A→M0-B→M0-C. No learned actor was evaluated on the final M0-C test bank. M1 must not begin. This is a training stability and coordination result, not evidence of a task/reward contradiction or a low-level physics failure.

The frozen recipe SHA-256 was `7613978a7295d438ee81d04808c1ebe8c6fd44d9092904d0208f431bba3d21f7` for all three seeds. BenchMARL 1.5.2 MAPPO/TorchRL 0.11.1 used the shared local actor, 89-field centralized critic, high-level speed/heading actions, and unchanged V3 task reward. The integration smoke ran 1,536 transitions with reduced evaluation counts and forced promotion gates; it traversed A→B→C in one experiment, saved stage-transition checkpoints, and produced finite losses. That smoke is not a learning result.

## Frozen development checkpoints

Every development bank has 50 scenarios. Selection uses fleet success, then collisions, deadlines, completion time, and path efficiency. Two consecutive passing evaluations were required for promotion.

| Seed | Stage reached | Final transitions | Best stage/step | Best dev success | Collisions | Deadlines | Final stage success | Final collisions | Final deadlines | Result |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 11 | A | 48,000 | A / 24,000 | 45/50 | 2 | 3 | 37/50 | 0 | 13 | A promotion failed |
| 22 | B | 96,000 | B / 54,000 | 43/50 | 7 | 0 | 34/50 | 16 | 0 | B promotion failed |
| 33 | A | 48,000 | A / 42,000 | 39/50 | 0 | 11 | 38/50 | 0 | 12 | A promotion failed |

Seed 11 reached the A success threshold once (45/50 at 24k), then fell to 23/50 at 30k; it never achieved two consecutive qualifying evaluations. Seed 33 peaked at 39/50. Seed 22 achieved 50/50 and 49/50 at the final two A evaluations, so it promoted to B at 48k. Its first B evaluation scored 43/50 but had 7/50 collisions, above the declared 10% ceiling. B then regressed to 34/50 and 16/50 collisions by 96k. No B evaluation met both gates twice.

## Failure types and coordination diagnostics

| Seed / selected dev checkpoint | One-agent failures | Both-agent failures | Deadlock episodes | Mean speed command | Mean absolute relative-heading action | Controller saturation |
|---|---:|---:|---:|---:|---:|---:|
| 11 / A24k | 4 | 1 | 0 | 1.03 m/s | 0.06 | 8.1% |
| 22 / B54k | 5 | 2 | 0 | 1.35 m/s | 0.06 | 3.3% |
| 33 / A42k | 11 | 0 | 0 | 0.70 m/s | 0.13 | 12.2% |

At the declared failure checkpoints, seeds 11 and 33 were deadline dominated (13 and 12 deadlines); seed 22 was collision dominated (16 collisions). The diagnostic deadlock criterion—both unreached vessels below 0.15 m/s with negligible progress for at least 5 s—triggered on none of the 50 development cases at those checkpoints. Thus the deadlines are not explained by both vessels simply stopping. Seed 22's B-stage mean commanded speed rose to 1.51 m/s while collisions rose, consistent with a more aggressive but poorly coordinated policy. This is an observation from the traces and metrics, not a proven causal mechanism.

The geometric controller, through the same high-level action path, succeeded on A-dev 50/50 and B-dev 40/50; all fixed geometries passed the scenario validator. This supports basic task feasibility. Each seed's untrained actor scored 0/50 on A-dev. A separate read-only evaluation of those step-zero actors on C-dev also scored 0/50 in all seeds (collisions/deadlines: 50/0, 19/31, 50/0). Reward decomposition and physical colliders are retained per evaluation episode, and short per-agent traces are saved under each run directory.

## Test discipline and next gate

The new C-test bank contains 100 unique geometries and has hash `e94e660a9abff6c65d62c34187e4b1b8faf6c8eda8ebbd18d93216d9293b6de3`. The geometric controller's 93/100 C-test result is the prescribed feasibility reference. **No learned C-test result exists**, because no actor reached the final stage. Best-dev files for stopped seeds are copies of their selected A/B checkpoints and are labelled with stage in `selection-summary.json`; they are not M0-C-qualified policies.

Per-run manifests, stage histories, dev evaluations, selected actor hashes, BenchMARL scalar logs, checkpoints, and trajectories are in `runs/benchmark-v3/marl-m0-curriculum-v1-seed{11,22,33}/`. The machine-readable aggregate is `artifacts/marl-m0-curriculum-v1/summary.json`. The installed BenchMARL `Experiment.state_dict()` does not serialize optimizer state. The live optimizer was retained through seed 22's A→B transition, but the saved stage-transition checkpoint is not claimed as an optimizer-continuous resume point.

An artifact-only cleanup after the runs copied each failed seed's selected A/B checkpoint to the conventional `best-dev` filenames and recorded its SHA-256 in `selection-summary.json`. The runner now performs that copy automatically for future stopped runs. This did not change training, development evaluation, checkpoint selection, or the frozen recipe.

**Recommendation:** Stop this recipe. Investigate the M0-A deadline/regression pattern and the M0-B collision increase with a new predeclared experiment version before any M1 or physical-backend MARL run. Do not reuse the C-test bank for recipe tuning.
