"""Headless entry point for the same local runner used by the web API."""

import argparse
import json
from pathlib import Path
import time

from bcod_sim.config.resolver import load_config
from bcod_sim.web.runner import LocalRunner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="manta-runner")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=Path.cwd()/".bcod_runs")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="serve web API and local runner on loopback")
    start.add_argument("--port", type=int, default=8000)
    run = commands.add_parser("run", help="submit a complete experiment without a browser")
    run.add_argument("experiment", type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--mode", choices=("scripted_waypoint", "policy_evaluation"), default="scripted_waypoint")
    run.add_argument("--policy-id")
    run.add_argument("--seed", type=int)
    run.add_argument("--wait", action="store_true")
    for name in ("status", "stop", "kill"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
    args = parser.parse_args(argv)
    if args.command == "start":
        import uvicorn
        from bcod_sim.web.api import create_app
        app = create_app(artifact_root=args.artifact_root, project_root=args.project_root,
                         frontend_dist=args.project_root/"web_client"/"dist")
        uvicorn.run(app, host="127.0.0.1", port=args.port)
        return 0
    runner = LocalRunner(args.artifact_root, args.project_root)
    if args.command == "run":
        document = load_config(args.experiment)
        if set(document) != {"config", "definitions"}:
            parser.error("experiment must contain config and definitions")
        result = runner.submit(args.run_id, document["config"], document["definitions"],
            mode=args.mode, seed=args.seed, policy_id=args.policy_id)
        if args.wait:
            while result["state"] in {"QUEUED", "RUNNING"}:
                time.sleep(.2)
                result = runner.status(args.run_id)
        print(json.dumps(result, indent=2))
        return 0 if result["state"] not in {"FAILED", "KILLED"} else 1
    result = getattr(runner, args.command)(args.run_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
