"""Optional ETA display on the shared training and evaluation commands."""

import json

from bcod_sim.benchmark.runner import main


def test_train_and_evaluate_eta_bar(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_steps": 2}))
    checkpoints = tmp_path / "checkpoints"
    main(["train", "--sim", "bcod", "--total-steps", "2",
          "--checkpoint-interval", "1", "--checkpoint-dir", str(checkpoints),
          "--config", str(config), "--eta-bar"])
    stderr = capsys.readouterr().err
    assert "bcod train" in stderr and "2/2 ETA 00:00:00" in stderr
    assert json.loads((checkpoints / "run-config.json").read_text())["eta_bar"] is True
    assert (checkpoints / "checkpoint-000000002.pt").is_file()

    output = tmp_path / "evaluation.json"
    main(["evaluate", "--sim", "bcod", "--checkpoint",
          str(checkpoints / "checkpoint-000000002.pt"), "--episodes", "1",
          "--output", str(output), "--eta-bar"])
    stderr = capsys.readouterr().err
    assert "bcod evaluate" in stderr and "1/1 ETA 00:00:00" in stderr
    assert len(json.loads(output.read_text())["episodes"]) == 1


def test_eta_bar_is_optional(tmp_path, capsys):
    checkpoints = tmp_path / "checkpoints"
    main(["train", "--sim", "bcod", "--total-steps", "1",
          "--checkpoint-interval", "1", "--checkpoint-dir", str(checkpoints)])
    assert "ETA" not in capsys.readouterr().err
    assert json.loads((checkpoints / "run-config.json").read_text())["eta_bar"] is False
