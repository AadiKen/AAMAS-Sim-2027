"""Case-A CLI dispatch accepts the selected update interval without launching BCOD."""
import sys

from bcod_sim.benchmark import audit_learning


def test_case_a_cli_passes_update_interval_to_existing_train_mode(monkeypatch, tmp_path):
    captured=[]
    monkeypatch.setattr(audit_learning,'train',lambda args:captured.append(args))
    monkeypatch.setattr(sys,'argv',['audit_learning','train','--sim','bcod-reduced',
        '--kind','A','--steps','5000','--update-interval','10','--seed','11',
        '--output',str(tmp_path/'case-a')])
    audit_learning.main()
    assert len(captured)==1
    args=captured[0]
    assert (args.sim,args.kind,args.steps,args.update_interval,args.seed)==(
        'bcod-reduced','A',5000,10,11)
    assert args.update_mode=='global' and not args.surge_only and not args.actor_only


def test_case_a_cli_accepts_diagnostic_initial_yaw_std(monkeypatch, tmp_path):
    captured=[]
    monkeypatch.setattr(audit_learning,'train',lambda args:captured.append(args))
    monkeypatch.setattr(sys,'argv',['audit_learning','train','--sim','bcod-reduced',
        '--kind','A','--initial-yaw-raw-std','0.10','--output',str(tmp_path/'case-a')])
    audit_learning.main()
    assert len(captured)==1
    assert captured[0].initial_yaw_raw_std==0.10
    assert not captured[0].surge_only
