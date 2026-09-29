import pytest
import torch
from bcod_sim.benchmark_v3.deadline_harness.algorithms import MAPPO,TD3Actor
from bcod_sim.benchmark_v3.deadline_harness.__main__ import marl_validation,sarl_validation

def test_cpu_checkpoint_roundtrip(tmp_path):
    for cls in (MAPPO,TD3Actor):
        model=cls().cpu();p=tmp_path/(cls.__name__+'.pt');torch.save(model.state_dict(),p)
        loaded=cls().cpu();loaded.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
        assert all(torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters()))

def test_device_forward_backward_cpu():
    for cls in (MAPPO,TD3Actor):
        model=cls().cpu();x=torch.zeros((2,model.obs_mean.numel()))
        y=model.mean_action(x) if cls is MAPPO else model(x)
        y.sum().backward()
        assert all(p.device.type=='cpu' for p in model.parameters())

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_checkpoint_loads_on_cpu(tmp_path):
    model=MAPPO().cuda();p=tmp_path/'cuda.pt';torch.save(model.state_dict(),p)
    cpu=MAPPO();cpu.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
    assert all(p.device.type=='cpu' for p in cpu.parameters())
