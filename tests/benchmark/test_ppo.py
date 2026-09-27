import math

import numpy as np
import pytest
import torch

from bcod_sim.benchmark.ppo import PPOPolicy, gae, initialize_case_a, ppo_update


def test_gae_terminal_zero_bootstrap_and_no_cross_episode_credit():
    advantage, returns = gae([1., 2., 10.], [0., 0., 0.], [0., 99., 0.],
                             [False, True, False], [False, False, False], .9, .95)
    # The second transition is terminal, so the next episode's 10 cannot leak in.
    assert advantage.tolist() == pytest.approx([1 + .9*.95*2, 2, 10])
    assert returns.tolist() == pytest.approx(advantage.tolist())


def test_gae_timeout_bootstraps_but_does_not_cross_reset():
    advantage, _ = gae([1., 2., 10.], [0., 0., 0.], [0., 3., 0.],
                       [False, False, False], [False, True, False], .9, .95)
    assert advantage.tolist() == pytest.approx([1 + .9*.95*(2 + .9*3), 2 + .9*3, 10])


def test_gae_rollout_boundary_bootstraps():
    advantage, _ = gae([1., 2.], [0., 0.], [0., 4.],
                       [False, False], [False, False], .9, .95)
    assert advantage.tolist() == pytest.approx([1 + .9*.95*(2 + .9*4), 2 + .9*4])


def test_transformed_policy_range_density_and_initialization():
    policy = PPOPolicy()
    initialize_case_a(policy, np.zeros(47, dtype=np.float32))
    obs = torch.zeros((1, 47))
    mean = policy.mean(obs)
    action = policy.transform(mean)[0]
    assert action.tolist() == pytest.approx([.05, 0.], abs=1e-6)
    assert policy.log_std.exp().tolist() == pytest.approx([math.exp(-.7), .1], rel=1e-6)
    np.testing.assert_allclose(policy.transform(torch.tensor(
        [[-20., -20.], [0., 0.], [20., 20.]])).detach().numpy(),
        [[0., -1.], [.5, 0.], [1., 1.]], atol=1e-6)
    # Numerical change-of-variables check in the non-saturated interior.
    latent = torch.tensor([[.3, -.2]])
    normal = torch.distributions.Normal(policy.mean(obs), policy.log_std.exp())
    eps = 1e-4
    def f(z): return policy.transform(z)[0]
    j0 = (f(latent + torch.tensor([[eps, 0.]]))[0] - f(latent - torch.tensor([[eps, 0.]]))[0]) / (2*eps)
    j1 = (f(latent + torch.tensor([[0., eps]]))[1] - f(latent - torch.tensor([[0., eps]]))[1]) / (2*eps)
    expected = normal.log_prob(latent).sum(-1) - torch.log(j0*j1)
    assert policy.log_probability(obs, latent).item() == pytest.approx(expected.item(), abs=1e-3)


def test_ppo_update_keeps_actor_critic_parameters_separate():
    torch.manual_seed(11)
    policy = PPOPolicy()
    actor_parameters = list(policy.actor.parameters()) + [policy.log_std]
    critic_parameters = list(policy.critic.parameters())
    assert not set(map(id, actor_parameters)) & set(map(id, critic_parameters))
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=3e-4)
    observations = torch.randn(16, 47)
    with torch.no_grad():
        _, latent, old_logp = policy.sample(observations)
    result = ppo_update(policy, actor_optimizer, critic_optimizer,
                        (observations, latent, old_logp, torch.ones(16), torch.arange(16.).float()),
                        epochs=2, minibatch_size=8, clip_epsilon=.2,
                        entropy_coefficient=.01, value_coefficient=.5, max_grad_norm=.5)
    assert math.isfinite(result["policy_loss"])
    assert math.isfinite(result["value_loss"])
    assert result["actor_gradient_norm_after_clipping"] <= .5
    assert result["critic_gradient_norm_after_clipping"] <= .5
