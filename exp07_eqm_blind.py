"""Experiment 7: EqM blind at D=2 — paper's untested third corner.

Paper's Table 1 includes Equilibrium Matching (EqM) with target = u - x_clean,
gain ν(t) ≈ t² (vanishes faster than velocity), drift coefficient = 0.
Theoretically EqM should be the MOST stable parameterization — even more so
than velocity. The paper analyzes it but never tests blind EqM at D=2.

Test: train EqM blind on circles at D=2 with 3 samplers, compare to v-pred.
Hypothesis: EqM blind clean rings at D=2 where v-pred only mostly worked.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import gon_toolkit as G

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


X = G.make_circles(n=400, seed=SEED)
N = 1000

# ----- Train both -----
print("Training v-pred (FM, alpha=0)...")
v_model, _ = G.train_param(X, alpha=0.0, dim=2, steps=15000, batch=128,
                            lr=1e-3, seed=SEED, hidden=128, depth=4)

print("Training EqM blind (target = u - x_clean)...")
eqm_model, _ = G.train_eqm(X, dim=2, steps=15000, batch=128, lr=1e-3, seed=SEED)


# ----- Sample both with ODE -----
@torch.no_grad()
def sample_eqm_ode(model, n=500, dim=2, n_steps=400, sigma_init=1.0, seed=0):
    """For EqM, the trained field is f*(u) = u - x*(u) = drift toward manifold.
    Sampling: u_{k+1} = u_k - dt * f(u_k) (gradient descent on energy).
    """
    torch.manual_seed(seed)
    x = sigma_init * torch.randn(n, dim)
    ts = np.linspace(1.0, 0.001, n_steps)
    for i in range(n_steps - 1):
        f = model(x)
        dt = ts[i + 1] - ts[i]
        x = x + dt * f
    return x.numpy()


@torch.no_grad()
def sample_eqm_langevin(model, n=500, dim=2, n_steps=1500, sigma_init=1.0,
                        step_size=0.005, seed=0):
    torch.manual_seed(seed)
    x = sigma_init * torch.randn(n, dim)
    for i in range(n_steps):
        score_like = -model(x)  # EqM: f = u - x_clean, so -f = x_clean - u points to manifold
        sigma = step_size * (1 - 0.5 * i / n_steps)
        x = x + step_size * score_like + np.sqrt(2 * sigma) * torch.randn_like(x)
    return x.numpy()


# v-pred ODE (baseline)
v_ode = G.sample_ode(v_model, n=N, dim=2, n_steps=400, sigma_init=1.0, seed=SEED)

# EqM ODE
eqm_ode = sample_eqm_ode(eqm_model, n=N, dim=2, n_steps=400, sigma_init=1.0, seed=SEED)

# EqM Langevin
eqm_lang = sample_eqm_langevin(eqm_model, n=N, dim=2, n_steps=1500,
                                sigma_init=1.0, step_size=0.005, seed=SEED)

# Ground truth
X_truth = G.make_circles(n=N, seed=SEED + 100)

w2_v = G.sliced_w2(v_ode, X_truth, n_proj=128)
w2_eqm = G.sliced_w2(eqm_ode, X_truth, n_proj=128)
w2_eqm_l = G.sliced_w2(eqm_lang, X_truth, n_proj=128)

print(f"\nW2 vs ground truth:")
print(f"  v-pred ODE:      {w2_v:.3f}")
print(f"  EqM ODE:         {w2_eqm:.3f}")
print(f"  EqM Langevin:    {w2_eqm_l:.3f}")


# ----- Render -----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                          sharex=True, sharey=True, constrained_layout=True)

panels = [
    ("v-pred (FM) + ODE",   v_ode,    w2_v,    r"$v_\theta = \mathbb{E}[\varepsilon - x | u]$"),
    ("EqM blind + ODE",     eqm_ode,  w2_eqm,  r"$f_\theta = \mathbb{E}[u - x | u]$"),
    ("EqM blind + Langevin", eqm_lang, w2_eqm_l, r"$x \leftarrow x + \eta(x_\theta - x) + \sqrt{2\eta}\xi$"),
]
for ax, (title, samples, w2, eq) in zip(axes, panels):
    ax.scatter(X_truth[:, 0], X_truth[:, 1], s=4, c="lightgray", alpha=0.7)
    ax.scatter(samples[:, 0], samples[:, 1], s=4, c="crimson", alpha=0.55)
    ax.set_title(title, fontsize=11)
    ax.text(0.04, 0.96, eq, transform=ax.transAxes, va="top", fontsize=8.5,
             bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=2))
    ax.text(0.96, 0.04, f"$W_2$ = {w2:.3f}", transform=ax.transAxes,
             ha="right", fontsize=10,
             bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=2))
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("EqM blind at D=2 — paper's untested third corner",
              fontsize=12, y=1.04)
fig.savefig("exp07_eqm_blind.png", dpi=150, bbox_inches="tight")
print("Saved exp07_eqm_blind.png")
