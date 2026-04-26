"""Experiment 4 (the headline): Cracking D=2 with stochastic samplers.

The paper says D=2 is structurally hard for autonomous deterministic sampling
because Lemma 5's proximity concentration argument requires D > 2.

Hypothesis: this is a property of the *deterministic ODE sampler*, not of the
trained field itself. SDE / Langevin samplers integrate over the ambiguous
posterior and should produce clean samples on the same trained field.

Test: train one v-MLP on 2D circles at D=2. Sample 1000 points three ways:
  - Euler ODE (deterministic, paper's failure case)
  - Euler-Maruyama SDE (stochastic, with decreasing noise schedule)
  - Annealed Langevin (Markov chain with score-like updates)
Measure W2 vs ground truth circles. Plot 3-panel comparison.

If SDE/Langevin clearly beat ODE, we have the headline.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import gon_toolkit as G

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


# ----- Data: 2D concentric circles, D=2 (no embedding) -----
X = G.make_circles(n=400, r_inner=0.4, r_outer=0.9, noise=0.02, seed=SEED)
print(f"Data: {X.shape} at D=2")


# ----- Train velocity field -----
print("\nTraining v-pred MLP at D=2 (bigger network, longer training)...")
t0 = time.perf_counter()
v_model, losses = G.train_param(X, alpha=0.0, dim=2, steps=15000, batch=128,
                                  lr=1e-3, seed=SEED, hidden=128, depth=4)
print(f"  done in {time.perf_counter() - t0:.1f}s, final MSE={losses[-1]:.4f}")


# ----- Sample three ways -----
N = 1000

print("\nSampling...")
t0 = time.perf_counter()
samples_ode = G.sample_ode(v_model, n=N, dim=2, n_steps=400, sigma_init=1.0, seed=SEED)
print(f"  Euler ODE: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
samples_sde = G.sample_sde(v_model, n=N, dim=2, n_steps=400, sigma_init=1.0,
                            noise_scale=0.6, seed=SEED)
print(f"  Euler-Maruyama SDE: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
samples_lang = G.sample_langevin(v_model, n=N, dim=2, n_steps=1500, sigma_init=1.0,
                                  step_size=0.003, seed=SEED)
print(f"  Annealed Langevin: {time.perf_counter() - t0:.1f}s")


# ----- Ground truth: many fresh samples from the data distribution -----
X_truth = G.make_circles(n=N, r_inner=0.4, r_outer=0.9, noise=0.02, seed=SEED + 100)


# ----- Wasserstein-2 -----
X_truth2 = G.make_circles(n=N, r_inner=0.4, r_outer=0.9, noise=0.02, seed=SEED + 200)
w2_ode = G.sliced_w2(samples_ode, X_truth, n_proj=128)
w2_sde = G.sliced_w2(samples_sde, X_truth, n_proj=128)
w2_lang = G.sliced_w2(samples_lang, X_truth, n_proj=128)
w2_baseline = G.sliced_w2(X_truth2, X_truth, n_proj=128)

print(f"\nWasserstein-2 vs ground truth:")
print(f"  Baseline (data vs data): {w2_baseline:.3f}")
print(f"  Euler ODE:               {w2_ode:.3f}")
print(f"  Euler-Maruyama SDE:      {w2_sde:.3f}")
print(f"  Annealed Langevin:       {w2_lang:.3f}")


# ----- Render -----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True,
                          sharex=True, sharey=True)

LIM = 2.0
panels = [
    ("Euler ODE  (deterministic)",  samples_ode,  w2_ode,
        r"$dx = v(x)\, dt$"),
    ("Euler-Maruyama SDE",          samples_sde,  w2_sde,
        r"$dx = v(x)\, dt + \sigma(t)\, dW$"),
    ("Annealed Langevin",           samples_lang, w2_lang,
        r"$x \leftarrow x - \eta\, v(x) + \sqrt{2\eta}\, \xi$"),
]

for ax, (title, samples, w2, eq) in zip(axes, panels):
    ax.scatter(X_truth[:, 0], X_truth[:, 1], s=4, c="lightgray",
                alpha=0.7, label="data")
    ax.scatter(samples[:, 0], samples[:, 1], s=4, c="crimson", alpha=0.55,
                label="generated")
    ax.set_title(title, fontsize=11)
    ax.text(0.04, 0.96, eq, transform=ax.transAxes, va="top", fontsize=9,
             bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=2))
    ax.text(0.96, 0.04, f"$W_2$ = {w2:.3f}", transform=ax.transAxes,
             ha="right", fontsize=10,
             bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=2))
    ax.set_aspect("equal")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_xticks([]); ax.set_yticks([])

axes[0].legend(loc="lower right", fontsize=9, frameon=False)

fig.suptitle(
    "Cracking D=2: same trained v-field, three samplers",
    fontsize=12, y=1.04
)
fig.savefig("exp04_cracking_d2.png", dpi=150, bbox_inches="tight")
print("\nSaved exp04_cracking_d2.png")
