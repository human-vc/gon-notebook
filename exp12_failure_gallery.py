"""Experiment 12: Failure-mode gallery — phase diagram on three datasets.

Re-runs the (α, D) phase diagram (Exp 5) on:
  - moons        (smooth manifold, semicircular)
  - 8-GMM ring   (disconnected, multimodal)
  - swiss roll   (curved, single-component)

Tests whether the velocity-wins-at-low-D finding generalizes across geometry.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import gon_toolkit as G

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


def train_and_w2(X, alpha, D, X_truth_2d, n_samples=400, train_steps=3000, seed=0):
    model, _ = G.train_param(X, alpha=alpha, dim=D, steps=train_steps,
                              batch=128, lr=1e-3, seed=seed,
                              hidden=128, depth=3)
    samples = G.sample_ode(model, n=n_samples, dim=D, n_steps=250,
                            sigma_init=1.0, seed=seed)
    # Project the fresh truth (n_samples points) up using the SAME projection
    X_truth_D = G.project_up(X_truth_2d, D, seed=seed)
    return G.sliced_w2(samples, X_truth_D, n_proj=128)


datasets = [
    ("moons",      G.make_moons(n=300, noise=0.05, seed=SEED),
                    G.make_moons(n=400, noise=0.05, seed=SEED + 100)),
    ("8-GMM ring", G.make_gmm_ring(n=300, k=8, r=0.7, sigma=0.04, seed=SEED),
                    G.make_gmm_ring(n=400, k=8, r=0.7, sigma=0.04, seed=SEED + 100)),
    ("swiss",      G.make_swiss_2d(n=300, noise=0.05, seed=SEED),
                    G.make_swiss_2d(n=400, noise=0.05, seed=SEED + 100)),
]

alpha_vals = [0.0, 0.5, 1.0]
D_vals = [2, 8, 32]

print(f"Sweeping {len(alpha_vals)} α × {len(D_vals)} D = "
      f"{len(alpha_vals) * len(D_vals)} runs × {len(datasets)} datasets "
      f"= {len(alpha_vals) * len(D_vals) * len(datasets)} total")

t_start = time.perf_counter()
all_grids = {}
for ds_name, X, X_truth in datasets:
    print(f"\n{ds_name}:")
    grid = np.zeros((len(alpha_vals), len(D_vals)))
    for i, alpha in enumerate(alpha_vals):
        for j, D in enumerate(D_vals):
            t0 = time.perf_counter()
            w2 = train_and_w2(X, alpha=alpha, D=D, X_truth_2d=X_truth,
                                train_steps=3000, seed=SEED)
            grid[i, j] = w2
            print(f"  α={alpha:.2f}, D={D:3d}: W2={w2:.3f}  "
                  f"({time.perf_counter() - t0:.1f}s)")
    all_grids[ds_name] = grid

print(f"\nTotal: {time.perf_counter() - t_start:.1f}s")


# ----- Render: 1 × N_datasets heatmaps -----
fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4.0),
                          constrained_layout=True)

vmax = max(g.max() for g in all_grids.values())
vmin = min(g.min() for g in all_grids.values())

for ax, (ds_name, X, _) in zip(axes, datasets):
    grid = all_grids[ds_name]
    im = ax.imshow(grid.T, cmap="viridis_r", aspect="auto", origin="lower",
                    extent=[-0.5, len(alpha_vals) - 0.5, -0.5, len(D_vals) - 0.5],
                    vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(alpha_vals)))
    ax.set_xticklabels([f"{a:.1f}" for a in alpha_vals])
    ax.set_yticks(range(len(D_vals)))
    ax.set_yticklabels(D_vals)
    ax.set_xlabel(r"α  (0 = velocity, 1 = noise)")
    if ax is axes[0]:
        ax.set_ylabel("ambient dim D")
    ax.set_title(ds_name, fontsize=11)
    for i in range(len(alpha_vals)):
        for j in range(len(D_vals)):
            v = grid[i, j]
            color = "white" if v > (vmin + vmax) / 2 else "black"
            ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02,
                     label=r"$W_2$ to ground truth  (lower better)")
cbar.outline.set_visible(False)
fig.suptitle("Failure-mode gallery — phase diagram across three 2D toy datasets",
              fontsize=11.5, y=1.04)
fig.savefig("exp12_failure_gallery.png", dpi=150, bbox_inches="tight")
print("Saved exp12_failure_gallery.png")

np.savez("exp12_failure_grids.npz",
          **{k: v for k, v in all_grids.items()},
          alpha_vals=alpha_vals, D_vals=D_vals)
print("Saved exp12_failure_grids.npz")
