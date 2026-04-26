"""Experiment 5: (alpha, D) phase diagram colored by sample quality.

Goh's α-β plane analog. The paper analyzes parameterizations as a binary table
(Table 2). We fill in the interior — continuous α from 0 (velocity) to 1 (noise),
across D from 2 to 64. Color by Wasserstein-2 of generated samples vs ground truth.

This becomes the iconic figure.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import gon_toolkit as G

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


def train_and_sample(X, alpha, D, n_samples=500, train_steps=8000, seed=0):
    """Train at given (alpha, D), sample, return W2 against ground truth at this D."""
    model, _ = G.train_param(X, alpha=alpha, dim=D, steps=train_steps,
                              batch=128, lr=1e-3, seed=seed,
                              hidden=128, depth=3)
    samples = G.sample_ode(model, n=n_samples, dim=D, n_steps=300,
                            sigma_init=1.0, seed=seed)
    # Ground truth at this D
    X_truth = G.make_circles(n=n_samples, seed=seed + 100)
    X_truth_D = G.project_up(X_truth, D, seed=seed)  # use the SAME projection
    w2 = G.sliced_w2(samples, X_truth_D, n_proj=128)
    return w2, samples


X = G.make_circles(n=300, seed=SEED)

alpha_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
D_vals = [2, 4, 8, 16, 32, 64]

print(f"Sweeping {len(alpha_vals)} α × {len(D_vals)} D = {len(alpha_vals) * len(D_vals)} runs")
print(f"Estimated time: ~{len(alpha_vals) * len(D_vals) * 6} sec")

W2_grid = np.zeros((len(alpha_vals), len(D_vals)))
t_start = time.perf_counter()

for i, alpha in enumerate(alpha_vals):
    for j, D in enumerate(D_vals):
        t0 = time.perf_counter()
        w2, _ = train_and_sample(X, alpha=alpha, D=D, train_steps=4000, seed=SEED)
        W2_grid[i, j] = w2
        elapsed = time.perf_counter() - t0
        print(f"  α={alpha:.2f}, D={D:3d}: W2={w2:.3f}  ({elapsed:.1f}s)")

print(f"\nTotal: {time.perf_counter() - t_start:.1f}s")

# Save grid for reuse
np.savez("exp05_phase_grid.npz",
          W2_grid=W2_grid, alpha_vals=alpha_vals, D_vals=D_vals)
print("Saved exp05_phase_grid.npz")


# ----- Render -----
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

# Use log scale for D on the y-axis via custom tick positions
W2_clipped = np.clip(W2_grid, 0, np.percentile(W2_grid, 95))
im = ax.imshow(W2_clipped.T, cmap="viridis_r", aspect="auto", origin="lower",
                extent=[-0.5, len(alpha_vals) - 0.5, -0.5, len(D_vals) - 0.5])
ax.set_xticks(range(len(alpha_vals)))
ax.set_xticklabels([f"{a:.2f}" for a in alpha_vals])
ax.set_yticks(range(len(D_vals)))
ax.set_yticklabels(D_vals)
ax.set_xlabel(r"parameterization $\alpha$  (0 = velocity, 1 = noise-pred)")
ax.set_ylabel(r"ambient dim $D$")

# Annotate W2 in each cell
for i in range(len(alpha_vals)):
    for j in range(len(D_vals)):
        v = W2_grid[i, j]
        color = "white" if v > np.percentile(W2_grid, 50) else "black"
        ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                fontsize=9, color=color)

cbar = fig.colorbar(im, ax=ax, label=r"$W_2$ to ground truth (lower better)")
cbar.outline.set_visible(False)
ax.set_title("Sample quality across the (α, D) parameterization plane",
              fontsize=11)
fig.savefig("exp05_phase_diagram.png", dpi=150, bbox_inches="tight")
print("Saved exp05_phase_diagram.png")
