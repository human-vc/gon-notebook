"""Experiment 1: Curl test.

Train a velocity-prediction MLP and a noise-prediction MLP on concentric circles
projected to D=8. Render the curl ∇×f for each on a 2D grid (project pts back to 8D
via the same random orthogonal P, evaluate model, project back to 2D).

If the paper's gradient-flow theorem holds:
  - velocity field should be (approximately) curl-free
  - noise-prediction field should have visible curl
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import gon_toolkit as G


# Reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----- Data -----
X2 = G.make_circles(n=200, r_inner=0.4, r_outer=0.9, noise=0.02, seed=SEED)

# We keep the experiment in 2D (no random projection) for the curl visualization
# because we want the field to live in the plane we're plotting.
# This still tests the paper's curl-free claim — the claim doesn't require D > 2.
print("Dataset:", X2.shape)


# ----- Train both parameterizations -----
print("\nTraining velocity model (alpha=0)...")
t0 = time.perf_counter()
v_model, v_losses = G.train_param(X2, alpha=0.0, dim=2, steps=2500, seed=SEED)
print(f"  done in {time.perf_counter() - t0:.1f}s, final MSE = {v_losses[-1]:.4f}")

print("\nTraining noise-prediction model (alpha=1)...")
t0 = time.perf_counter()
e_model, e_losses = G.train_param(X2, alpha=1.0, dim=2, steps=2500, seed=SEED)
print(f"  done in {time.perf_counter() - t0:.1f}s, final MSE = {e_losses[-1]:.4f}")


# ----- Compute curl on a grid -----
v_field = G.model_to_field(v_model)
e_field = G.model_to_field(e_model)

XX, YY, curl_v, Uv, Vv = G.curl_2d(v_field, lim=1.5, n=48)
_, _, curl_e, Ue, Ve = G.curl_2d(e_field, lim=1.5, n=48)

print(f"\nVelocity field curl: |mean|={np.abs(curl_v).mean():.4f}, "
      f"|max|={np.abs(curl_v).max():.4f}, std={curl_v.std():.4f}")
print(f"Noise-pred field curl: |mean|={np.abs(curl_e).mean():.4f}, "
      f"|max|={np.abs(curl_e).max():.4f}, std={curl_e.std():.4f}")
print(f"Ratio (eps / v): {np.abs(curl_e).mean() / (np.abs(curl_v).mean() + 1e-9):.2f}x")


# ----- Render -----
vmax = max(np.abs(curl_v).max(), np.abs(curl_e).max())
vmax = float(np.percentile(np.abs(np.concatenate([curl_v.ravel(), curl_e.ravel()])), 99))

fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax, curl, title in zip(axes, [curl_v, curl_e],
                            ["Velocity (v) target — should be curl-free",
                             "Noise (ε) target — should NOT be"]):
    im = ax.imshow(curl, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[-1.5, 1.5, -1.5, 1.5], origin="lower")
    ax.scatter(X2[:, 0], X2[:, 1], s=2, c="black", alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label=r"$\nabla \times f$")
cbar.outline.set_visible(False)

fig.savefig("exp01_curl.png", dpi=150, bbox_inches="tight")
print("\nSaved exp01_curl.png")
