"""Experiment 2: Does the ν(t) regularizer break the alpha-collapse?

Pilot finding (feasibility_check.py): naive learned-alpha collapsed to ~1.0
everywhere (mean=0.997, std=0.001) — the loss is degenerate.

Hypothesis: regularize alpha with a penalty proportional to the noise-prediction
effective gain ν(t) ∝ 1/b(t)². This punishes alpha→1 (noise-pred) where
sampling is unstable, breaking the flat-minimum problem.

Test: train at three eta_reg values, inspect alpha(x, t) heatmap. Want to see
spatial variation, not collapse.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import gon_toolkit as G

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


def alpha_heatmap(model, lim=1.5, n=48, t_value=0.3):
    """Render alpha(x, t) on a 2D grid at a fixed t."""
    g = np.linspace(-lim, lim, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    pts = torch.from_numpy(
        np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    )
    t = torch.full((pts.shape[0], 1), t_value, dtype=torch.float32)
    with torch.no_grad():
        _, alpha = model(pts, t)
    return XX, YY, alpha.squeeze().numpy().reshape(n, n)


X = G.make_circles(n=200, seed=SEED)

results = []
eta_values = [0.0, 1.0, 5.0, 20.0]

for eta in eta_values:
    print(f"\nTraining learned-alpha with eta_reg={eta} ...")
    t0 = time.perf_counter()
    model, mse_losses, reg_losses = G.train_alpha_mulan(
        X, dim=2, steps=2500, eta_reg=eta, seed=SEED
    )
    elapsed = time.perf_counter() - t0

    # Inspect alpha at t=0.3 (mid noise) and t=0.05 (near manifold)
    XX, YY, A_mid = alpha_heatmap(model, lim=1.5, n=40, t_value=0.3)
    _, _, A_near = alpha_heatmap(model, lim=1.5, n=40, t_value=0.05)
    _, _, A_far = alpha_heatmap(model, lim=1.5, n=40, t_value=0.9)

    print(f"  done in {elapsed:.1f}s, final MSE={mse_losses[-1]:.4f}, "
          f"final ν-reg={reg_losses[-1]:.4f}")
    print(f"  α at t=0.05 (near manifold): mean={A_near.mean():.3f}, "
          f"std={A_near.std():.3f}, range=[{A_near.min():.3f}, {A_near.max():.3f}]")
    print(f"  α at t=0.30 (mid):           mean={A_mid.mean():.3f}, "
          f"std={A_mid.std():.3f}, range=[{A_mid.min():.3f}, {A_mid.max():.3f}]")
    print(f"  α at t=0.90 (far / noise):   mean={A_far.mean():.3f}, "
          f"std={A_far.std():.3f}, range=[{A_far.min():.3f}, {A_far.max():.3f}]")

    results.append((eta, A_near, A_mid, A_far, XX, YY))


# ----- Figure: 4 rows (eta values) × 3 columns (t levels) -----
fig, axes = plt.subplots(len(eta_values), 3, figsize=(11, 3.2 * len(eta_values)),
                          constrained_layout=True)

for row, (eta, A_near, A_mid, A_far, XX, YY) in enumerate(results):
    panels = [
        (A_near, f"η={eta}, t=0.05 (near manifold)"),
        (A_mid,  f"η={eta}, t=0.30 (mid)"),
        (A_far,  f"η={eta}, t=0.90 (far / pure noise)"),
    ]
    for col, (A, title) in enumerate(panels):
        ax = axes[row, col]
        im = ax.imshow(A, cmap="RdBu_r", vmin=0.0, vmax=1.0,
                       extent=[-1.5, 1.5, -1.5, 1.5], origin="lower")
        ax.scatter(X[:, 0], X[:, 1], s=1, c="black", alpha=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02,
                     label=r"$\alpha(x, t)$  (0=velocity, 1=noise-pred)")
cbar.outline.set_visible(False)
fig.suptitle("Learned α(x, t) under increasing ν(t) regularization",
              fontsize=12, y=1.005)

fig.savefig("exp02_mulan.png", dpi=150, bbox_inches="tight")
print("\nSaved exp02_mulan.png")
