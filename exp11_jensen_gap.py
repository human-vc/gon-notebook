"""Experiment 11: The Jensen Gap heatmap (Eq 66).

Paper introduces the "Jensen Gap" as the source of noise-prediction instability
(Sec 6, Eq 66 in App F.2) — and never plots it.

  Δv_noise ∝ ‖u - x*‖ · |b'(t)/b(t)| · |b(t) · E_{τ|u}[1/b(τ)] - 1|

Render at three values of t to show the gap explodes as t → 0.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import gon_toolkit as G

X = G.make_circles(n=120, seed=0)
t_grid = np.linspace(0.02, 0.95, 22)

t_evals = [0.05, 0.20, 0.60]

print("Computing Jensen Gap on 36×36 grid at three noise levels...")
results = []
for t_eval in t_evals:
    t0 = time.perf_counter()
    XX, YY, gap = G.jensen_gap_grid(X, t_grid, sched_fn=G.fm_schedule,
                                      t_eval=t_eval, lim=1.5, n=36)
    elapsed = time.perf_counter() - t0
    print(f"  t={t_eval}:  mean={gap.mean():.2f}  max={gap.max():.2f}  "
          f"99%ile={np.percentile(gap, 99):.2f}  ({elapsed:.1f}s)")
    results.append((t_eval, gap))


# ----- Render -----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3),
                          sharey=True, constrained_layout=True)

vmax = max(np.percentile(g, 98) for _, g in results)
for ax, (t_eval, gap) in zip(axes, results):
    im = ax.imshow(np.clip(gap, 0, vmax), cmap="magma_r",
                    extent=[-1.5, 1.5, -1.5, 1.5], origin="lower",
                    vmin=0, vmax=vmax)
    ax.scatter(X[:, 0], X[:, 1], s=4, c="cyan", alpha=0.8)
    ax.set_title(f"t = {t_eval}", fontsize=11)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02,
                     label=r"$\Delta v_{\mathrm{noise}}$  (Jensen Gap, Eq 66)")
cbar.outline.set_visible(False)
fig.suptitle("Jensen Gap heatmap — noise-pred instability source visualized",
              fontsize=12, y=1.04)
fig.savefig("exp11_jensen_gap.png", dpi=150, bbox_inches="tight")
print("Saved exp11_jensen_gap.png")
