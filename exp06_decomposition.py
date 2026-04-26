"""Experiment 6: Three-component decomposition (Eq 14) of the closed-form
optimal autonomous field. Renders Natural Gradient, Transport Correction,
Linear Drift as separate quiver plots.
"""

import numpy as np
import matplotlib.pyplot as plt

import gon_toolkit as G

X = G.make_circles(n=120, seed=0)
t_grid = np.linspace(0.05, 0.95, 18)

print("Computing three-component decomposition on a 16x16 grid...")
import time
t0 = time.perf_counter()
XX, YY, natural, transport, drift = G.decompose_field_grid(
    X, t_grid, sched_fn=G.fm_schedule, lim=1.4, n=16
)
print(f"  done in {time.perf_counter() - t0:.1f}s")

# Total field for reference
total = natural + transport + drift

# Magnitudes
n_mag = np.linalg.norm(natural, axis=-1)
t_mag = np.linalg.norm(transport, axis=-1)
d_mag = np.linalg.norm(drift, axis=-1)

print(f"\nComponent magnitudes:")
print(f"  Natural Gradient: mean={n_mag.mean():.3f}, max={n_mag.max():.3f}")
print(f"  Transport Correction: mean={t_mag.mean():.3f}, max={t_mag.max():.3f}")
print(f"  Linear Drift: mean={d_mag.mean():.3f}, max={d_mag.max():.3f}")


# ----- Render: 1×3 panel -----
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5),
                          sharex=True, sharey=True, constrained_layout=True)

components = [
    ("Natural Gradient",   natural,    G.STABLE),
    ("Transport Correction", transport, G.ACCENT),
    ("Linear Drift",       drift,     G.UNSTABLE),
]

# Use shared scale for visual comparability
max_mag = max(n_mag.max(), t_mag.max(), d_mag.max())

for ax, (name, vec, color) in zip(axes, components):
    U = vec[..., 0]; V = vec[..., 1]
    M = np.linalg.norm(vec, axis=-1)
    q = ax.quiver(XX, YY, U, V, M,
                  cmap="viridis",
                  scale_units="xy", scale=max_mag * 1.5,
                  width=0.005, headwidth=4, headlength=5,
                  pivot="mid", alpha=0.85)
    ax.scatter(X[:, 0], X[:, 1], s=4, c=G.DATA, alpha=0.7, zorder=5)
    ax.set_title(name, fontsize=11, color=color)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Eq 14: f*(u) decomposes into three geometric components",
              fontsize=12, y=1.04)
fig.savefig("exp06_decomposition.png", dpi=150, bbox_inches="tight")
print("Saved exp06_decomposition.png")
