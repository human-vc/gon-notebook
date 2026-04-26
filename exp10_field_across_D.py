"""Experiment 10: Closed-form autonomous field across D ∈ {2, 8, 32, 128}.

Reproduces paper Fig 5 with the analytical f*(u) — no training. Demonstrates
the "transport correction vanishes" story: at D=2 the field is messy, at D=128
it collapses into a clean radial-flow toward the manifold.

For each D:
  - project the 2D circles into ℝ^D via P (D × 2 random orthogonal columns)
  - evaluate f*(u) on a grid IN the data plane: u = P @ [x, y]
  - project the field back to 2D for visualization: P^T @ f*(P @ [x, y])
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import gon_toolkit as G

X2 = G.make_circles(n=80, seed=0)
t_grid = np.linspace(0.05, 0.95, 16)
D_values = [2, 8, 32, 128]

print(f"Evaluating closed-form f*(u) at D ∈ {D_values}, 18×18 grid each...")

results = []
for D in D_values:
    t0 = time.perf_counter()
    # Project up to ℝ^D
    rng = np.random.default_rng(0)
    M = rng.standard_normal((D, 2)).astype(np.float32)
    P, _ = np.linalg.qr(M)  # D × 2 orthonormal cols
    XD = (X2 @ P.T).astype(np.float32)

    # Evaluate on a 2D slice that sits inside the data plane
    n = 18
    g = np.linspace(-1.4, 1.4, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    UV = np.zeros((n, n, 2), dtype=np.float32)
    for i, x in enumerate(g):
        for j, y in enumerate(g):
            u_D = (P @ np.array([x, y], dtype=np.float32))  # back into ℝ^D
            f_D = G.f_star_discrete(u_D, XD, t_grid, G.fm_schedule)
            UV[j, i] = P.T @ f_D  # project field back to 2D
    elapsed = time.perf_counter() - t0
    print(f"  D={D:3d}: |f|_mean={np.linalg.norm(UV, axis=-1).mean():.3f}  "
          f"|f|_max={np.linalg.norm(UV, axis=-1).max():.3f}  ({elapsed:.1f}s)")
    results.append((D, XX, YY, UV))


# ----- Render: 1×4 quiver panel -----
fig, axes = plt.subplots(1, 4, figsize=(15, 4.0),
                          sharex=True, sharey=True, constrained_layout=True)

for ax, (D, XX, YY, UV) in zip(axes, results):
    U = UV[..., 0]; V = UV[..., 1]
    M = np.linalg.norm(UV, axis=-1)
    ax.quiver(XX, YY, U, V, M,
                cmap="viridis", scale_units="xy", scale=14,
                width=0.005, headwidth=4, headlength=5,
                pivot="mid", alpha=0.85)
    ax.scatter(X2[:, 0], X2[:, 1], s=6, c="crimson", alpha=0.7, zorder=5)
    ax.set_title(f"D = {D}", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Closed-form $f^*(u)$ across ambient dimension D — transport correction vanishes as D grows",
              fontsize=11.5, y=1.04)
fig.savefig("exp10_field_across_D.png", dpi=150, bbox_inches="tight")
print("Saved exp10_field_across_D.png")
