"""Experiment 9: The conformal metric — paper's central concept made visual.

Section 5.3 of the paper proves that:
  - ‖∇E_marg(u)‖ → ∞ as u → manifold     (Eq 12, the singularity)
  - λ̄(u) → 0 at exactly the rate that cancels it
  - Their product = Natural Gradient term, bounded near manifold

Paper proves this analytically. Never visualizes it. We do.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import gon_toolkit as G

t0 = time.perf_counter()

X = G.make_circles(n=120, seed=0)
t_grid = np.linspace(0.05, 0.95, 18)

print("Computing conformal metric on 40x40 grid...")
XX, YY, lam_bar, grad_norm, grad_field = G.conformal_grid(
    X, t_grid, sched_fn=G.fm_schedule, lim=1.5, n=40
)
print(f"  done in {time.perf_counter() - t0:.1f}s")

product = lam_bar * grad_norm

print(f"\n‖∇E_marg‖:  min={grad_norm.min():.3f}  max={grad_norm.max():.3f}  "
      f"99%ile={np.percentile(grad_norm, 99):.3f}")
print(f"λ̄(u):       min={lam_bar.min():.3f}   max={lam_bar.max():.3f}")
print(f"product:    min={product.min():.3f}    max={product.max():.3f}  "
      f"99%ile={np.percentile(product, 99):.3f}")


# ----- Render: 1×3 panel -----
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5),
                          sharex=True, sharey=True, constrained_layout=True)

panels = [
    (r"$\|\nabla E_{\mathrm{marg}}(u)\|$  —  diverges at manifold",
        np.clip(grad_norm, 0, np.percentile(grad_norm, 99)),
        "Reds"),
    (r"$\overline{\lambda}(u)$  —  vanishes at manifold",
        lam_bar,
        "Blues"),
    (r"$\overline{\lambda}(u) \cdot \|\nabla E_{\mathrm{marg}}\|$  —  bounded",
        np.clip(product, 0, np.percentile(product, 99)),
        "viridis"),
]

for ax, (title, field, cmap) in zip(axes, panels):
    im = ax.imshow(field, cmap=cmap, extent=[-1.5, 1.5, -1.5, 1.5],
                    origin="lower")
    ax.scatter(X[:, 0], X[:, 1], s=4, c="black", alpha=0.6, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

fig.suptitle("The conformal metric counteracts the energy singularity (paper Sec 5.3)",
              fontsize=12, y=1.04)
fig.savefig("exp09_conformal_metric.png", dpi=150, bbox_inches="tight")
print("Saved exp09_conformal_metric.png")


# ----- Bonus: 1D radial slice through origin to manifold -----
fig2, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
n_radial = 200
r = np.linspace(0.01, 1.4, n_radial)
# along x-axis, going from origin outward toward inner ring (r=0.4) and outer ring (r=0.9)
lam_r = np.zeros(n_radial)
grad_r = np.zeros(n_radial)
for k, rr in enumerate(r):
    u = np.array([rr, 0.0], dtype=np.float32)
    p_t_u = G._posterior_t(u, X, t_grid, G.fm_schedule)
    a = np.array([G.fm_schedule(t)[0] for t in t_grid])
    b = np.array([G.fm_schedule(t)[1] for t in t_grid])
    c = np.array([G.fm_schedule(t)[2] for t in t_grid])
    d = np.array([G.fm_schedule(t)[3] for t in t_grid])
    lam_t = (b / a) * (d * a - c * b)
    lam_r[k] = (p_t_u * lam_t).sum()
    D_star = G._D_star(u, X, t_grid, a, b)
    grad_E_t = (u[None, :] - a[:, None] * D_star) / (b[:, None] ** 2 + 1e-12)
    grad_E_marg = (p_t_u[:, None] * grad_E_t).sum(0)
    grad_r[k] = np.linalg.norm(grad_E_marg)

prod_r = lam_r * grad_r
ax.plot(r, grad_r, color="#F5A623", lw=2, label=r"$\|\nabla E_{\mathrm{marg}}\|$ (diverges)")
ax.plot(r, lam_r * grad_r.max() / max(lam_r.max(), 1e-6), color="#009688", lw=2, ls="--",
        label=r"$\overline{\lambda}(u)$ (vanishes — rescaled)")
ax.plot(r, prod_r, color="#542788", lw=2.5, label=r"product (bounded)")
for ring in (0.4, 0.9):
    ax.axvline(ring, color="black", alpha=0.4, ls=":", lw=1)
ax.set_xlabel("radial distance from origin")
ax.set_ylabel("magnitude")
ax.set_title("Radial slice — singularity meets metric, product stays bounded",
              fontsize=11)
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig2.savefig("exp09_radial_slice.png", dpi=150, bbox_inches="tight")
print("Saved exp09_radial_slice.png")
