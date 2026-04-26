"""Experiment 1b: Curl on the CLOSED-FORM optimal autonomous field.

The paper's gradient-flow claim is that f*(u) implements Riemannian gradient flow
on the marginal energy E_marg in the limits where transport correction vanishes
(Sec 5.2 — high D global concentration, Sec 5.3 — near-manifold proximity).

So we should expect:
  - At low D and far from the manifold: visible curl (transport correction active)
  - At high D and/or near the manifold: curl → 0 (Riemannian gradient flow regime)

We render f*(u) at D ∈ {2, 8, 32, 128} on a 2D plane (slicing the 2D circles
embedded in higher D) and measure curl at each.
"""

import numpy as np
import matplotlib.pyplot as plt

import gon_toolkit as G

SEED = 0
np.random.seed(SEED)


def closed_form_curl_at_D(D, X2, lim=1.5, n=32):
    """Compute curl of the closed-form f*(u) restricted to the embedded 2D plane.

    We embed X2 into R^D via a random orthogonal P (D × 2), with columns
    p1, p2. Points on the 2D plane in R^D are u = c1*p1 + c2*p2. We compute
    f*(u) ∈ R^D and then project the field BACK onto the 2D plane via P^T,
    yielding a 2D vector field. We compute curl of that.
    """
    if D == 2:
        XD = X2.copy()
        P = np.eye(2, dtype=np.float32)
    else:
        rng = np.random.default_rng(SEED)
        M = rng.standard_normal((D, 2)).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        P = Q  # D × 2
        XD = (X2 @ P.T).astype(np.float32)

    g = np.linspace(-lim, lim, n)
    XX, YY = np.meshgrid(g, g)
    U = np.zeros((n, n), dtype=np.float32)
    V = np.zeros((n, n), dtype=np.float32)

    t_grid = np.linspace(0.02, 0.98, 24)

    for i, x in enumerate(g):
        for j, y in enumerate(g):
            u_2d = np.array([x, y], dtype=np.float32)
            u_D = u_2d @ P.T  # back into R^D
            f_D = G.f_star_discrete(u_D, XD, t_grid)
            f_2d = f_D @ P  # project back to 2D
            U[j, i], V[j, i] = f_2d

    h = g[1] - g[0]
    dV_dx = np.gradient(V, h, axis=1)
    dU_dy = np.gradient(U, h, axis=0)
    curl = dV_dx - dU_dy
    return XX, YY, U, V, curl


X2 = G.make_circles(n=120, seed=SEED)

D_values = [2, 8, 32, 128]
results = {}
for D in D_values:
    print(f"D={D}...", flush=True)
    XX, YY, U, V, curl = closed_form_curl_at_D(D, X2, lim=1.5, n=24)
    results[D] = (XX, YY, U, V, curl)
    print(f"  |curl| mean={np.abs(curl).mean():.4f}, "
          f"max={np.abs(curl).max():.4f}, std={curl.std():.4f}")

# Compute global vmax for shared color scale
all_curl = np.concatenate([results[D][4].ravel() for D in D_values])
vmax = float(np.percentile(np.abs(all_curl), 95))

fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
for ax, D in zip(axes, D_values):
    XX, YY, U, V, curl = results[D]
    im = ax.imshow(curl, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[-1.5, 1.5, -1.5, 1.5], origin="lower")
    ax.scatter(X2[:, 0], X2[:, 1], s=2, c="black", alpha=0.6)
    ax.set_title(f"D = {D}\n|curl| mean = {np.abs(curl).mean():.4f}",
                 fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label=r"$\nabla \times f^*$")
cbar.outline.set_visible(False)
fig.suptitle("Curl of optimal autonomous field f*(u) — should vanish as D grows", y=1.05)

fig.savefig("exp01b_curl_analytical.png", dpi=150, bbox_inches="tight")
print("\nSaved exp01b_curl_analytical.png")
