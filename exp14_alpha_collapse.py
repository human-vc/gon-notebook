"""Experiment 14: save alpha(x, t) heatmaps for the three failed configs
so the notebook can render them as panels rather than a bar chart.
"""

import time
import numpy as np
import torch

import gon_toolkit as G
from exp08_mulan_structural import StructuredAlphaMLP, train_structured

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


def alpha_heatmap(model, lim=1.5, n=48, t_value=0.05):
    g = np.linspace(-lim, lim, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    pts = torch.from_numpy(
        np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    )
    t = torch.full((pts.shape[0], 1), t_value, dtype=torch.float32)
    with torch.no_grad():
        _, alpha = model(pts, t)
    return alpha.squeeze().numpy().reshape(n, n)


X = G.make_circles(n=200, seed=SEED)
N = 48
T_EVAL = 0.05  # near-manifold (where collapse hurts most)

print("[1/3] Naive learned-alpha (no regularizer) ...")
t0 = time.perf_counter()
m_naive, _, _ = G.train_alpha_mulan(X, dim=2, steps=2500, eta_reg=0.0, seed=SEED)
A_naive = alpha_heatmap(m_naive, n=N, t_value=T_EVAL)
print(f"  done in {time.perf_counter() - t0:.1f}s, "
      f"α mean={A_naive.mean():.3f} std={A_naive.std():.3f}")

print("\n[2/3] ν(t)-regularized learned-alpha ...")
t0 = time.perf_counter()
m_nu, _, _ = G.train_alpha_mulan(X, dim=2, steps=2500, eta_reg=5.0, seed=SEED)
A_nu = alpha_heatmap(m_nu, n=N, t_value=T_EVAL)
print(f"  done in {time.perf_counter() - t0:.1f}s, "
      f"α mean={A_nu.mean():.3f} std={A_nu.std():.3f}")

print("\n[3/3] Structural polynomial-in-t α ...")
t0 = time.perf_counter()
m_poly = train_structured(X, dim=2, steps=2500, seed=SEED)
A_poly = alpha_heatmap(m_poly, n=N, t_value=T_EVAL)
print(f"  done in {time.perf_counter() - t0:.1f}s, "
      f"α mean={A_poly.mean():.3f} std={A_poly.std():.3f}")

np.savez("exp14_alpha_fields.npz",
          alpha_naive=A_naive,
          alpha_nu=A_nu,
          alpha_poly=A_poly,
          stats_naive=np.array([A_naive.mean(), A_naive.std()]),
          stats_nu=np.array([A_nu.mean(), A_nu.std()]),
          stats_poly=np.array([A_poly.mean(), A_poly.std()]),
          data_X=X)
print("\nSaved exp14_alpha_fields.npz")
