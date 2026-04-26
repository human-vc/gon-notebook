"""Experiment 13: data for the iconic phase-diagram figure (Goh α-β-plane analog).

Trains 4 representative (α, D) configurations on circles, samples 400 points
each, saves the scatter plus a denser W2 surface (for smooth heatmap).
"""

import time
import numpy as np
import torch

import gon_toolkit as G

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

X_train = G.make_circles(n=300, r_inner=0.4, r_outer=0.9, noise=0.02, seed=SEED)
X_truth_2d = G.make_circles(n=400, r_inner=0.4, r_outer=0.9, noise=0.02,
                             seed=SEED + 100)


def train_sample_w2(X, alpha, D, n=400, steps=3500, seed=0):
    model, _ = G.train_param(X, alpha=alpha, dim=D, steps=steps,
                              batch=128, lr=1e-3, seed=seed,
                              hidden=128, depth=3)
    samples_D = G.sample_ode(model, n=n, dim=D, n_steps=250,
                              sigma_init=1.0, seed=seed)
    # project samples back to 2D for visualization (use SAME projection)
    if D == 2:
        samples_2d = samples_D
    else:
        rng = np.random.default_rng(seed)
        Mp = rng.standard_normal((D, 2)).astype(np.float32)
        Pp, _ = np.linalg.qr(Mp)
        samples_2d = samples_D @ Pp  # (n, D) @ (D, 2) -> (n, 2)
    truth_D = G.project_up(X_truth_2d, D, seed=seed)
    w2 = G.sliced_w2(samples_D, truth_D, n_proj=128)
    return samples_2d, w2


# ----- 4 corner configurations -----
corners = [
    ("velocity_lowD",  0.0,  2),
    ("noise_lowD",     1.0,  2),
    ("velocity_highD", 0.0, 32),
    ("noise_highD",    1.0, 32),
]

scatters = {}
w2s = {}
print("Training 4 corner configurations...")
for name, alpha, D in corners:
    t0 = time.perf_counter()
    samples, w2 = train_sample_w2(X_train, alpha=alpha, D=D, seed=SEED)
    scatters[name] = samples
    w2s[name] = w2
    print(f"  {name:18s} α={alpha:.1f} D={D:2d}: W2={w2:.3f}  "
          f"({time.perf_counter() - t0:.1f}s)")

# ----- Denser phase grid for smooth heatmap (9 alphas × 8 Ds = 72 runs) -----
alpha_dense = np.linspace(0.0, 1.0, 9)
D_dense = np.array([2, 3, 4, 6, 8, 16, 32, 64])
print(f"\nDense phase grid: {len(alpha_dense)}×{len(D_dense)} = "
      f"{len(alpha_dense) * len(D_dense)} runs")
W2_dense = np.zeros((len(alpha_dense), len(D_dense)))

t_start = time.perf_counter()
for i, alpha in enumerate(alpha_dense):
    for j, D in enumerate(D_dense):
        t0 = time.perf_counter()
        _, w2 = train_sample_w2(X_train, alpha=float(alpha), D=int(D),
                                  steps=2500, seed=SEED)
        W2_dense[i, j] = w2
        print(f"  α={alpha:.2f}, D={int(D):3d}: W2={w2:.3f}  "
              f"({time.perf_counter() - t0:.1f}s)")

print(f"\nTotal: {time.perf_counter() - t_start:.1f}s")

np.savez("exp13_iconic.npz",
          alpha_dense=alpha_dense, D_dense=D_dense, W2_dense=W2_dense,
          # corner scatters
          velocity_lowD=scatters["velocity_lowD"],
          noise_lowD=scatters["noise_lowD"],
          velocity_highD=scatters["velocity_highD"],
          noise_highD=scatters["noise_highD"],
          w2_velocity_lowD=w2s["velocity_lowD"],
          w2_noise_lowD=w2s["noise_lowD"],
          w2_velocity_highD=w2s["velocity_highD"],
          w2_noise_highD=w2s["noise_highD"],
          truth_2d=X_truth_2d)
print("Saved exp13_iconic.npz")
