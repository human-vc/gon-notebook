"""Experiment 8: MuLAN-style structural fix for the alpha-collapse.

Earlier we showed:
  - Naive learned-α: collapses to α=1 (noise-pred, unstable)
  - ν(t) regularizer: collapses to α=0 (velocity, stable but degenerate)

The paper MuLAN (arXiv:2312.13236) solves the analogous degeneracy with a
*structural* constraint: their α(x, t) is a degree-5 polynomial in t with
sign constraints that force monotonicity. The structure prevents collapse.

We adapt: parameterize α(x, t) as a polynomial in t with coefficients output
by a network of x. Specifically: α(x, t) = sigmoid(p_0(x) + p_1(x)*t + p_2(x)*t²)
where p_i(x) are network heads. This bakes in t-dependence while letting x
modulate it. The network can't collapse to a constant α because t is hardwired.

Test: train this and see if α varies meaningfully in BOTH x and t.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import gon_toolkit as G

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


class StructuredAlphaMLP(nn.Module):
    """Network outputs target prediction + 3 polynomial coefficients for α(x, t).

    α(x, t) = sigmoid(p0(x) + p1(x) * t + p2(x) * t²)

    This forces structural t-dependence — α can't be constant in t.
    """

    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        # Three polynomial coefficient heads
        self.p0 = nn.Linear(hidden, 1)
        self.p1 = nn.Linear(hidden, 1)
        self.p2 = nn.Linear(hidden, 1)
        # Initialize so initial alpha is roughly 0.5 (no commitment)
        nn.init.zeros_(self.p0.bias)
        nn.init.zeros_(self.p1.bias)
        nn.init.zeros_(self.p2.bias)
        nn.init.zeros_(self.p0.weight); nn.init.zeros_(self.p1.weight); nn.init.zeros_(self.p2.weight)

    def forward(self, u, t):
        h = self.shared(u)
        p0 = self.p0(h); p1 = self.p1(h); p2 = self.p2(h)
        logits = p0 + p1 * t + p2 * (t ** 2)
        alpha = torch.sigmoid(logits)
        target_pred = self.target_head(torch.cat([u, t], dim=-1))
        return target_pred, alpha


def train_structured(X, dim=2, steps=4000, batch=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xd_t = torch.from_numpy(X.astype(np.float32))
    model = StructuredAlphaMLP(dim=dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(Xd_t), (batch,))
        x_clean = Xd_t[idx]
        t = torch.rand(batch, 1) * 0.98 + 0.01
        eps = torch.randn_like(x_clean)
        u = (1 - t) * x_clean + t * eps
        pred, alpha = model(u, t)
        v_target = eps - x_clean
        target = alpha * eps + (1 - alpha) * v_target
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            losses.append(loss.item())
    return model


def alpha_grid(model, t_value, lim=1.5, n=40):
    g = np.linspace(-lim, lim, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    pts = torch.from_numpy(np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32))
    t = torch.full((pts.shape[0], 1), t_value, dtype=torch.float32)
    with torch.no_grad():
        _, alpha = model(pts, t)
    return XX, YY, alpha.squeeze().numpy().reshape(n, n)


X = G.make_circles(n=200, seed=SEED)

print("Training structured α (polynomial in t with x-dependent coefficients)...")
t0 = time.perf_counter()
model = train_structured(X, dim=2, steps=8000, seed=SEED)
print(f"  done in {time.perf_counter() - t0:.1f}s")


# ----- Evaluate alpha at multiple t -----
t_values = [0.05, 0.30, 0.60, 0.95]
results = []
for tv in t_values:
    XX, YY, A = alpha_grid(model, tv, lim=1.5, n=40)
    results.append((tv, A))
    print(f"  t={tv}: α mean={A.mean():.3f}, std={A.std():.3f}, "
          f"range=[{A.min():.3f}, {A.max():.3f}]")


# ----- Render -----
fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True, constrained_layout=True)

for ax, (tv, A) in zip(axes, results):
    im = ax.imshow(A, cmap="RdBu_r", vmin=0.0, vmax=1.0,
                   extent=[-1.5, 1.5, -1.5, 1.5], origin="lower")
    ax.scatter(X[:, 0], X[:, 1], s=2, c="black", alpha=0.6)
    ax.set_title(f"t = {tv}", fontsize=11)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02,
                     label=r"$\alpha(x, t)$  (0=velocity, 1=noise-pred)")
cbar.outline.set_visible(False)
fig.suptitle("Structured learned α — polynomial in t with x-dependent coefficients",
              fontsize=12, y=1.04)
fig.savefig("exp08_mulan_structural.png", dpi=150, bbox_inches="tight")
print("Saved exp08_mulan_structural.png")
