"""GoN toolkit, closed-form fields, parameterizations, samplers,
learned-alpha, curl, Wasserstein-2, for the marimo notebook on
arXiv:2602.18428.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "this function requires torch; install with "
            "`uv pip install torch`. The notebook itself doesn't need "
            "torch since it loads pre-computed .npz results."
        )


def _no_grad_decorator(fn):
    if _HAS_TORCH:
        return torch.no_grad()(fn)
    return fn

STABLE = "#009688"
UNSTABLE = "#F5A623"
ACCENT = "#542788"
DATA = "#222222"
GRID = "#E5E5E5"
MUTED = "#9B9B9B"

def fm_schedule(t):
    return 1.0 - t, t, -1.0, 1.0

def make_circles(n=200, r_inner=0.4, r_outer=0.9, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n_inner = n // 2
    n_outer = n - n_inner
    theta_in = rng.uniform(0, 2 * np.pi, n_inner)
    theta_out = rng.uniform(0, 2 * np.pi, n_outer)
    pts = np.concatenate([
        np.stack([r_inner * np.cos(theta_in), r_inner * np.sin(theta_in)], axis=1),
        np.stack([r_outer * np.cos(theta_out), r_outer * np.sin(theta_out)], axis=1),
    ], axis=0)
    pts = pts + noise * rng.standard_normal(pts.shape)
    return pts.astype(np.float32)

def make_moons(n=200, noise=0.05, seed=0):
    from sklearn.datasets import make_moons as _mm
    X, _ = _mm(n_samples=n, noise=noise, random_state=seed)

    X = X - X.mean(axis=0)
    X = X / np.abs(X).max()
    return X.astype(np.float32)

def make_gmm_ring(n=200, k=8, r=0.7, sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(k) / k
    centers = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    idx = rng.integers(0, k, size=n)
    return (centers[idx] + sigma * rng.standard_normal((n, 2))).astype(np.float32)

def make_swiss_2d(n=200, noise=0.05, seed=0):
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    X = np.stack([X[:, 0], X[:, 2]], axis=1) / 9.0
    return X.astype(np.float32)

def project_up(X2, D, seed=0):
    """Embed 2D data into ℝ^D via a random orthogonal projection P (D × 2)."""
    if D == X2.shape[1]:
        return X2.copy()
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((D, X2.shape[1])).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return (X2 @ Q.T).astype(np.float32)

def _posterior_t(u, X, t_grid, sched_fn=fm_schedule):
    """p(t|u) up to normalization, on the t_grid, for discrete X. Returns (T,) array."""
    a = np.array([sched_fn(t)[0] for t in t_grid])
    b = np.array([sched_fn(t)[1] for t in t_grid])
    diff = u[None, None, :] - a[:, None, None] * X[None, :, :]
    sq = (diff ** 2).sum(-1) / (2 * b[:, None] ** 2 + 1e-12)
    log_pu_kt = -sq - 0.5 * X.shape[1] * np.log(2 * np.pi * b[:, None] ** 2 + 1e-12)
    log_pu_t = np.logaddexp.reduce(log_pu_kt, axis=1) - np.log(len(X))
    log_p_t_u = log_pu_t
    log_p_t_u = log_p_t_u - log_p_t_u.max()
    p = np.exp(log_p_t_u)
    return p / p.sum()

def f_star_discrete(u, X, t_grid, sched_fn=fm_schedule):
    """Optimal autonomous field f*(u) for discrete X (Eq 33 + 35)."""
    a = np.array([sched_fn(t)[0] for t in t_grid])
    b = np.array([sched_fn(t)[1] for t in t_grid])
    c = np.array([sched_fn(t)[2] for t in t_grid])
    d = np.array([sched_fn(t)[3] for t in t_grid])

    diff = u[None, None, :] - a[:, None, None] * X[None, :, :]
    sq = (diff ** 2).sum(-1) / (2 * b[:, None] ** 2 + 1e-12)
    log_w = -sq
    log_w = log_w - log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w); w = w / w.sum(axis=1, keepdims=True)
    D_star = w @ X

    log_pu_kt = -sq - 0.5 * X.shape[1] * np.log(2 * np.pi * b[:, None] ** 2 + 1e-12)
    log_pu_t = np.logaddexp.reduce(log_pu_kt, axis=1) - np.log(len(X))
    log_p_t_u = log_pu_t - log_pu_t.max()
    p_t_u = np.exp(log_p_t_u); p_t_u = p_t_u / p_t_u.sum()

    inner = (d / b)[:, None] * u[None, :] + (c - d * a / b)[:, None] * D_star
    return (p_t_u[:, None] * inner).sum(axis=0)

def field_grid(X, t_grid, sched_fn=fm_schedule, lim=1.5, n=24):
    """Render f*(u) on an n×n grid in [-lim, lim]²."""
    g = np.linspace(-lim, lim, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    field = np.zeros((n, n, 2), dtype=np.float32)
    for i, x in enumerate(g):
        for j, y in enumerate(g):
            field[j, i] = f_star_discrete(np.array([x, y], dtype=np.float32),
                                          X, t_grid, sched_fn)
    return XX, YY, field

def decompose_field_grid(X, t_grid, sched_fn=fm_schedule, lim=1.5, n=20):
    """Returns XX, YY, natural_grad (n,n,2), transport (n,n,2), drift (n,n,2)."""
    g = np.linspace(-lim, lim, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    natural = np.zeros((n, n, 2), dtype=np.float32)
    transport = np.zeros((n, n, 2), dtype=np.float32)
    drift = np.zeros((n, n, 2), dtype=np.float32)

    a = np.array([sched_fn(t)[0] for t in t_grid])
    b = np.array([sched_fn(t)[1] for t in t_grid])
    c = np.array([sched_fn(t)[2] for t in t_grid])
    d = np.array([sched_fn(t)[3] for t in t_grid])

    lam_t = (b / a) * (d * a - c * b)

    for i, x in enumerate(g):
        for j, y in enumerate(g):
            u = np.array([x, y], dtype=np.float32)
            p_t_u = _posterior_t(u, X, t_grid, sched_fn)
            lam_bar = (p_t_u * lam_t).sum()
            c_bar = (p_t_u * (c / a)).sum()

            diff = u[None, :] - a[:, None] * (
                _D_star(u, X, t_grid, a, b)
            )
            grad_E_t = diff / (b[:, None] ** 2 + 1e-12)
            grad_E_marg = (p_t_u[:, None] * grad_E_t).sum(0)
            natural[j, i] = lam_bar * grad_E_marg

            tc = ((lam_t - lam_bar)[:, None] * (grad_E_t - grad_E_marg[None, :]))
            transport[j, i] = (p_t_u[:, None] * tc).sum(0)

            drift[j, i] = c_bar * u

    return XX, YY, natural, transport, drift

def _D_star(u, X, t_grid, a, b):
    """Posterior denoiser per t: returns (T, D)."""
    diff = u[None, None, :] - a[:, None, None] * X[None, :, :]
    sq = (diff ** 2).sum(-1) / (2 * b[:, None] ** 2 + 1e-12)
    log_w = -sq
    log_w = log_w - log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w); w = w / w.sum(axis=1, keepdims=True)
    return w @ X

def conformal_grid(X, t_grid, sched_fn=fm_schedule, lim=1.5, n=40):
    """Returns XX, YY, lam_bar, grad_E_marg_norm, product on a 2D grid.

    lam_bar(u) = E_{t|u}[lambda(t)],  lambda(t) = (b/a)(d*a - c*b)  [Eq 15]
    grad_E_marg(u) = E_{t|u}[(u - a*D*_t(u)) / b^2]                  [Eq 11]

    Their product is the Natural Gradient term (Eq 17): bounded near manifold.
    """
    g = np.linspace(-lim, lim, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    a = np.array([sched_fn(t)[0] for t in t_grid])
    b = np.array([sched_fn(t)[1] for t in t_grid])
    c = np.array([sched_fn(t)[2] for t in t_grid])
    d = np.array([sched_fn(t)[3] for t in t_grid])
    lam_t = (b / a) * (d * a - c * b)

    lam_bar = np.zeros((n, n))
    grad_norm = np.zeros((n, n))
    grad_field = np.zeros((n, n, 2))
    for i, x in enumerate(g):
        for j, y in enumerate(g):
            u = np.array([x, y], dtype=np.float32)
            p_t_u = _posterior_t(u, X, t_grid, sched_fn)
            lam_bar[j, i] = (p_t_u * lam_t).sum()
            D_star = _D_star(u, X, t_grid, a, b)
            grad_E_t = (u[None, :] - a[:, None] * D_star) / (b[:, None] ** 2 + 1e-12)
            gE = (p_t_u[:, None] * grad_E_t).sum(0)
            grad_field[j, i] = gE
            grad_norm[j, i] = np.linalg.norm(gE)
    return XX, YY, lam_bar, grad_norm, grad_field

def conformal_radial(X, t_grid, x_target, direction, r_vals,
                     sched_fn=fm_schedule):
    """Compute lam_bar(u) and grad_E_marg_norm(u) along u = x_target + r * direction."""
    x_target = np.asarray(x_target, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    r_vals = np.asarray(r_vals, dtype=np.float32)
    a = np.array([sched_fn(t)[0] for t in t_grid])
    b = np.array([sched_fn(t)[1] for t in t_grid])
    c = np.array([sched_fn(t)[2] for t in t_grid])
    d = np.array([sched_fn(t)[3] for t in t_grid])
    lam_t = (b / a) * (d * a - c * b)
    R = len(r_vals)
    lam_bar = np.zeros(R)
    grad_norm = np.zeros(R)
    for i in range(R):
        u = (x_target + r_vals[i] * direction).astype(np.float32)
        p_t_u = _posterior_t(u, X, t_grid, sched_fn)
        lam_bar[i] = (p_t_u * lam_t).sum()
        diff = u[None, None, :] - a[:, None, None] * X[None, :, :]
        sq = (diff ** 2).sum(-1) / (2 * b[:, None] ** 2 + 1e-12)
        log_w = -sq
        log_w = log_w - log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w); w = w / w.sum(axis=1, keepdims=True)
        D_star = w @ X
        grad_E_t = (u[None, :] - a[:, None] * D_star) / (b[:, None] ** 2 + 1e-12)
        gE = (p_t_u[:, None] * grad_E_t).sum(0)
        grad_norm[i] = np.linalg.norm(gE)
    return lam_bar, grad_norm

def jensen_gap_grid(X, t_grid, sched_fn=fm_schedule, t_eval=0.05, lim=1.5, n=40):
    """Compute the noise-prediction Jensen Gap on a 2D grid (Eq 66):

      Δv_noise ∝ ‖u - x*‖ · |b'(t)/b(t)| · |b(t) · E_{τ|u}[1/b(τ)] - 1|

    Evaluated at one fixed t (where the singularity bites). x* is the nearest
    data point. Use Flow Matching schedule: b(t) = t, so b' = 1, b'/b = 1/t.
    """
    g = np.linspace(-lim, lim, n).astype(np.float32)
    XX, YY = np.meshgrid(g, g)
    a_grid = np.array([sched_fn(t)[0] for t in t_grid])
    b_grid = np.array([sched_fn(t)[1] for t in t_grid])

    gap = np.zeros((n, n))
    for i, x in enumerate(g):
        for j, y in enumerate(g):
            u = np.array([x, y], dtype=np.float32)

            d2 = ((X - u[None, :]) ** 2).sum(-1)
            x_near = X[np.argmin(d2)]
            dist = np.linalg.norm(u - x_near)

            p_t_u = _posterior_t(u, X, t_grid, sched_fn)
            inv_b_mean = (p_t_u / (b_grid + 1e-6)).sum()

            b_eval = t_eval
            jensen = abs(b_eval * inv_b_mean - 1.0)
            prefactor = abs(1.0 / b_eval)
            gap[j, i] = dist * prefactor * jensen
    return XX, YY, gap

if _HAS_TORCH:
    class TinyMLP(nn.Module):
        def __init__(self, dim_in=2, dim_out=2, hidden=64, depth=3):
            super().__init__()
            layers = [nn.Linear(dim_in, hidden), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            layers += [nn.Linear(hidden, dim_out)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

def train_param(X, alpha=0.0, dim=None, steps=2000, batch=64, lr=3e-4, seed=0,
                hidden=128, depth=3):
    _require_torch()
    torch.manual_seed(seed)
    dim = dim or X.shape[1]
    Xd = project_up(X, dim, seed=seed) if dim != X.shape[1] else X
    Xd_t = torch.from_numpy(Xd)
    model = TinyMLP(dim_in=dim, dim_out=dim, hidden=hidden, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(Xd_t), (batch,))
        x_clean = Xd_t[idx]
        t = torch.rand(batch, 1) * 0.98 + 0.01
        eps = torch.randn_like(x_clean)
        u = (1 - t) * x_clean + t * eps
        v_target = eps - x_clean
        eps_target = eps
        target = alpha * eps_target + (1 - alpha) * v_target
        loss = ((model(u) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            losses.append(loss.item())
    return model, losses

def train_eqm(X, dim=None, steps=2000, batch=64, lr=3e-4, seed=0):
    _require_torch()
    torch.manual_seed(seed)
    dim = dim or X.shape[1]
    Xd = project_up(X, dim, seed=seed) if dim != X.shape[1] else X
    Xd_t = torch.from_numpy(Xd)
    model = TinyMLP(dim_in=dim, dim_out=dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(Xd_t), (batch,))
        x_clean = Xd_t[idx]
        t = torch.rand(batch, 1) * 0.98 + 0.01
        eps = torch.randn_like(x_clean)
        u = (1 - t) * x_clean + t * eps
        target = u - x_clean
        loss = ((model(u) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            losses.append(loss.item())
    return model, losses

if _HAS_TORCH:
    class AlphaMLP(nn.Module):
        """Outputs (predicted target, alpha(x,t)), alpha bounded to [0, 1]."""

        def __init__(self, dim=2, hidden=64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(dim + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
            )
            self.target_head = nn.Linear(hidden, dim)
            self.alpha_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

        def forward(self, u, t):
            h = self.shared(torch.cat([u, t], dim=-1))
            return self.target_head(h), self.alpha_head(h)

def train_alpha_mulan(X, dim=2, steps=3000, batch=64, lr=3e-4, eta_reg=2.0, seed=0):
    """Train learned-alpha(x,t) with a ν(t) regularizer."""
    _require_torch()
    torch.manual_seed(seed)
    Xd = project_up(X, dim, seed=seed) if dim != X.shape[1] else X
    Xd_t = torch.from_numpy(Xd)
    model = AlphaMLP(dim=dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses_mse, losses_reg = [], []
    for step in range(steps):
        idx = torch.randint(0, len(Xd_t), (batch,))
        x_clean = Xd_t[idx]
        t = torch.rand(batch, 1) * 0.98 + 0.01
        eps = torch.randn_like(x_clean)
        u = (1 - t) * x_clean + t * eps
        pred, alpha = model(u, t)
        v_target = eps - x_clean
        target = alpha * eps + (1 - alpha) * v_target
        mse = ((pred - target) ** 2).mean()
        nu_reg = ((alpha ** 2) / (t ** 2 + 1e-3)).mean()
        loss = mse + eta_reg * nu_reg
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            losses_mse.append(mse.item())
            losses_reg.append(nu_reg.item())
    return model, losses_mse, losses_reg

@_no_grad_decorator
def sample_ode(model, n=500, dim=2, n_steps=200, sigma_init=1.5, seed=0):
    """Euler ODE: dx/dt = -v(x). Integrate t: 1 → 0."""
    torch.manual_seed(seed)
    x = sigma_init * torch.randn(n, dim)
    ts = np.linspace(1.0, 0.01, n_steps)
    for i in range(n_steps - 1):
        v = model(x)
        dt = ts[i + 1] - ts[i]
        x = x + dt * v
    return x.numpy()

@_no_grad_decorator
def sample_sde(model, n=500, dim=2, n_steps=200, sigma_init=1.5,
               noise_scale=0.5, seed=0):
    """Euler-Maruyama: dx = -v(x) dt + sigma(t) sqrt(|dt|) dW."""
    torch.manual_seed(seed)
    x = sigma_init * torch.randn(n, dim)
    ts = np.linspace(1.0, 0.01, n_steps)
    for i in range(n_steps - 1):
        v = model(x)
        dt = ts[i + 1] - ts[i]
        sigma = noise_scale * np.sqrt(ts[i])
        x = x + dt * v + sigma * np.sqrt(abs(dt)) * torch.randn_like(x)
    return x.numpy()

@_no_grad_decorator
def sample_langevin(model, n=500, dim=2, n_steps=400, sigma_init=1.5,
                    step_size=0.005, seed=0):
    """Annealed Langevin on the trained field treated as score-like."""
    torch.manual_seed(seed)
    x = sigma_init * torch.randn(n, dim)
    for i in range(n_steps):
        score_like = -model(x)
        sigma = step_size * (1.0 - 0.5 * i / n_steps)
        x = x + step_size * score_like + np.sqrt(2 * sigma) * torch.randn_like(x)
    return x.numpy()

def sliced_w2(X, Y, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Xp = X @ dirs.T
    Yp = Y @ dirs.T
    return float(np.sqrt(np.mean((np.sort(Xp, 0) - np.sort(Yp, 0)) ** 2)))

def curl_2d(field_fn, lim=1.5, n=48):
    """Compute ∇×f on an n×n grid. Returns (XX, YY, curl, U, V)."""
    g = np.linspace(-lim, lim, n)
    XX, YY = np.meshgrid(g, g)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    UV = field_fn(pts).reshape(n, n, 2)
    U, V = UV[..., 0], UV[..., 1]
    h = g[1] - g[0]
    dV_dx = np.gradient(V, h, axis=1)
    dU_dy = np.gradient(U, h, axis=0)
    curl = dV_dx - dU_dy
    return XX, YY, curl, U, V

def model_to_field(model):
    """Wrap a torch model so it accepts/returns numpy."""
    def f(pts_np):
        with torch.no_grad():
            t = torch.from_numpy(pts_np.astype(np.float32))
            return model(t).numpy()
    return f
