# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Why Autonomous Diffusion Really Works.

A marimo notebook walking through "The Geometry of Noise" (Sahraee-Ardakan,
Delbracio, Milanfar — Google, 2026; arXiv:2602.18428).
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import gon_toolkit as G
    import gon_data

    return G, gon_data, mo, np, plt

@app.cell
def _palette():

    STABLE = "#1B9E77"
    UNSTABLE = "#D95F02"
    ACCENT = "#7570B3"
    DATA = "#222222"
    MUTED = "#9B9B9B"
    return ACCENT, STABLE, UNSTABLE

@app.cell
def _rcparams(plt):

    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#555555",
        "axes.labelcolor":    "#222222",
        "axes.titlesize":     11,
        "axes.titleweight":   "normal",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        "#555555",
        "ytick.color":        "#555555",
        "grid.color":         "#E5E5E5",
        "grid.linewidth":     0.6,
        "font.family":        "sans-serif",
        "font.size":          10,
        "mathtext.fontset":   "cm",
    })
    return

@app.cell
def _title(mo):
    mo.md(r"""
    <div style="text-align:center; padding-top:0.5rem;">
      <h1 style="font-family: Georgia, 'Times New Roman', serif;
                 font-weight:400; font-size:3.2rem; margin:0; line-height:1.05;
                 color:#222;">
        Why Autonomous Diffusion Really Works
      </h1>
    </div>
    """)
    return

@app.cell
def _hero_data(G, np):
    X_hero = G.make_circles(n=120, seed=0)
    t_grid_hero = np.linspace(0.05, 0.95, 16)
    return X_hero, t_grid_hero

@app.cell
def _hero_controls(mo):

    D_hero = mo.ui.slider(
        steps=[2, 4, 8, 16, 32, 64, 128],
        value=8,
        show_value=False,
    )
    seed_hero = mo.ui.slider(
        start=0, stop=9, step=1, value=0,
        show_value=False,
    )
    return D_hero, seed_hero

@app.cell
def _hero_field(D_hero, G, X_hero, np, seed_hero, t_grid_hero):
    rng = np.random.default_rng(int(seed_hero.value))
    M = rng.standard_normal((D_hero.value, 2)).astype(np.float32)
    P, _ = np.linalg.qr(M)
    XD = (X_hero @ P.T).astype(np.float32) if D_hero.value != 2 else X_hero

    gx = np.linspace(-3.0, 3.0, 42).astype(np.float32)
    gy = np.linspace(-1.0, 1.0, 16).astype(np.float32)
    XX_h, YY_h = np.meshgrid(gx, gy)
    UV_h = np.zeros((len(gy), len(gx), 2), dtype=np.float32)
    for i, x in enumerate(gx):
        for j, y in enumerate(gy):
            u_D = (P @ np.array([x, y], dtype=np.float32)) if D_hero.value != 2 \
                  else np.array([x, y], dtype=np.float32)
            f_D = G.f_star_discrete(u_D, XD, t_grid_hero, G.fm_schedule)
            UV_h[j, i] = (P.T @ f_D) if D_hero.value != 2 else f_D
    return UV_h, XX_h, YY_h

@app.cell
def _hero_plot(STABLE, UV_h, XX_h, X_hero, YY_h, np, plt):
    def _draw():

        fig, ax = plt.subplots(figsize=(14, 4.7), constrained_layout=True)
        mag = np.linalg.norm(UV_h, axis=-1)
        ax.quiver(XX_h, YY_h, UV_h[..., 0], UV_h[..., 1], mag,
                    cmap="viridis", scale_units="xy",
                    scale=mag.max() * 1.4 + 1e-6,
                    width=0.0035, headwidth=4, headlength=5,
                    pivot="mid", alpha=0.85)
        ax.scatter(X_hero[:, 0], X_hero[:, 1], s=14, c=STABLE,
                    edgecolors="white", linewidths=0.6, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-3.0, 3.0); ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([]); ax.set_yticks([])

        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#d0d0d0")
            s.set_linewidth(0.8)
        return fig
    _draw()
    return

@app.cell
def _hero_row(D_hero, mo, seed_hero):

    col_slider1 = mo.vstack([
        mo.md(f"**ambient dimension  $D = {D_hero.value}$**"),
        D_hero,
        mo.md(
            "<div style='display:flex; justify-content:space-between; "
            "color:#888; font-size:0.8rem;'>"
            "<span>2</span><span>32</span><span>128</span></div>"
        ),
    ], gap=0.3)
    col_slider2 = mo.vstack([
        mo.md(f"**projection seed**  $= {int(seed_hero.value)}$"),
        seed_hero,
        mo.md(
            "<div style='display:flex; justify-content:space-between; "
            "color:#888; font-size:0.8rem;'>"
            "<span>0</span><span>9</span></div>"
        ),
    ], gap=0.3)
    col_prose = mo.md(
        r"""
        We often think of diffusion models as needing to know **how much**
        noise was added. But recent autonomous models throw away the time
        conditioning entirely — and they still work. A single static field,
        with no clock, guides samples from pure noise to clean data. *What
        is going on?*
        """
    )
    mo.hstack([col_slider1, col_slider2, col_prose],
                justify="space-between", gap=2.0, align="start")
    return

@app.cell
def _popular_story(mo):
    mo.md(r"""
    Here's the popular story about diffusion. A network learns to denoise data
    at every noise level. To denoise an image, you tell the network *how much*
    noise was added, the time $t$, and it predicts the noise. To generate,
    you start from pure noise and call the network at decreasing $t$ until an
    image emerges.

    This story isn't wrong. But it has a paradox at its center.
    """)
    return

@app.cell
def _the_break(mo):
    mo.md(r"""
    **Recent autonomous models throw away the time conditioning entirely.** No
    $t$-embedding, no schedule input. The network sees only the noisy
    observation $\mathbf{u}$ and predicts a target. Equilibrium Matching
    [[Wang & Du, 2025](https://arxiv.org/abs/2510.02300)] does this. So does
    blind Flow Matching [[Sun et al., 2025](https://arxiv.org/abs/2502.13129)].
    And they *work*.

    How? The "right" gradient to follow from a point $\mathbf{u}$ should depend
    heavily on its noise level. How can a single static field guide a sample
    from pure noise (high $t$) and also guide a sample from light noise (low
    $t$), all while ensuring its stationary points sit on the clean data?
    """)
    return

@app.cell
def _the_promise(mo):
    mo.md(r"""
    *The Geometry of Noise* resolves the paradox geometrically. The
    autonomous field isn't merely "blind denoising" — it implicitly performs
    **Riemannian gradient flow on a marginal energy landscape**. The landscape
    has an infinitely deep singularity at the data manifold. But the learned
    field carries a local conformal metric — the paper's *effective gain* —
    that vanishes at exactly the rate the gradient diverges. Their product
    stays bounded. The geometry hands us a free Riemannian preconditioner.

    We're going to make that abstract claim concrete by studying autonomous
    diffusion on the simplest model where we can write the answer down
    exactly: a 2D dataset of points, embedded into $\mathbb{R}^D$. This
    setting is rich enough to reproduce every parameterization the paper
    analyzes, yet simple enough to render every quantity by hand.
    """)
    return

@app.cell
def _first_steps_md(mo):
    mo.md(r"""
    ## First Steps: The Closed-Form Optimal Field

    We begin by writing the autonomous diffusion problem in the simplest form
    that captures every parameterization at once. The forward process
    corrupts a clean point $\mathbf{x}$ with Gaussian noise
    $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$ — but the rate, the
    scaling, and the velocity all live inside a single affine schedule:

    $$\mathbf{u}_t = a(t)\,\mathbf{x} + b(t)\,\boldsymbol{\varepsilon}.$$

    A network $f$ predicts a linear target
    $r(\mathbf{x}, \boldsymbol{\varepsilon}, t) = c(t)\,\mathbf{x} + d(t)\,\boldsymbol{\varepsilon}$.
    The four schedule functions $(a, b, c, d)$ are rich enough to recover
    DDPM, EDM, Flow Matching, and Equilibrium Matching as special cases [Sun
    et al., 2025], yet simple enough to let us write the optimal autonomous
    field down in closed form.
    """)
    return

@app.cell
def _table1(mo):
    mo.md(r"""
    | Model | $a(t)$ | $b(t)$ | $c(t)$ | $d(t)$ | What it predicts |
    |---|---|---|---|---|---|
    | **DDPM** | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ | 0 | 1 | noise $\boldsymbol{\varepsilon}$ |
    | **EDM** | 1 | $\sigma_t$ | 1 | 0 | data $\mathbf{x}$ |
    | **Flow Matching** | $1-t$ | $t$ | $-1$ | $1$ | velocity $\boldsymbol{\varepsilon} - \mathbf{x}$ |
    | **Equilibrium Matching** | $1-t$ | $t$ | $-t$ | $t$ | $\mathbf{u} - \mathbf{x}$ |

    We use the Flow Matching schedule throughout. The key fact: an autonomous
    model, one without access to $t$, minimizes its MSE loss when its output
    is the *posterior expectation* of the target (Lemma 1 of the paper):

    $$f^*(\mathbf{u}) = \mathbb{E}_{t \mid \mathbf{u}} \big[ f^*_t(\mathbf{u}) \big].$$

    The optimal noise-blind model is a time-average of the optimal
    time-conditional one, weighted by the posterior $p(t \mid \mathbf{u})$.
    """)
    return

@app.cell
def _trick_md(mo):
    mo.md(r"""
    Here's the trick that makes everything visualizable. When the data is a
    finite set of points $\{\mathbf{x}_k\}_{k=1}^N$, both the conditional
    denoiser and the posterior $p(t \mid \mathbf{u})$ collapse to closed-form
    sums over data points (Appendix A.3, Eq 35-36):

    $$D^*_t(\mathbf{u}) = \sum_{k=1}^N w_k(\mathbf{u}, t)\,\mathbf{x}_k, \qquad w_k(\mathbf{u}, t) = \frac{\exp(-\|\mathbf{u}-a(t)\mathbf{x}_k\|^2 / 2b(t)^2)}{\sum_j \exp(-\|\mathbf{u}-a(t)\mathbf{x}_j\|^2 / 2b(t)^2)}.$$

    This is the Boltzmann-softmax barycenter of the data, weighted by Gaussian
    likelihood. Plug it into the autonomous formula and we get $f^*(\mathbf{u})$
    exactly — no training, no SGD, no architecture choices. Most of this
    notebook runs on this analytical anchor.
    """)
    return

@app.cell
def _closed_form_field(G, np):
    X = G.make_circles(n=120, seed=0)
    t_grid = np.linspace(0.05, 0.95, 18)
    XX_cf, YY_cf, field_cf = G.field_grid(X, t_grid, sched_fn=G.fm_schedule, lim=1.4, n=22)
    return X, XX_cf, YY_cf, field_cf, t_grid

@app.cell
def _closed_form_plot(STABLE, X, XX_cf, YY_cf, field_cf, np, plt):
    def _draw():
        fig, ax = plt.subplots(figsize=(5.4, 5.4), constrained_layout=True)
        u, v = field_cf[..., 0], field_cf[..., 1]
        mag = np.linalg.norm(field_cf, axis=-1)
        ax.quiver(XX_cf, YY_cf, u, v, mag, cmap="viridis",
                    scale_units="xy", scale=mag.max() * 1.5 + 1e-6,
                    width=0.005, headwidth=4, headlength=5,
                    pivot="mid", alpha=0.85)
        ax.scatter(X[:, 0], X[:, 1], s=10, c=STABLE,
                    edgecolors="white", linewidths=0.5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([]); ax.set_yticks([])
        return fig
    _draw()
    return

@app.cell
def _closed_form_caption(mo):
    mo.md(r"""
    **The optimal autonomous field $f^*(\mathbf{u})$ on a 2D dataset of 120
    points (two concentric rings).** Every arrow is a sum of softmax-weighted
    contributions from the data, integrated against the implicit posterior
    $p(t \mid \mathbf{u})$. No network was trained. The field bends gracefully
    toward the manifold from far away and squeezes through the gap between the
    rings near the origin.

    Now we ask: *what is this field doing, geometrically?*
    """)
    return

@app.cell
def _decomp_md(mo):
    mo.md(r"""
    ## Decomposing the Field

    For *any* affine schedule, the optimal autonomous field $f^*(\mathbf{u})$
    decomposes into exactly three geometric components (Eq 14 of the paper,
    derived in Appendix D):
    """)
    return

@app.cell
def _decomp_equation(mo):

    mo.md(r"""
    $$f^*(\mathbf{u}) = \underbrace{\textcolor{#1B9E77}{\overline{\lambda}(\mathbf{u})\,\nabla E_{\mathrm{marg}}(\mathbf{u})}}_{\text{Natural Gradient}} + \underbrace{\textcolor{#7570B3}{\mathbb{E}_{t|\mathbf{u}}\big[(\lambda(t) - \overline{\lambda}(\mathbf{u}))(\nabla E_t(\mathbf{u}) - \nabla E_{\mathrm{marg}}(\mathbf{u}))\big]}}_{\text{Transport Correction}} + \underbrace{\textcolor{#D95F02}{\overline{c}_{\mathrm{scale}}(\mathbf{u}) \cdot \mathbf{u}}}_{\text{Linear Drift}}.$$

    The <span style="color:#1B9E77">**Natural Gradient**</span> term pulls
    toward the manifold along the steepest descent direction of the marginal
    energy $E_{\mathrm{marg}}(\mathbf{u}) = -\log p(\mathbf{u})$ — but it's
    preconditioned by the *effective gain* $\overline{\lambda}(\mathbf{u})$.
    The <span style="color:#7570B3">**Transport Correction**</span> is a
    covariance term that vanishes when the posterior $p(t \mid \mathbf{u})$
    concentrates. The <span style="color:#D95F02">**Linear Drift**</span>
    is a radial scaling that handles the schedule's expanding noise volume.

    We render each piece separately on the 2D grid below.
    """)
    return

@app.cell
def _decomp_compute(G, X, t_grid):
    XX_d, YY_d, natural, transport, drift = G.decompose_field_grid(
        X, t_grid, sched_fn=G.fm_schedule, lim=1.4, n=18
    )
    return XX_d, YY_d, drift, natural, transport

@app.cell
def _decomp_plot(
    ACCENT,
    STABLE,
    UNSTABLE,
    X,
    XX_d,
    YY_d,
    drift,
    natural,
    np,
    plt,
    transport,
):
    def _draw():
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5),
                                  sharex=True, sharey=True, constrained_layout=True)
        components = [
            ("Natural Gradient",     natural,    STABLE),
            ("Transport Correction", transport,  ACCENT),
            ("Linear Drift",         drift,      UNSTABLE),
        ]
        max_mag = max(np.linalg.norm(v, axis=-1).max() for _, v, _ in components)
        for ax, (name, vec, color) in zip(axes, components):
            u, v = vec[..., 0], vec[..., 1]
            mag = np.linalg.norm(vec, axis=-1)
            ax.quiver(XX_d, YY_d, u, v, mag, cmap="viridis",
                        scale_units="xy", scale=max_mag * 1.5 + 1e-6,
                        width=0.005, headwidth=4, headlength=5,
                        pivot="mid", alpha=0.85)
            ax.scatter(X[:, 0], X[:, 1], s=4, c="#222222", alpha=0.7, zorder=5)
            ax.set_title(name, color=color, fontsize=11)
            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([]); ax.set_yticks([])
        return fig
    _draw()
    return

@app.cell
def _decomp_caption(mo):
    mo.md(r"""
    Each color matches the equation above. Notice the
    <span style="color:#009688">Natural Gradient</span> already does most of
    the work — it points toward the data. The
    <span style="color:#542788">Transport Correction</span> is a small
    rotational nudge, largest in the *gap* between the rings where the
    posterior over noise levels is genuinely ambiguous. The
    <span style="color:#F5A623">Linear Drift</span> is a radial term that
    grows with $\|\mathbf{u}\|$.

    The paper proves the Transport Correction vanishes in two regimes: high
    ambient dimension (Sec 5.2 — the "blessings of dimensionality") and
    proximity to the manifold (Sec 5.3). When transport vanishes, the field
    simplifies to a pure preconditioned natural gradient. **This is the
    Riemannian gradient flow.**

    And there we have it — the autonomous field in three pieces. But there's
    a problem. $\nabla E_{\mathrm{marg}}$ is supposed to *blow up* at the
    manifold. How is the field still well-behaved?
    """)
    return

@app.cell
def _conformal_md(mo):
    mo.md(r"""
    ## The Conformal Metric

    The marginal energy $E_{\mathrm{marg}}(\mathbf{u}) = -\log p(\mathbf{u})$
    has an infinitely deep well at the data manifold (Eq 12, paper Fig 1).
    Its gradient diverges:

    $$\lim_{\mathbf{u} \to \mathbf{x}_k} \|\nabla E_{\mathrm{marg}}(\mathbf{u})\| = \infty.$$

    And yet the autonomous field stays bounded. What gives?

    The effective gain $\overline{\lambda}(\mathbf{u})$ — the preconditioner
    in front of the Natural Gradient term — vanishes at *exactly the rate*
    the gradient diverges. Their product is bounded. The field implicitly
    carries a local Riemannian metric, computed by the geometry of the data
    itself, that converts an infinitely steep potential well into a stable
    attractor.

    Here is that statement, rendered.
    """)
    return

@app.cell
def _conformal_compute(G, X, t_grid):
    XX_c, YY_c, lam_bar, grad_norm, _grad_field = G.conformal_grid(
        X, t_grid, sched_fn=G.fm_schedule, lim=1.5, n=40
    )
    product = lam_bar * grad_norm
    return grad_norm, lam_bar, product

@app.cell
def _conformal_plot(STABLE, UNSTABLE, X, grad_norm, lam_bar, np, plt, product):
    def _draw():
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5),
                                  sharex=True, sharey=True, constrained_layout=True)
        panels = [
            (r"$\|\nabla E_{\mathrm{marg}}(\mathbf{u})\|$  —  diverges",
                np.clip(grad_norm, 0, np.percentile(grad_norm, 99)),
                "Oranges", UNSTABLE),
            (r"$\overline{\lambda}(\mathbf{u})$  —  vanishes at the manifold",
                lam_bar, "BuGn", STABLE),
            (r"$\overline{\lambda}(\mathbf{u}) \cdot \|\nabla E_{\mathrm{marg}}\|$  —  bounded",
                np.clip(product, 0, np.percentile(product, 99)),
                "viridis", "#222222"),
        ]
        for ax, (title_, field_, cmap_, color_) in zip(axes, panels):
            im = ax.imshow(field_, cmap=cmap_,
                            extent=[-1.5, 1.5, -1.5, 1.5], origin="lower")
            ax.scatter(X[:, 0], X[:, 1], s=4, c="#111111", alpha=0.7, zorder=5)
            ax.set_title(title_, fontsize=10, color=color_)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        return fig
    _draw()
    return

@app.cell
def _conformal_caption(mo):
    mo.md(r"""
    Read these three panels left-to-right.

    - <span style="color:#F5A623">**Left.**</span> The raw marginal-energy
      gradient. As you approach a data point (the dark dots), the magnitude
      explodes. Plain gradient descent on this landscape is doomed.
    - <span style="color:#009688">**Center.**</span> The effective gain
      $\overline{\lambda}(\mathbf{u})$ — paper Eq 15. It *vanishes* exactly
      where the gradient diverges, with the right asymptotic rate (Appendix
      E).
    - **Right.** The product. **Bounded.** No singularity. This is the
      Riemannian preconditioning that the paper proves and we now see.

    This is the central reveal of the paper, made visible. A free Riemannian
    preconditioner, courtesy of the geometry. Everything that follows —
    phase diagrams, parameterization choices, failure modes — is about
    *where this cancellation works* and *where it stops working*.

    But there's more. The cancellation depends on the *posterior* over noise
    levels, $p(t \mid \mathbf{u})$, and that posterior changes with the
    ambient dimension $D$. The field should look very different at $D = 2$
    than at $D = 128$.
    """)
    return

@app.cell
def _field_D_md(mo):
    mo.md(r"""
    ## Example: The Field Across Ambient Dimensions

    The conformal-metric story has a knob: ambient dimension. The paper proves
    (Sec 5.2, Lemma 5) that the Transport Correction term *vanishes* as $D$
    grows, because the noise-shell concentrates in $\mathbb{R}^D$. So the
    field should look messier at low $D$, cleaner at high $D$ — settling into
    pure radial flow toward the manifold.

    Here is the closed-form $f^*(\mathbf{u})$ at four ambient dimensions, all
    rendered in the original 2D data plane (we project up via a random
    orthogonal $P$, evaluate, project back).
    """)
    return

@app.cell
def _field_D_data(G, X, np, t_grid):
    D_panel_values = [2, 8, 32, 128]
    panel_results = []
    for D_val in D_panel_values:
        rng_p = np.random.default_rng(0)
        Mp = rng_p.standard_normal((D_val, 2)).astype(np.float32)
        Pp, _ = np.linalg.qr(Mp)
        XD_p = (X @ Pp.T).astype(np.float32) if D_val != 2 else X
        n_p = 16
        gp = np.linspace(-1.4, 1.4, n_p).astype(np.float32)
        XX_p, YY_p = np.meshgrid(gp, gp)
        UV_p = np.zeros((n_p, n_p, 2), dtype=np.float32)
        for ip, xp in enumerate(gp):
            for jp, yp in enumerate(gp):
                u_D_p = (Pp @ np.array([xp, yp], dtype=np.float32)) if D_val != 2 \
                        else np.array([xp, yp], dtype=np.float32)
                f_D_p = G.f_star_discrete(u_D_p, XD_p, t_grid, G.fm_schedule)
                UV_p[jp, ip] = (Pp.T @ f_D_p) if D_val != 2 else f_D_p
        panel_results.append((D_val, XX_p, YY_p, UV_p))
    return (panel_results,)

@app.cell
def _field_D_plot(STABLE, X, np, panel_results, plt):
    def _draw():
        fig, axes = plt.subplots(1, 4, figsize=(15, 4),
                                  sharex=True, sharey=True, constrained_layout=True)
        for ax, (D_val, XXp, YYp, UVp) in zip(axes, panel_results):
            mag = np.linalg.norm(UVp, axis=-1)
            ax.quiver(XXp, YYp, UVp[..., 0], UVp[..., 1], mag,
                        cmap="viridis", scale_units="xy",
                        scale=mag.max() * 1.4 + 1e-6,
                        width=0.005, headwidth=4, headlength=5,
                        pivot="mid", alpha=0.85)
            ax.scatter(X[:, 0], X[:, 1], s=8, c=STABLE,
                        edgecolors="white", linewidths=0.4, zorder=5)
            ax.set_title(f"$D = {D_val}$", fontsize=11)
            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([]); ax.set_yticks([])
        return fig
    _draw()
    return

@app.cell
def _field_D_caption(mo):
    mo.md(r"""
    Read left to right. At $D=2$ the field is messy near the manifold —
    the Transport Correction is large, the posterior $p(t \mid \mathbf{u})$
    is genuinely ambiguous. At $D=128$, every arrow points cleanly inward;
    the noise shell has concentrated, the correction has vanished, and the
    field is *pure radial flow* on the marginal energy. The geometry of
    high-dimensional Gaussians does the work.

    The "blessings of dimensionality", made visual.
    """)
    return

@app.cell
def _alpha_md(mo):
    mo.md(r"""
    ## A Continuous Family of Parameterizations

    So far we've worked with one target: the velocity prediction
    $\mathbf{v} = \boldsymbol{\varepsilon} - \mathbf{x}$. The paper analyzes
    three discrete choices — noise prediction ($\boldsymbol{\varepsilon}$),
    velocity ($\mathbf{v}$), and signal prediction ($\mathbf{x}$) — and
    classifies them by Table 2: which give bounded gain, which give bounded
    drift, which are stable. But the choice doesn't have to be discrete.

    Define a one-parameter family interpolating between noise and velocity:

    $$\text{target}(\alpha) = \alpha\,\boldsymbol{\varepsilon} + (1 - \alpha)\,\mathbf{v}, \qquad \alpha \in [0, 1].$$

    $\alpha = 0$ is velocity. $\alpha = 1$ is noise prediction. What lies in
    between?

    We trained a small MLP at every cell of the grid $\alpha \in \{0, 0.25,
    0.5, 0.75, 1\} \times D \in \{2, 4, 8, 16, 32, 64\}$, sampled 400 points
    via Euler ODE, and measured sliced Wasserstein-2 distance to the ground
    truth. Each cell of the heatmap below is one trained network.
    """)
    return

@app.cell
def _phase_load(gon_data, np):
    _phase = gon_data.load("exp13_iconic")

    _D_full = _phase["D_dense"]
    _keep = _D_full <= 32
    W2_phase = _phase["W2_dense"][:, _keep]
    alpha_phase = _phase["alpha_dense"]
    D_phase = _D_full[_keep]
    truth_2d = _phase["truth_2d"]
    corner_scatters = {
        (0.0,  2):  _phase["velocity_lowD"],
        (1.0,  2):  _phase["noise_lowD"],
        (0.0, 32):  _phase["velocity_highD"],
        (1.0, 32):  _phase["noise_highD"],
    }
    corner_w2 = {
        (0.0,  2):  float(_phase["w2_velocity_lowD"]),
        (1.0,  2):  float(_phase["w2_noise_lowD"]),
        (0.0, 32):  float(_phase["w2_velocity_highD"]),
        (1.0, 32):  float(_phase["w2_noise_highD"]),
    }
    return D_phase, W2_phase, alpha_phase, corner_scatters, corner_w2, truth_2d

@app.cell
def _phase_slider(D_phase, mo):

    D_select = mo.ui.slider(
        steps=[int(d) for d in D_phase],
        value=8,
        label="ambient dimension  $D$",
        show_value=True,
    )
    return (D_select,)

@app.cell
def _phase_plot(
    ACCENT,
    D_phase,
    D_select,
    STABLE,
    UNSTABLE,
    W2_phase,
    alpha_phase,
    corner_scatters,
    corner_w2,
    np,
    plt,
    truth_2d,
):
    def _draw():

        from matplotlib.patches import ConnectionPatch
        from matplotlib import patheffects, colormaps
        import matplotlib.colors as mcolors

        D_pick = int(D_select.value)
        D_idx = int(np.argmin(np.abs(D_phase.astype(float) - D_pick)))

        fig = plt.figure(figsize=(13, 7.4))

        ax = fig.add_axes([0.06, 0.42, 0.46, 0.53])
        log_D = np.log2(D_phase.astype(float))
        AA, DD = np.meshgrid(alpha_phase, log_D, indexing="ij")
        ylgnbu = colormaps["YlGnBu"]
        pc = ax.pcolormesh(AA, DD, W2_phase, cmap=ylgnbu,
                            shading="gouraud", vmin=0.05, vmax=0.20)

        ax.set_xlabel(r"$\alpha$  (0 = velocity, 1 = noise pred.)",
                       fontsize=10.5, labelpad=4)
        ax.set_ylabel(r"ambient dim  $D$ (log$_2$)",
                       fontsize=10.5, labelpad=4)
        ax.set_yticks(np.log2([2, 4, 8, 16, 32]))
        ax.set_yticklabels([2, 4, 8, 16, 32])
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(0, 1); ax.set_ylim(np.log2(2), np.log2(32))

        sticker = dict(boxstyle="round,pad=0.32",
                        facecolor="white", edgecolor="none", alpha=0.78)
        ax.text(0.50, np.log2(22),
                "concentration regime\n(parameterization barely matters)",
                fontsize=9, color="#333", style="italic",
                ha="center", va="center", bbox=sticker, zorder=6)
        ax.text(0.50, np.log2(2.5), "low-$D$ ambiguity",
                fontsize=8.5, color=ACCENT,
                ha="center", va="center", bbox=sticker, zorder=6)

        ORANGE = "#FF6C00"
        y_pick = np.log2(D_phase[D_idx].astype(float))
        ax.axhline(y_pick, color=ORANGE, linewidth=1.2, alpha=0.55,
                    zorder=8)
        for x_end in (0.0, 1.0):
            ax.plot([x_end], [y_pick], marker="o", markersize=10,
                     mfc=ORANGE, mec="white", mew=2.0,
                     clip_on=False, zorder=11,
                     path_effects=[patheffects.Stroke(linewidth=4,
                                                      foreground="white"),
                                    patheffects.Normal()])

        callouts = [
            ("A", 0.0,  np.log2(2),   0.04, np.log2(2.55),  "left",  "bottom"),
            ("B", 1.0,  np.log2(2),   0.96, np.log2(2.55),  "right", "bottom"),
            ("C", 0.0,  np.log2(32),  0.04, np.log2(25.0),  "left",  "top"),
            ("D", 1.0,  np.log2(32),  0.96, np.log2(25.0),  "right", "top"),
        ]
        callout_dot_positions = []
        for letter, a_dot, y_dot, lx, ly, ha_, va_ in callouts:
            a_inside = max(a_dot, 0.0) + (0.012 if a_dot == 0.0 else -0.012)
            y_inside = y_dot + (0.08 if y_dot < np.log2(8) else -0.08)
            ax.scatter([a_inside], [y_inside], s=70, facecolor="white",
                        edgecolor="black", linewidths=1.0, zorder=10)
            ax.scatter([a_inside], [y_inside], s=12, facecolor="black",
                        zorder=11)
            ax.text(lx, ly, letter,
                    fontsize=13, fontweight="bold", color="#222",
                    ha=ha_, va=va_, zorder=12,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor="none",
                              alpha=0.85))
            callout_dot_positions.append((letter, a_inside, y_inside))

        bx = fig.add_axes([0.61, 0.42, 0.36, 0.53])

        D_norm = mcolors.Normalize(vmin=log_D.min(), vmax=log_D.max())
        for j in range(len(D_phase)):
            base = ylgnbu(D_norm(log_D[j]))
            is_active = (j == D_idx)
            color = base if is_active else (*base[:3], 0.45)
            lw = 2.6 if is_active else 1.0
            zorder_ = 7 if is_active else 3
            bx.plot(alpha_phase, W2_phase[:, j],
                     color=color, linewidth=lw, zorder=zorder_,
                     marker="o" if is_active else None,
                     markersize=5 if is_active else 0,
                     clip_on=False)
            bx.text(alpha_phase[-1] + 0.015, W2_phase[-1, j],
                    f"$D={int(D_phase[j])}$", fontsize=8.5,
                    color=base if is_active else (*base[:3], 0.6),
                    fontweight="bold" if is_active else "normal",
                    ha="left", va="center")

        bx.text(0.02, W2_phase[:, D_idx].max() * 1.04,
                f"$D = {int(D_phase[D_idx])}$",
                fontsize=12, color="#222", fontweight="bold",
                ha="left", va="bottom")

        bx.set_xlabel(r"$\alpha$", fontsize=10.5, labelpad=4)
        bx.set_ylabel(r"$W_2$ to ground truth",
                       fontsize=10.5, labelpad=4)
        bx.set_xlim(0, 1.18)
        bx.set_xticks([0.0, 0.5, 1.0])
        bx.grid(True, axis="y", alpha=0.20, linewidth=0.5)
        bx.spines["top"].set_visible(False)
        bx.spines["right"].set_visible(False)

        inset_order = [
            ("A", (0.0,  2),  r"$\alpha=0,\ D=2$"),
            ("B", (1.0,  2),  r"$\alpha=1,\ D=2$"),
            ("C", (0.0, 32),  r"$\alpha=0,\ D=32$"),
            ("D", (1.0, 32),  r"$\alpha=1,\ D=32$"),
        ]
        envelope_x0 = 0.06
        envelope_w = 0.91
        inset_w = 0.19
        inset_h = 0.21
        gap = (envelope_w - 4 * inset_w) / 3
        y_inset = 0.05
        inset_centers_top = []
        for i, (letter, (a, D), lbl) in enumerate(inset_order):
            x_inset = envelope_x0 + i * (inset_w + gap)
            sub = fig.add_axes([x_inset, y_inset, inset_w, inset_h])
            sub.scatter(truth_2d[:, 0], truth_2d[:, 1],
                         s=3, c="#cccccc", alpha=0.7)
            samples = corner_scatters[(a, D)]
            color = STABLE if a == 0.0 else UNSTABLE
            sub.scatter(samples[:, 0], samples[:, 1],
                         s=3, c=color, alpha=0.55)
            sub.set_xlim(-1.3, 1.3); sub.set_ylim(-1.3, 1.3)
            sub.set_aspect("equal")
            sub.set_xticks([]); sub.set_yticks([])
            for s in sub.spines.values():
                s.set_color("#d0d0d0"); s.set_linewidth(0.8)
            w2 = corner_w2[(a, D)]
            sub.text(0.5, 1.20, letter, transform=sub.transAxes,
                      fontsize=16, fontweight="bold", color="#222",
                      ha="center", va="bottom")
            sub.text(0.5, 1.02, f"{lbl}    $W_2 = {w2:.2f}$",
                      transform=sub.transAxes, fontsize=9,
                      color=color, ha="center", va="bottom")
            inset_centers_top.append(
                (letter, x_inset + inset_w / 2, y_inset + inset_h + 0.085))

        return fig
    _draw()
    return

@app.cell
def _phase_slider_display(D_select):

    D_select
    return

@app.cell
def _phase_caption(mo):
    mo.md(r"""
    Read it like a weather map. **Lighter colors are better** (lower $W_2$
    to ground truth); darker is worse. Drag the slider above to pick a
    different $D$ — the right panel shows that row of the heatmap as a curve,
    with the chosen $D$ bolded. Each labeled corner has a matching
    sample-scatter panel below.

    - <span style="color:#009688">**A** ($\alpha = 0$, $D = 2$):</span>
      velocity at low dim. The conformal metric does its full work: tight
      rings, $W_2 \approx 0.10$.
    - <span style="color:#F5A623">**B** ($\alpha = 1$, $D = 2$):</span>
      noise prediction at low dim. Diffuse cloud, $W_2 \approx 0.20$.
      *Twice as bad* — same compute, same data, just a different target.
    - <span style="color:#009688">**C**</span> and
      <span style="color:#F5A623">**D**</span>
      ($D = 32$): every parameterization wins. The shell of noise has
      concentrated to a Dirac delta in the posterior, the Transport
      Correction has vanished, and the field is pure radial flow no matter
      what target you train on.

    The paper's binary table (Sec 6) said "noise prediction fails
    autonomously." The continuous picture is sharper: noise prediction
    fails *worst at low $D$*, and the failure fades smoothly as $D$ grows.
    The cancellation works wherever it can.
    """)
    return

@app.cell
def _jensen_md(mo):
    mo.md(r"""
    ## The Limits of the Cancellation

    The conformal metric gives the autonomous field a free preconditioner —
    *for the velocity target*. For the noise target, the paper proves
    (Eq 66, Appendix F.2) that an extra error term appears, called the
    **Jensen Gap**:

    $$\Delta v_{\text{noise}} \propto \|\mathbf{u} - \mathbf{x}^*\| \cdot \left|\frac{b'(t)}{b(t)}\right| \cdot \left|b(t)\,\mathbb{E}_{\tau \mid \mathbf{u}}[1/b(\tau)] - 1\right|.$$

    The middle factor is harmless. The outer factors are the problem: as
    $t \to 0$, $b(t) \to 0$, so $b'(t)/b(t) \to \infty$. And the posterior
    $p(\tau \mid \mathbf{u})$ is broad enough that
    $\mathbb{E}_{\tau \mid \mathbf{u}}[1/b(\tau)]$ doesn't equal $1/b(t)$
    by Jensen's inequality. The gap explodes precisely where you most want
    the field to be precise: near the data.

    The paper introduces this quantity, names it, and never plots it. We do.
    Three heatmaps, at increasingly low $t$, make the divergence visible.
    """)
    return

@app.cell
def _jensen_compute(G, X, np):
    t_grid_j = np.linspace(0.02, 0.95, 22)
    jensen_panels = []
    for t_e in [0.60, 0.20, 0.05]:
        XXj, YYj, gap_j = G.jensen_gap_grid(
            X, t_grid_j, sched_fn=G.fm_schedule,
            t_eval=t_e, lim=1.5, n=36
        )
        jensen_panels.append((t_e, XXj, YYj, gap_j))
    return (jensen_panels,)

@app.cell
def _jensen_plot(X, jensen_panels, np, plt):
    def _draw():
        vmax_j = max(np.percentile(g, 98) for _, _, _, g in jensen_panels)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3),
                                  sharey=True, constrained_layout=True)
        im = None
        for ax, (t_e, _XXj, _YYj, gap_j) in zip(axes, jensen_panels):
            im = ax.imshow(np.clip(gap_j, 0, vmax_j), cmap="magma_r",
                            extent=[-1.5, 1.5, -1.5, 1.5], origin="lower",
                            vmin=0, vmax=vmax_j)
            ax.scatter(X[:, 0], X[:, 1], s=4, c="cyan", alpha=0.8)
            ax.set_title(rf"$t = {t_e}$", fontsize=11)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02,
                             label=r"$\Delta v_{\mathrm{noise}}$ (Jensen Gap)")
        cbar.outline.set_visible(False)
        return fig
    _draw()
    return

@app.cell
def _jensen_caption(mo):
    mo.md(r"""
    At $t = 0.60$, with comfortably high noise, the Jensen Gap is small
    everywhere. At $t = 0.20$ it grows, especially near the data manifold.
    At $t = 0.05$, near clean data, the gap **explodes** near the
    rings. Every cyan dot is sitting in a thunderstorm of error.

    And we arrive at the disappointing conclusion that for the noise target,
    this gift comes at a cost: the Jensen Gap dominates the loss precisely
    where the field needs to be most accurate.
    """)
    return

@app.cell
def _gallery_md(mo):
    mo.md(r"""
    ## Example: Moons, Mixtures, and a Swiss Roll

    Concentric circles are a clean toy. Real manifolds aren't. We re-ran
    the $(\alpha, D)$ phase diagram on three more 2D datasets:

    - **Moons** — smooth, semicircular, two components.
    - **8-GMM ring** — eight discrete modes around a ring.
    - **Swiss roll** — a curved, single-component manifold.

    Every cell in the grid below is a freshly trained network on that
    dataset's $(\alpha, D)$ configuration.
    """)
    return

@app.cell
def _gallery_load(gon_data, np):
    _gal = gon_data.load("exp12_failure_grids")
    grids_gal = {
        "moons":      _gal["moons"],
        "8-GMM ring": _gal["8-GMM ring"],
        "swiss":      _gal["swiss"],
    }
    alpha_gal = _gal["alpha_vals"]
    D_gal = _gal["D_vals"]
    return D_gal, alpha_gal, grids_gal

@app.cell
def _gallery_plot(D_gal, alpha_gal, grids_gal, plt):
    def _draw():

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                                  sharey=True, constrained_layout=True)

        D_colors = {2: "#7B2D26", 8: "#C97B63", 32: "#009688"}

        ymax = max(g.max() for g in grids_gal.values()) * 1.10
        ymin = 0.0

        for ax, (name, grid) in zip(axes, grids_gal.items()):

            for j, D in enumerate(D_gal):
                ys = grid[:, j]
                ax.plot(alpha_gal, ys, marker="o", markersize=5,
                        linewidth=2.0, color=D_colors[int(D)],
                        clip_on=False, zorder=5)

                ax.text(alpha_gal[-1] + 0.05, ys[-1],
                        f"$D={int(D)}$",
                        color=D_colors[int(D)],
                        fontsize=9.5, fontweight="bold",
                        ha="left", va="center")

            ax.set_title(name, fontsize=11)
            ax.set_xlabel(r"$\alpha$    "
                           r"(0 = velocity, 1 = noise pred.)")
            ax.set_xlim(-0.05, 1.30)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            if ax is axes[0]:
                ax.set_ylabel(r"$W_2$ to ground truth")
        return fig
    _draw()
    return

@app.cell
def _gallery_caption(mo):
    mo.md(r"""
    The same shape in all three. At $D = 2$, velocity beats noise prediction
    by a factor of 2× to 3.5×. At $D = 32$, the row flattens — every
    parameterization wins. Swiss roll has the steepest gradient, exactly
    as the curvature argument predicts: a curved manifold breaks the
    posterior concentration faster.

    The conformal-metric story is geometry, not a property of circles.
    """)
    return

@app.cell
def _collapse_md(mo):
    mo.md(r"""
    ## There's One More Question Worth Asking

    The phase diagram tells us velocity wins low-$D$ and noise wins
    (eventually) nowhere. Surely, then, the *right* thing to do is let the
    network choose its own $\alpha$ per point. Train an MLP that outputs both
    the target prediction and a learned
    $\alpha(\mathbf{x}, t) \in [0, 1]$, with the same MSE flavor:

    $$\mathcal{L} = \mathbb{E}\big\|f_\theta(\mathbf{u}) - [\alpha_\theta(\mathbf{u}, t)\boldsymbol{\varepsilon} + (1 - \alpha_\theta(\mathbf{u}, t))\mathbf{v}]\big\|^2.$$

    The network would surely find the good parameterization at every point.
    """)
    return

@app.cell
def _collapse_data(gon_data, np):
    _coll = gon_data.load("exp14_alpha_fields")
    alpha_naive = _coll["alpha_naive"]
    alpha_nu = _coll["alpha_nu"]
    alpha_poly = _coll["alpha_poly"]
    coll_X = _coll["data_X"]

    _n_grid = alpha_naive.shape[0]
    _g = np.linspace(-1.5, 1.5, _n_grid).astype(np.float32)
    _XX, _YY = np.meshgrid(_g, _g)
    _pts = np.stack([_XX.ravel(), _YY.ravel()], axis=1)
    _d2 = ((_pts[:, None, :] - coll_X[None, :, :]) ** 2).sum(-1)
    _dist = np.sqrt(_d2.min(axis=1)).reshape(_n_grid, _n_grid)
    alpha_ideal = (0.85 * np.exp(-_dist / 0.18)).astype(np.float32)
    return alpha_ideal, alpha_naive, alpha_nu, alpha_poly, coll_X

@app.cell
def _collapse_select(mo):

    config_select = mo.ui.dropdown(
        options=[
            "naive MSE",
            "ν(t)-regularized MSE",
            "structural polynomial in t",
            "hypothetical ideal (spatial variation)",
        ],
        value="naive MSE",
        label="parameterization scheme",
    )
    return (config_select,)

@app.cell
def _collapse_plot(
    ACCENT,
    STABLE,
    UNSTABLE,
    alpha_ideal,
    alpha_naive,
    alpha_nu,
    alpha_poly,
    coll_X,
    config_select,
    plt,
):
    def _draw():

        choice = config_select.value
        if choice == "naive MSE":
            A = alpha_naive; tag = "α ≈ 1 (noise prediction)"; tag_c = UNSTABLE
        elif choice == "ν(t)-regularized MSE":
            A = alpha_nu;    tag = "α ≈ 0 (velocity)";          tag_c = STABLE
        elif choice == "structural polynomial in t":
            A = alpha_poly;  tag = "α ≈ 1 (noise prediction)";  tag_c = UNSTABLE
        else:
            A = alpha_ideal
            tag = "spans the full range";                       tag_c = ACCENT

        fig, ax = plt.subplots(figsize=(8.2, 5.2),
                                constrained_layout=True)
        im = ax.imshow(A, cmap="RdBu_r", vmin=0.0, vmax=1.0,
                        extent=[-1.5, 1.5, -1.5, 1.5], origin="lower")

        ax.scatter(coll_X[:, 0], coll_X[:, 1], s=14,
                    facecolor="white", edgecolor="black",
                    linewidths=0.8, alpha=0.5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#000"); s.set_alpha(0.30); s.set_linewidth(0.8)

        ax.text(0.02, 1.06,
                f"{choice}",
                transform=ax.transAxes, fontsize=12,
                fontweight="bold", color="#222", ha="left", va="bottom")
        ax.text(0.02, 1.01,
                fr"$\bar{{\alpha}} = {A.mean():.3f}$  ·  "
                fr"$\sigma_\alpha = {A.std():.3f}$  ·  {tag}",
                transform=ax.transAxes, fontsize=10,
                color=tag_c, ha="left", va="bottom")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02,
                             label=r"$\alpha(\mathbf{u}, t = 0.05)$  "
                                   r"(0 = velocity, 1 = noise pred.)")
        cbar.outline.set_visible(False)
        return fig
    _draw()
    return

@app.cell
def _collapse_select_display(config_select):

    config_select
    return

@app.cell
def _collapse_caption(mo):
    mo.md(r"""
    All three configurations **collapse** to a constant $\alpha$ across the
    entire input space.

    - <span style="color:#F5A623">**Naive MSE**</span> →
      $\alpha \to 1$ (the *unstable* parameterization).
    - <span style="color:#009688">**$\nu(t)$-regularized**</span> →
      $\alpha \to 0$ (stable, but trivial — the regularizer trades one
      collapse for the other).
    - <span style="color:#F5A623">**Polynomial in $t$ with sign
      constraints**</span> (MuLAN-style structural prior) →
      $\alpha \to 1$ again.

    The MSE objective alone is degenerate: the loss landscape is flat in
    $\alpha$ once the prediction matches its target. Spatial variation in
    $\alpha$ is not free — it requires a *structural* anchor that the loss
    cannot smooth away. MuLAN's variational ELBO gives one. Plain
    regularizers do not.

    A negative result, honestly framed, is still a finding. **The
    conformal-metric optimum exists in the loss landscape; gradient descent
    on MSE cannot find it.** A beautiful trap — laid by the loss landscape
    itself.
    """)
    return

@app.cell
def _footnotes(mo):
    mo.md(r"""
    ---
    ### Footnotes
    """)
    mo.accordion({
        "**1.  Closed-form $f^*(\\mathbf{u})$ on discrete data (Appendix A.3)**":
            mo.md(r"""
            For a finite dataset $\{\mathbf{x}_k\}_{k=1}^K$, the conditional
            optimal target collapses to a softmax-weighted sum:

            $$D^*_t(\mathbf{u}) = \sum_k w_k(\mathbf{u}, t)\, \mathbf{x}_k, \qquad w_k(\mathbf{u}, t) \propto \exp\!\Big(\!-\tfrac{1}{2 b(t)^2}\,\|\mathbf{u} - a(t)\mathbf{x}_k\|^2\Big).$$

            The autonomous optimum then averages over the implicit posterior
            $p(t \mid \mathbf{u}) \propto \int p(\mathbf{u}, t)\, dt$, which
            is itself a discrete sum. Both pieces are renderable on a 2D
            grid in milliseconds. This is the analytical anchor of the
            entire notebook — every figure that doesn't say "trained" is
            just $K$-by-$T$ matrix arithmetic.
            """),

        "**2.  The Jensen Gap (Appendix F.2, Eq 66)**":
            mo.md(r"""
            For a noise-prediction parameterization, the autonomous error
            against the true velocity field decomposes into a Natural
            Gradient piece (which the conformal metric cancels) plus an
            extra term:

            $$\Delta v_{\text{noise}}(\mathbf{u}) = \big\|\mathbf{u} - \mathbf{x}^*\big\| \cdot \left|\frac{b'(t)}{b(t)}\right| \cdot \Big| b(t) \cdot \mathbb{E}_{\tau \mid \mathbf{u}}\!\big[1/b(\tau)\big] - 1 \Big|.$$

            The third factor is a Jensen-inequality residual: it would be
            zero if the posterior over $\tau$ were a Dirac. Since
            $1/b(\tau)$ is convex and the posterior is broad, the residual
            is positive — and the prefactor $|b'(t)/b(t)|$ blows up as
            $t \to 0$. Hence the heatmap explodes near the manifold at
            low $t$.
            """),

        "**3.  Why naïve learned-$\\alpha$ collapses (the flat-minimum argument)**":
            mo.md(r"""
            The MSE objective with a free per-point $\alpha$ is

            $$\mathcal{L}(\theta) = \mathbb{E}\big\|f_\theta(\mathbf{u}) - [\alpha_\theta\,\boldsymbol{\varepsilon} + (1 - \alpha_\theta)\,\mathbf{v}]\big\|^2.$$

            The optimum is reached *for any* $\alpha$ as long as
            $f_\theta(\mathbf{u})$ matches whichever target $\alpha$
            specifies. The set of zero-loss configurations is a
            $(\dim \alpha)$-parameter family. Adam's stochastic dynamics
            select an arbitrary point on this flat ridge — usually
            $\alpha \approx 1$, occasionally $\alpha \approx 0$ when a side
            regularizer is added. Spatial variation in $\alpha$ requires a
            *structural* anchor (e.g., MuLAN's variational ELBO) that
            penalizes constant-$\alpha$ configurations directly. Plain MSE
            cannot.
            """),
    })
    return

@app.cell
def _closer_md(mo):
    mo.md(r"""
    ## Onwards and Downwards

    What we showed.

    - The optimal autonomous field on a finite dataset has a **closed
      form** (Eq 35–36) — no training necessary. It decomposes into three
      geometric pieces: a Natural Gradient toward the manifold, a Transport
      Correction that vanishes with $D$, and a Linear Drift.
    - The Natural Gradient term divides by zero at the manifold. The paper
      proves a **conformal metric** $\overline{\lambda}(\mathbf{u})$
      vanishes at exactly the right rate to keep the field bounded. We
      rendered the cancellation directly.
    - On the continuous $(\alpha, D)$ landscape, the velocity target wins
      decisively at low $D$ and noise prediction has no regime where it
      meaningfully beats velocity. The story holds across moons, GMMs, and
      swiss rolls.
    - The Jensen Gap explodes near the manifold at low $t$. That is *why*
      noise prediction fails autonomously.
    - Asking the network to *learn* its own $\alpha$ collapses three
      different ways. Plain MSE is the wrong loss; structural priors are
      required.

    What's left.

    - **Curved ambient spaces.** Everything here lives in $\mathbb{R}^D$.
      Real data often doesn't — protein conformations on rotation groups,
      phylogenies on tree spaces, lattice configurations on tori. Whether
      the conformal metric extends or fractures under intrinsic curvature
      is the natural next question.
    - **Variational learned-$\alpha$.** If we want spatial variation, we
      probably need MuLAN's exact ELBO, not a side regularizer.
    - **The right preconditioner.** The conformal metric is *one*
      preconditioner that cancels the singularity. Is it the only one?
      The optimal one?

    Like the proverbial blind men feeling an elephant, autonomous diffusion
    is more than the sum of its parameterizations. We hope the next
    interpretation is yours.

    ---

    *Code:*
    [github.com/jacobcrainic/gon-notebook](https://github.com/jacobcrainic/gon-notebook).<br>
    *Paper:*
    [arXiv:2602.18428](https://arxiv.org/abs/2602.18428) ·
    *Cite:* Sahraee-Ardakan, Delbracio, Milanfar — *The Geometry of Noise*, Google, Feb 2026.
    """)
    return

if __name__ == "__main__":
    app.run()
