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
def _hero_plot(STABLE, UV_h, XX_h, X_hero, YY_h, mo, np, plt):
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
    mo.center(_draw())
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
        The standard view of a diffusion model is that the network needs
        to know how much noise has been added before it can decide which
        way to point. Recent autonomous models drop that information
        entirely: the network sees only the noisy observation and produces
        a single static vector field, and yet samples drawn through that
        field arrive at the clean data. How a single static arrow at every
        location can guide a sample from pure noise and also from light
        noise, while ensuring its stationary points sit on the data, is
        the question this notebook is about. The figure shows one such
        field on a 2D dataset of two interleaving rings, lifted into
        $\mathbb{R}^D$ by a random orthogonal projection; varying the
        dimension below changes how the field responds.
        """
    )
    mo.hstack([col_slider1, col_slider2, col_prose],
                justify="space-between", gap=2.0, align="start")
    return

@app.cell
def _roadmap(mo):
    mo.md(r"""
    Four stops:

    1. The optimal autonomous field on a finite dataset, written in closed
       form.
    2. A three-piece geometric decomposition, and the conformal
       cancellation that keeps it bounded.
    3. The continuous $(\alpha, D)$ phase plane that sharpens the paper's
       binary table.
    4. Where the cancellation breaks, and why letting the network learn
       its own $\alpha$ collapses.
    """)
    return

@app.cell
def _popular_story(mo):
    mo.md(r"""
    The standard account of a diffusion model goes as follows. A network is
    trained to remove noise from data across a continuum of noise levels,
    and at inference it is told the current level $t$ so that it knows
    which scale of structure it should be uncovering at each step.
    Generation begins from pure Gaussian noise and proceeds by repeated
    application of the network at gradually decreasing values of $t$, with
    the sample sharpening a little more each time, until what remains looks
    like a draw from the data distribution. The account is accurate, and
    almost every modern diffusion system can be described this way. The
    puzzle is that several recent models manage to do without it.
    """)
    return

@app.cell
def _the_break(mo):
    mo.md(r"""
    A handful of recent models remove the time input entirely. They see
    only the noisy observation $\mathbf{u}$ and emit a target vector, with
    no embedding for $t$ and no schedule channel of any kind. Equilibrium
    Matching [[Wang & Du, 2025](https://arxiv.org/abs/2510.02300)] is one
    such model, and blind Flow Matching
    [[Sun et al., 2025](https://arxiv.org/abs/2502.13129)] is another. Both
    produce samples that closely match the data distribution despite having
    no way to distinguish a heavily corrupted input from a lightly
    corrupted one. The behavior is unexpected, because the direction of
    steepest descent at $\mathbf{u}$ ought to depend strongly on how much
    noise it carries: the same arrow asked to do the work at every noise
    level should not be enough, and yet it apparently is.
    """)
    return

@app.cell
def _the_promise(mo):
    mo.md(r"""
    The recent paper *The Geometry of Noise* resolves the puzzle by
    recasting autonomous diffusion as Riemannian gradient flow on a
    marginal energy landscape. The landscape is badly behaved: the
    energy diverges at the data manifold, so the gradient one would
    naively follow grows without bound near a clean sample. The
    autonomous field, however, carries an implicit local metric, which
    the paper calls the effective gain, that contracts at the same rate
    the gradient explodes. The product remains finite, and the metric
    is supplied by the geometry of the data rather than learned.
    """)
    return

@app.cell
def _the_promise_2(mo):
    mo.md(r"""
    The remainder of this notebook works through the claim on the
    smallest model that exhibits all of the relevant behavior: a finite
    collection of points in two dimensions, optionally lifted into
    $\mathbb{R}^D$ by a random orthogonal projection. The setting is
    rich enough to support every parameterization the paper analyzes,
    and small enough that the optimal field can be written down in
    closed form and every relevant quantity plotted by hand.
    """)
    return

@app.cell
def _first_steps_md(mo):
    mo.md(r"""
    ## First Steps: The Closed-Form Optimal Field

    A natural starting point is to write the forward and reverse processes
    in the most general affine form, since this absorbs every
    parameterization the paper considers into a single set of coefficients.
    A clean point $\mathbf{x}$ is corrupted with Gaussian noise
    $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$ according to

    $$\mathbf{u}_t = a(t)\,\mathbf{x} + b(t)\,\boldsymbol{\varepsilon},$$

    so the schedule is described by two functions $a(t)$ and $b(t)$. The
    network is asked to predict a linear combination of the same two
    ingredients, $r(\mathbf{x}, \boldsymbol{\varepsilon}, t) = c(t)\,\mathbf{x} + d(t)\,\boldsymbol{\varepsilon}$,
    and the choice of the four functions $(a, b, c, d)$ is enough to
    reproduce DDPM, EDM, Flow Matching, and Equilibrium Matching as
    particular cases [Sun et al., 2025]. Within this framework, the
    optimal autonomous field admits a closed-form expression, which the
    rest of the section makes explicit.
    """)
    return

@app.cell
def _table1(mo):
    table = mo.md(r"""
    | Model | $a(t)$ | $b(t)$ | $c(t)$ | $d(t)$ | What it predicts |
    |---|---|---|---|---|---|
    | **DDPM** | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ | 0 | 1 | noise $\boldsymbol{\varepsilon}$ |
    | **EDM** | 1 | $\sigma_t$ | 1 | 0 | data $\mathbf{x}$ |
    | **Flow Matching** | $1-t$ | $t$ | $-1$ | $1$ | velocity $\boldsymbol{\varepsilon} - \mathbf{x}$ |
    | **Equilibrium Matching** | $1-t$ | $t$ | $-t$ | $t$ | $\mathbf{u} - \mathbf{x}$ |
    """)
    prose = mo.md(r"""
    The Flow Matching schedule is used throughout. The fact on which
    everything that follows depends is Lemma 1 of the paper, which states
    that an autonomous model, one without access to $t$, minimizes its MSE
    loss when its output equals the posterior expectation of the target:

    $$f^*(\mathbf{u}) = \mathbb{E}_{t \mid \mathbf{u}} \big[ f^*_t(\mathbf{u}) \big].$$

    The optimal noise-blind model is therefore a time-average of the
    optimal time-conditional one, weighted by the posterior
    $p(t \mid \mathbf{u})$ over noise levels consistent with the
    observation $\mathbf{u}$.
    """)
    mo.vstack([mo.center(table), prose], gap=1.0)
    return

@app.cell
def _trick_md(mo):
    mo.md(r"""
    When the data distribution is supported on a finite set of points
    $\{\mathbf{x}_k\}_{k=1}^N$, both the conditional denoiser and the
    posterior $p(t \mid \mathbf{u})$ reduce to closed-form sums over the
    data, as derived in Appendix A.3 of the paper (Eq 35–36):

    $$D^*_t(\mathbf{u}) = \sum_{k=1}^N w_k(\mathbf{u}, t)\,\mathbf{x}_k, \qquad w_k(\mathbf{u}, t) = \frac{\exp(-\|\mathbf{u}-a(t)\mathbf{x}_k\|^2 / 2b(t)^2)}{\sum_j \exp(-\|\mathbf{u}-a(t)\mathbf{x}_j\|^2 / 2b(t)^2)}.$$

    The conditional denoiser is therefore a softmax-weighted barycenter of
    the data points, with each weight given by a Gaussian likelihood.
    Substituting this expression into the autonomous formula above yields
    $f^*(\mathbf{u})$ exactly, so the optimal field on a finite dataset can
    be evaluated without training a network. Almost every figure in the
    sections that follow is computed in this way, and the few that involve
    trained networks are clearly indicated.
    """)
    return

@app.cell
def _closed_form_field(G, np):
    X = G.make_circles(n=120, seed=0)
    t_grid = np.linspace(0.05, 0.95, 18)
    XX_cf, YY_cf, field_cf = G.field_grid(X, t_grid, sched_fn=G.fm_schedule, lim=1.4, n=22)
    return X, XX_cf, YY_cf, field_cf, t_grid

@app.cell
def _closed_form_plot(STABLE, X, XX_cf, YY_cf, field_cf, mo, np, plt):
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
    mo.center(_draw())
    return

@app.cell
def _closed_form_caption(mo):
    mo.md(r"""
    The figure above shows the optimal autonomous field $f^*(\mathbf{u})$
    for a dataset consisting of 120 points distributed on two concentric
    rings. Each arrow is a softmax-weighted average of contributions from
    the data points, integrated against the implicit posterior over noise
    levels, and was obtained directly from the closed-form expression
    rather than learned. The field points inward toward the manifold from
    a distance and threads through the gap between the rings near the
    origin. The natural next step is to ask what it is doing in geometric
    terms.
    """)
    return

@app.cell
def _decomp_md(mo):
    mo.md(r"""
    ## Decomposing the Field

    For an arbitrary affine schedule, the optimal autonomous field
    $f^*(\mathbf{u})$ admits a decomposition into three geometric
    components. The decomposition is Eq 14 of the paper, derived in
    Appendix D, and isolates the three distinct mechanisms by which the
    field acts on $\mathbf{u}$:
    """)
    return

@app.cell
def _decomp_equation(mo):

    mo.md(r"""
    $$f^*(\mathbf{u}) = \underbrace{\textcolor{#1B9E77}{\overline{\lambda}(\mathbf{u})\,\nabla E_{\mathrm{marg}}(\mathbf{u})}}_{\text{Natural Gradient}} + \underbrace{\textcolor{#7570B3}{\mathbb{E}_{t|\mathbf{u}}\big[(\lambda(t) - \overline{\lambda}(\mathbf{u}))(\nabla E_t(\mathbf{u}) - \nabla E_{\mathrm{marg}}(\mathbf{u}))\big]}}_{\text{Transport Correction}} + \underbrace{\textcolor{#D95F02}{\overline{c}_{\mathrm{scale}}(\mathbf{u}) \cdot \mathbf{u}}}_{\text{Linear Drift}}.$$

    The first term, the <span style="color:#1B9E77">Natural Gradient</span>,
    is the direction of steepest descent on the marginal energy
    $E_{\mathrm{marg}}(\mathbf{u}) = -\log p(\mathbf{u})$, scaled by the
    effective gain $\overline{\lambda}(\mathbf{u})$ that acts as a local
    preconditioner. The
    <span style="color:#7570B3">Transport Correction</span> is a covariance
    between $\lambda(t)$ and the conditional gradient that disappears
    whenever the posterior $p(t \mid \mathbf{u})$ concentrates on a single
    noise level, and the <span style="color:#D95F02">Linear Drift</span>
    is the residual radial component left over after the energy-aligned
    terms have been factored out, given by
    $\overline{c}_{\mathrm{scale}}(\mathbf{u}) = \mathbb{E}_{t|\mathbf{u}}[c(t)/a(t)]$.
    The figure that follows plots each of the three components separately
    on the same 2D grid, with colors matched to the equation above.
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
    mo,
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
    mo.center(_draw())
    return

@app.cell
def _decomp_caption(mo):
    mo.md(r"""
    The three panels correspond to the three terms of the decomposition.
    The Natural Gradient does most of the work, pointing consistently
    toward the data from every position in the plane. The Transport
    Correction adds a small rotational adjustment, largest in the gap
    between the rings where the posterior over noise levels is genuinely
    uncertain. The Linear Drift contributes a radial term that grows
    with $\|\mathbf{u}\|$.
    """)
    return

@app.cell
def _decomp_caption_2(mo):
    mo.md(r"""
    The paper identifies two regimes in which the Transport Correction
    vanishes. The first is high ambient dimension (Sec 5.2), where the
    surface area of a Gaussian noise shell concentrates and the posterior
    over noise levels collapses onto a narrow band of $t$. The second is
    proximity to the manifold (Sec 5.3), where the same posterior
    collapses for a different reason. In either regime the field reduces
    to a preconditioned natural gradient, which is precisely the form a
    Riemannian gradient flow would take.
    """)
    return

@app.cell
def _decomp_caption_3(mo):
    mo.md(r"""
    One question remains. The marginal-energy gradient
    $\nabla E_{\mathrm{marg}}$ is unbounded near the manifold, since
    the energy itself diverges there, and yet the autonomous field is
    finite everywhere. How?
    """)
    return

@app.cell
def _conformal_md(mo):
    mo.md(r"""
    ## The Conformal Metric

    The marginal energy $E_{\mathrm{marg}}(\mathbf{u}) = -\log p(\mathbf{u})$
    has an infinitely deep well at the data manifold (Eq 12 of the paper,
    Fig 1), and its gradient diverges as $\mathbf{u}$ approaches any data
    point:

    $$\lim_{\mathbf{u} \to \mathbf{x}_k} \|\nabla E_{\mathrm{marg}}(\mathbf{u})\| = \infty.$$

    The Natural Gradient term of the field, by contrast, remains
    bounded throughout the domain, including arbitrarily close to a
    sample. The reason is that the effective gain
    $\overline{\lambda}(\mathbf{u})$, which sits in front of the Natural
    Gradient, vanishes at the same asymptotic rate as the gradient
    diverges. The product of the two is therefore bounded even though
    either factor on its own is not, and the velocity-target field
    behaves as if it were following a Riemannian gradient flow whose
    metric is supplied by the geometry of the data. The figure that
    follows renders the cancellation directly, by plotting each of the
    two factors and their product on the same domain.
    """)
    return

@app.cell
def _conformal_compute(G, X, np, t_grid):
    _idx = int(np.argmin(np.linalg.norm(X - np.array([0.9, 0.0]), axis=1)))
    x_target = X[_idx].astype(np.float32)
    direction = (x_target / np.linalg.norm(x_target)).astype(np.float32)
    r_vals = np.geomspace(1e-3, 0.5, 200).astype(np.float32)
    lam_bar, grad_norm = G.conformal_radial(
        X, t_grid, x_target, direction, r_vals
    )
    product = lam_bar * grad_norm
    return grad_norm, lam_bar, product, r_vals, x_target

@app.cell
def _conformal_plot(
    STABLE, UNSTABLE, grad_norm, lam_bar, mo, plt, product, r_vals,
):
    def _draw():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))

        ax.plot(r_vals, grad_norm, color=UNSTABLE, lw=1.4, zorder=2)
        ax.plot(r_vals, lam_bar,   color=STABLE,   lw=1.4, zorder=2)
        ax.plot(r_vals, product,   color="#111",   lw=2.0, zorder=3)

        ax.set_yscale("log")
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(1.0, 35.0)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xlabel(
            r"distance $r$ from a data point on the manifold",
            fontsize=10, labelpad=4,
        )

        x_lab = r_vals[-1] + 0.012
        ax.text(x_lab, grad_norm[-1] * 0.93,
                r"$\|\nabla E_{\mathrm{marg}}\|$",
                color=UNSTABLE, fontsize=10, va="center", ha="left")
        ax.text(x_lab, lam_bar[-1] * 1.07,
                r"$\overline{\lambda}$",
                color=STABLE, fontsize=10, va="center", ha="left")
        ax.text(x_lab, product[-1],
                r"$\overline{\lambda}\,\|\nabla E_{\mathrm{marg}}\|$",
                color="#111", fontsize=10, va="center", ha="left")

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_linewidth(0.6)
            ax.spines[s].set_color("#333")
        ax.tick_params(direction="out", length=3, width=0.6,
                        labelsize=9, colors="#333")
        ax.minorticks_off()

        fig.subplots_adjust(left=0.10, right=0.74, top=0.94, bottom=0.18)
        return fig
    mo.center(_draw())
    return

@app.cell
def _conformal_caption(mo):
    mo.md(r"""
    The figure traces the three quantities along a single radial slice
    that begins at a data point on the outer ring and runs outward away
    from the manifold, with the horizontal axis recording the distance
    $r$ from that point. The orange curve is the marginal-energy gradient
    $\|\nabla E_{\mathrm{marg}}\|$, peaking near the manifold and falling
    off as $r$ grows. The teal curve is the effective gain
    $\overline{\lambda}$ of Eq 15, which moves the opposite way.
    """)
    return

@app.cell
def _conformal_caption_2(mo):
    mo.md(r"""
    The black curve is the pointwise product of the two, and is what the
    autonomous field actually uses. Despite either factor changing by a
    factor of two or three across the slice, the product hovers in a
    narrow band on the log scale. Appendix E of the paper proves the
    asymptotic rates match exactly, and the figure makes the matching
    legible: where one factor would push a gradient method into
    instability, the other contracts at the right rate to bring the
    product back to something finite.
    """)
    return

@app.cell
def _conformal_caption_3(mo):
    mo.md(r"""
    The central claim of the paper is captured by this single curve. The
    Riemannian metric required to keep the gradient flow stable does not
    have to be designed or learned, since it is already present in the
    conditional denoiser, and the conformal cancellation is a property
    of the geometry rather than of any particular network. The remaining
    sections work out the conditions under which the cancellation holds
    and the conditions under which it begins to fail. The first such
    condition is dimensional.
    """)
    return

@app.cell
def _field_D_md(mo):
    mo.md(r"""
    ## A Blessing of Dimensions

    The conformal-metric argument has a free parameter, namely the ambient
    dimension into which the data is embedded. Lemma 5 of the paper
    (Sec 5.2) proves that the Transport Correction vanishes as $D$ grows,
    because the surface area of a Gaussian noise shell in $\mathbb{R}^D$
    concentrates more and more tightly with increasing dimension and the
    posterior $p(t \mid \mathbf{u})$ becomes nearly deterministic. The
    autonomous field at low $D$ should therefore appear noisier near the
    manifold, while the field at high $D$ should reduce to a clean radial
    flow toward the data.

    The figure below shows the closed-form $f^*(\mathbf{u})$ at four
    ambient dimensions, all visualized in the original 2D plane. Each
    panel is computed by sampling a random orthogonal map, embedding the
    data into $\mathbb{R}^D$ through it, evaluating the field there, and
    projecting the result back to two dimensions for display.
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
def _field_D_plot(STABLE, X, mo, np, panel_results, plt):
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
    mo.center(_draw())
    return

@app.cell
def _field_D_caption(mo):
    mo.md(r"""
    The four panels record the same field at four different ambient
    dimensions. At $D = 2$ the field is visibly disordered near the
    manifold, since the Transport Correction is large there and the
    posterior over noise levels is genuinely ambiguous because no
    concentration effect has yet kicked in. At $D = 128$ every arrow
    points cleanly inward, the correction term has shrunk to nothing, and
    the field reduces to the radial gradient flow predicted by the
    conformal-metric argument. The phenomenon, often called the blessing
    of dimensionality, is operating here in its purest form, and is
    responsible for much of the empirical success of high-dimensional
    diffusion.
    """)
    return

@app.cell
def _alpha_md(mo):
    mo.md(r"""
    ## The Velocity-Noise Knob

    Up to this point every figure has been computed for the velocity
    target $\mathbf{v} = \boldsymbol{\varepsilon} - \mathbf{x}$. The
    paper considers three targets — noise prediction, velocity, and
    signal prediction — and classifies each according to whether its
    gain and drift terms are bounded and whether the resulting flow is
    stable. There is no reason to restrict attention to those three
    points.
    """)
    return

@app.cell
def _alpha_md_2(mo):
    mo.md(r"""
    A continuous family between two of them can be defined as

    $$\text{target}(\alpha) = \alpha\,\boldsymbol{\varepsilon} + (1 - \alpha)\,\mathbf{v}, \qquad \alpha \in [0, 1],$$

    with $\alpha = 0$ recovering velocity prediction and $\alpha = 1$
    recovering noise prediction. Sweeping $\alpha$ continuously turns
    the binary classification of the paper into a smoothly varying
    landscape on which one can ask not only which targets work but how
    the quality varies between them.
    """)
    return

@app.cell
def _alpha_md_3(mo):
    mo.md(r"""
    To populate the landscape, a small MLP was trained at every cell of
    the grid $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$ paired with
    $D \in \{2, 4, 8, 16, 32, 64\}$. Each network was trained on the
    same dataset of two concentric rings, samples drawn by Euler
    integration of the ODE, and the sliced Wasserstein-2 distance from
    the samples to a fresh ground-truth set recorded. Every cell of the
    heatmap below is the output of one such trained network.
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
    D_phase,
    D_select,
    STABLE,
    UNSTABLE,
    W2_phase,
    alpha_phase,
    corner_scatters,
    corner_w2,
    mo,
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
                    fontsize=13, fontweight="bold", color="#111",
                    ha=ha_, va=va_, zorder=12,
                    path_effects=[
                        patheffects.Stroke(linewidth=3, foreground="white"),
                        patheffects.Normal(),
                    ])
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
    mo.center(_draw())
    return

@app.cell
def _phase_slider_display(D_select):

    D_select
    return

@app.cell
def _phase_caption(mo):
    mo.md(r"""
    The horizontal axis sweeps $\alpha$ from velocity on the left to
    noise prediction on the right; the vertical axis is the ambient
    dimension on a log scale; color encodes the sliced Wasserstein-2
    distance to the ground truth, lighter being better. Dragging the
    slider selects a row, reproduced as a line plot on the right with
    the chosen dimension highlighted. Four particular configurations,
    marked A through D, sit as scatter plots at the bottom.
    """)
    return

@app.cell
def _phase_caption_2(mo):
    mo.md(r"""
    At $D = 2$, configuration A (velocity) recovers the rings tightly
    with $W_2 \approx 0.10$, while configuration B (noise prediction)
    collapses to a diffuse cloud at $W_2 \approx 0.20$, roughly twice as
    far from the ground truth despite identical compute and identical
    data. At $D = 32$, both C and D succeed, since the posterior over
    noise levels concentrates and the distinction between targets
    disappears.
    """)
    return

@app.cell
def _phase_caption_3(mo):
    mo.md(r"""
    The heatmap interpolates smoothly between these regimes, sharpening
    the binary statement of the paper into a quantitative one: noise
    prediction fails most acutely at low ambient dimension, and the gap
    between targets closes gradually as $D$ rises. The conformal
    cancellation operates wherever the geometry permits it, and the
    parameterization choice matters only to the extent that the
    cancellation has not yet completed.
    """)
    return

@app.cell
def _jensen_md(mo):
    mo.md(r"""
    ## The Limits of the Cancellation

    The cancellation between $\overline{\lambda}$ and
    $\nabla E_{\mathrm{marg}}$ described above is exact for the velocity
    target, but it does not hold for the noise target. The paper shows in
    Eq 66 (Appendix F.2) that the autonomous error for noise prediction
    picks up an additional contribution, which the authors name the
    Jensen Gap:

    $$\Delta v_{\text{noise}} \propto \|\mathbf{u} - \mathbf{x}^*\| \cdot \left|\frac{b'(t)}{b(t)}\right| \cdot \left|b(t)\,\mathbb{E}_{\tau \mid \mathbf{u}}[1/b(\tau)] - 1\right|.$$

    The middle factor is benign, and the trouble lies in the two outer
    factors. Under the Flow Matching schedule, $b(t)$ approaches zero as
    $t \to 0$, so the ratio $b'(t)/b(t)$ grows without bound. The third
    factor is a Jensen-inequality residual that would vanish if the
    posterior over $\tau$ were a Dirac delta and is otherwise positive,
    with the residual growing as the posterior broadens. Because both
    factors are largest precisely near the manifold and at small $t$, the
    noise target carries an irreducible error in the regime where one
    would most want the field to be accurate. The paper introduces this
    quantity but does not plot it, so the three heatmaps below display
    its magnitude on the same domain as before, at three values of $t$.
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
def _jensen_plot(X, jensen_panels, mo, np, plt):
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
    mo.center(_draw())
    return

@app.cell
def _jensen_caption(mo):
    mo.md(r"""
    At $t = 0.60$, where the noise level is large, the Jensen Gap is
    uniformly small across the domain. At $t = 0.20$ the gap has grown,
    with the increase concentrated near the data manifold. At $t = 0.05$,
    almost on clean data, the gap is large in a thin annulus surrounding
    every data point and small everywhere else. The annulus is the region
    where the residual factor and the diverging $b'(t)/b(t)$ multiply, and
    the cyan markers indicating data points sit at its center. The figure
    therefore shows in concrete terms what Eq 66 expresses analytically,
    and it explains why the noise target fails in the autonomous setting:
    the parameterization is most inaccurate in exactly the region where
    one would most need it to be accurate.
    """)
    return

@app.cell
def _gallery_md(mo):
    mo.md(r"""
    ## Moons, Mixtures, and a Swiss Roll

    The dataset of two concentric rings used so far is a deliberately
    simple test case, and one might worry that the shape of the phase
    diagram is determined more by its geometry than by anything general.
    To check, the same procedure was repeated on three further 2D
    datasets: a pair of interleaving moons, an eight-component Gaussian
    mixture arranged in a ring, and a curved swiss-roll manifold. Each
    cell of the figure below is a separate MLP, trained from scratch on
    the indicated dataset and the indicated $(\alpha, D)$ configuration,
    with the resulting Wasserstein distance plotted as a function of
    $\alpha$ for each ambient dimension.
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
def _gallery_plot(D_gal, alpha_gal, grids_gal, mo, plt):
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
    mo.center(_draw())
    return

@app.cell
def _gallery_caption(mo):
    mo.md(r"""
    The three panels exhibit the same qualitative pattern. At $D = 2$,
    the velocity target is between two and three-and-a-half times closer
    to the ground truth than the noise target, with the precise gap
    depending on the dataset. At $D = 32$, the curves are essentially
    flat in $\alpha$, indicating that any parameterization works as well
    as any other once the posterior over noise levels has concentrated.
    The swiss roll exhibits the steepest dependence on $\alpha$,
    consistent with the curvature argument in the paper: a manifold with
    non-trivial curvature breaks the posterior concentration earlier
    than a flat one. The conformal-metric account is therefore reproduced
    across all three datasets, and what one is seeing is a story about
    geometry rather than a property of concentric circles.
    """)
    return

@app.cell
def _collapse_md(mo):
    mo.md(r"""
    ## A Tempting Generalization

    If the velocity target is best at low $D$ and the noise target offers
    no advantage anywhere, a natural next step is to let the network
    decide for itself which target to use at each point. One way to
    formalize this is to train an MLP whose output consists not only of
    a target prediction but also a per-point parameter
    $\alpha(\mathbf{x}, t) \in [0, 1]$, with the loss given by the same
    MSE form as before, evaluated at whichever target $\alpha$ specifies:

    $$\mathcal{L} = \mathbb{E}\big\|f_\theta(\mathbf{u}) - [\alpha_\theta(\mathbf{u}, t)\boldsymbol{\varepsilon} + (1 - \alpha_\theta(\mathbf{u}, t))\mathbf{v}]\big\|^2.$$

    One would expect the network to converge on a spatially varying
    $\alpha$, choosing values close to zero near the manifold, where
    noise prediction fails, and tolerating values close to one elsewhere,
    where any parameterization works. The figure that follows shows the
    distribution of values $\alpha$ takes across the input space for
    three different training schemes, alongside a hypothetical ideal in
    which the network would have learned to vary $\alpha$ smoothly with
    distance from the data.
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
def _collapse_plot(
    STABLE,
    alpha_ideal,
    alpha_naive,
    alpha_nu,
    alpha_poly,
    mo,
    np,
    plt,
):
    def _draw():
        configs = [
            ("Naive MSE",          alpha_naive),
            (r"$\nu(t)$-regularized", alpha_nu),
            ("Polynomial-in-$t$",  alpha_poly),
            ("Ideal",              alpha_ideal),
        ]
        bins = np.linspace(0.0, 1.0, 41)

        fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4),
                                  sharex=True, sharey=True)

        n_total = alpha_naive.size
        for ax, (name, A) in zip(axes, configs):
            ax.hist(A.ravel(), bins=bins,
                    color=STABLE, alpha=0.85,
                    edgecolor="white", linewidth=0.4)
            ax.text(0.0, 1.20, name,
                    transform=ax.transAxes,
                    ha="left", va="bottom",
                    fontsize=9.5, color="#111")
            ax.text(0.0, 1.04,
                    fr"$\bar{{\alpha}}={A.mean():.3f}$"
                    fr"   $\sigma={A.std():.3f}$",
                    transform=ax.transAxes,
                    ha="left", va="bottom",
                    fontsize=8.5, color="#666")
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([])
            ax.set_ylim(0, n_total)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.spines["bottom"].set_color("#333")
            ax.tick_params(direction="out", length=3, width=0.6,
                           labelsize=8.5, colors="#333")

        fig.supxlabel(r"$\alpha$", fontsize=10, y=0.04)
        fig.subplots_adjust(left=0.04, right=0.99, top=0.78,
                             bottom=0.22, wspace=0.18)
        return fig
    mo.center(_draw())
    return

@app.cell
def _collapse_caption(mo):
    mo.md(r"""
    Each of the three trained configurations converges to a value of
    $\alpha$ that is approximately constant across the entire input
    space. The naive MSE objective drives $\alpha$ to one and recovers
    noise prediction, which the previous sections established as the
    worse of the two endpoints. Adding a $\nu(t)$ regularizer drives
    $\alpha$ to zero and recovers velocity, but this is not a victory in
    any meaningful sense, since a constant $\alpha = 0$ is precisely
    what one would have written down by hand without the network. The
    structural polynomial-in-$t$ parameterization, which imposes the sign
    constraints used by MuLAN, also collapses to noise prediction.

    The reason the network cannot learn a useful spatially varying
    $\alpha$ is that the MSE objective with a free $\alpha$ is degenerate.
    Any value of $\alpha$ achieves the optimum as long as $f_\theta$
    matches whichever target $\alpha$ has selected, and the set of
    zero-loss configurations forms a flat ridge along which gradient
    descent can drift indefinitely. Recovering the conformal-metric
    optimum from this loss landscape requires a structural anchor that
    MSE alone cannot supply. The variational ELBO of MuLAN supplies one,
    while a $\nu(t)$ regularizer added on top of MSE does not. The
    conclusion is that the conformal-metric optimum exists in the
    function space available to the network, but the loss landscape on
    which gradient descent has been asked to find it does not select for
    it; the configuration is, in this sense, a trap laid by the loss
    landscape itself, and one that no amount of additional optimization
    will undo without a structurally different objective.
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
            For a finite dataset $\{\mathbf{x}_k\}_{k=1}^K$, the
            conditional optimal target reduces to a softmax-weighted sum:

            $$D^*_t(\mathbf{u}) = \sum_k w_k(\mathbf{u}, t)\, \mathbf{x}_k, \qquad w_k(\mathbf{u}, t) \propto \exp\!\Big(\!-\tfrac{1}{2 b(t)^2}\,\|\mathbf{u} - a(t)\mathbf{x}_k\|^2\Big).$$

            The autonomous optimum is obtained by averaging this expression
            over the implicit posterior
            $p(t \mid \mathbf{u}) \propto \int p(\mathbf{u}, t)\, dt$,
            which is itself a discrete sum. Both pieces evaluate on a 2D
            grid in milliseconds, and every figure in the notebook that
            does not explicitly involve a trained network is computed by
            $K$-by-$T$ matrix arithmetic of this form.
            """),

        "**2.  The Jensen Gap (Appendix F.2, Eq 66)**":
            mo.md(r"""
            For a noise-prediction parameterization, the autonomous error
            against the true velocity field decomposes into a Natural
            Gradient piece, which the conformal metric cancels, together
            with an additional term:

            $$\Delta v_{\text{noise}}(\mathbf{u}) = \big\|\mathbf{u} - \mathbf{x}^*\big\| \cdot \left|\frac{b'(t)}{b(t)}\right| \cdot \Big| b(t) \cdot \mathbb{E}_{\tau \mid \mathbf{u}}\!\big[1/b(\tau)\big] - 1 \Big|.$$

            The third factor is a Jensen-inequality residual that would be
            zero if the posterior over $\tau$ were a Dirac delta. Since
            $1/b(\tau)$ is convex and the posterior is broad, the residual
            is strictly positive, and the prefactor $|b'(t)/b(t)|$ grows
            without bound as $t \to 0$. The two effects compound near the
            manifold at small $t$, which is the regime in which the gap
            visualized above is largest.
            """),

        "**3.  Why naïve learned-$\\alpha$ collapses (the flat-minimum argument)**":
            mo.md(r"""
            The MSE objective with a free per-point $\alpha$ is given by

            $$\mathcal{L}(\theta) = \mathbb{E}\big\|f_\theta(\mathbf{u}) - [\alpha_\theta\,\boldsymbol{\varepsilon} + (1 - \alpha_\theta)\,\mathbf{v}]\big\|^2.$$

            Its global optimum is reached for any value of $\alpha$ as
            long as $f_\theta(\mathbf{u})$ matches whichever target
            $\alpha$ specifies, so the set of zero-loss configurations
            forms a flat ridge of dimension equal to the dimension of
            $\alpha$. Stochastic dynamics select an arbitrary point on
            the ridge, which in practice means $\alpha \approx 1$ for the
            naive objective and $\alpha \approx 0$ when a side
            regularizer is added. Recovering a usefully spatially varying
            $\alpha$ requires a structural anchor that penalizes constant
            $\alpha$ configurations directly, of which the variational
            ELBO of MuLAN is one example, and which the MSE objective on
            its own does not provide.
            """),
    })
    return

@app.cell
def _closer_md(mo):
    mo.md(r"""
    ## Onwards and Downwards

    A number of related observations have come together in the course of
    the preceding sections, and it may help to summarize the picture they
    form before turning to what remains open.

    The optimal autonomous field on a finite dataset is available in
    closed form, given by Eq 35–36 of the paper, so the analysis above
    required no trained networks except for the empirical sweeps. The
    closed-form field decomposes into three terms: a Natural Gradient
    that points toward the manifold, a Transport Correction that vanishes
    as the ambient dimension grows, and a Linear Drift that absorbs the
    schedule's expanding noise volume. The Natural Gradient is
    preconditioned by an effective gain that vanishes near the manifold
    at the same rate the marginal-energy gradient diverges, and the
    cancellation between the two is what makes the autonomous flow
    well-defined despite an unbounded underlying potential. Sweeping the
    parameterization $\alpha$ continuously, rather than treating velocity,
    noise, and signal prediction as discrete options, reveals that the
    velocity target wins clearly at low ambient dimension while noise
    prediction has no regime in which it meaningfully outperforms
    velocity. The pattern reproduces across moons, Gaussian mixtures, and
    a swiss roll, and is therefore best understood as a property of the
    geometry rather than of any specific dataset. The reason noise
    prediction fails autonomously turns out to be the Jensen Gap of
    Eq 66, which grows precisely near the manifold and at small $t$,
    exactly the region where one would most want the field to be
    accurate. Attempting finally to let the network choose its own
    $\alpha$ per point does not recover the conformal-metric optimum,
    because the MSE objective is degenerate in $\alpha$ and the
    resulting loss landscape contains a flat ridge along which gradient
    descent drifts to whichever endpoint the regularization happens to
    favor.

    Several questions remain open. The whole analysis takes place in
    $\mathbb{R}^D$, but a great deal of real data lives on curved
    spaces, with proteins on rotation groups, phylogenies on tree
    spaces, and lattice configurations on tori as standard examples,
    and it is not obvious whether the conformal cancellation survives
    intrinsic curvature or fractures in some way. Recovering a usefully
    spatially varying $\alpha$ likely requires the variational
    objective of MuLAN rather than a side regularizer added on top of
    MSE, and a clean implementation of that objective in the present
    setting would be informative. The conformal metric is one
    Riemannian preconditioner that succeeds in canceling the manifold
    singularity, but the question of whether it is the only such
    preconditioner, or in any sense the optimal one, has not been
    addressed here.

    On the strength of these results, autonomous diffusion is rather
    more than a heuristic. It is a Riemannian gradient flow whose metric
    is supplied by the data, and many of the engineering choices that
    distinguish one parameterization from another can be understood as
    questions about whether the relevant cancellation continues to hold.
    The remaining cases, and the right way to extend the picture beyond
    Euclidean ambient spaces, are open and worth pursuing.

    The picture in five lines, for the reader who has scrolled to the
    end:

    1. The optimal autonomous field on a finite dataset has a closed form
       (Eq 35–36) and admits a three-term geometric decomposition.
    2. A conformal metric vanishes at the manifold at exactly the rate
       $\nabla E_{\mathrm{marg}}$ diverges, which is what keeps the
       autonomous flow well-defined.
    3. Sweeping $\alpha$ continuously shows that velocity prediction
       wins at low $D$ and noise prediction has no regime in which it
       meaningfully wins.
    4. The Jensen Gap of Eq 66 grows precisely near the manifold at
       small $t$, which is why noise prediction fails autonomously.
    5. Letting the network learn its own $\alpha$ collapses three
       different ways, because the MSE objective is degenerate in
       $\alpha$ and selects no spatial structure.

    The open questions, by the same accounting: curved ambient spaces,
    a clean variational learned-$\alpha$, and whether the conformal
    metric is the only Riemannian preconditioner that does the job.

    ---

    *Code:*
    [github.com/human-vc/gon-notebook](https://github.com/human-vc/gon-notebook).<br>
    *Paper:*
    [arXiv:2602.18428](https://arxiv.org/abs/2602.18428) ·
    *Cite:* Sahraee-Ardakan, Delbracio, Milanfar, *The Geometry of Noise*, Google, Feb 2026.

    *— Jacob Crainic, alphaXiv × marimo, April 2026.*
    """)
    return

if __name__ == "__main__":
    app.run()
