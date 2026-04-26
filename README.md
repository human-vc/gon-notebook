# Why Autonomous Diffusion Really Works

A marimo notebook walking through *The Geometry of Noise* (Sahraee-Ardakan, Delbracio, Milanfar — Google, Feb 2026; [arXiv:2602.18428](https://arxiv.org/abs/2602.18428)) for the alphaXiv × marimo notebook competition.

The notebook renders the paper's central claim — that autonomous diffusion fields implicitly perform Riemannian gradient flow with a free conformal preconditioner — visible in 2D, makes the (α, D) parameterization landscape continuous and interactive, and characterizes where the story breaks.

## Run it

In molab (browser, no install):

```
https://molab.marimo.io/github/<user>/gon-notebook/blob/main/walkthrough.py/wasm
```

Locally:

```bash
uv venv
uv pip install marimo numpy torch matplotlib scikit-learn
uvx marimo edit walkthrough.py
```

## Files

- `walkthrough.py` — the notebook
- `gon_toolkit.py` — closed-form fields, samplers, conformal metric, Jensen Gap, training loops
- `exp*.py` — the experiments that produced the cached `.npz` data the notebook loads
- `exp05_phase_grid.npz`, `exp12_failure_grids.npz`, `exp13_iconic.npz`, `exp14_alpha_fields.npz` — cached results so the notebook loads instantly

## Findings

1. The optimal autonomous field on a finite dataset has a closed form (Eq 35–36); no training needed.
2. It decomposes into Natural Gradient + Transport Correction + Linear Drift (Eq 14), and the Transport Correction vanishes as ambient dimension grows.
3. The marginal-energy gradient diverges at the manifold; the conformal metric vanishes at exactly the right rate; their product is bounded.
4. On the continuous (α, D) phase plane, velocity wins decisively at low D, noise prediction has no regime where it meaningfully beats velocity, and the failure smoothly fades as D grows.
5. The Jensen Gap explodes near the manifold at low t — the source of the noise-prediction failure.
6. Asking the network to learn its own α collapses three different ways. Plain MSE is the wrong loss; structural priors are required.

## Credit

Paper: Sahraee-Ardakan, Delbracio, Milanfar (Google, 2026). Notebook design borrows heavily from Gabriel Goh's *Why Momentum Really Works* (Distill, 2017).
