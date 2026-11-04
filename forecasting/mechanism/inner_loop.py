"""One full-batch gradient step on a simplified SwiGLU inner model.

Re-implementation of the ViT^3 inner-loop update (Han et al., CVPR 2026,
arXiv:2512.01643; code github.com/LeapLabTHU/ViTTT, MIT license,
ttt_block.py::inner_train_simplified_swiglu). Checked against the
upstream module by a numerical parity test and against torch autograd
(upstream derives the gradients by hand because the inner loss is
vector-valued per head).

Recipe (ViT^3 defaults):
- inner model: f_W(x) = (x @ w1) * SiLU(x @ w2), per head, d_head ->
  d_head, no norm;
- inner loss: L = -mean_over_tokens(<f_W(k_i), v_i>) * scale;
- update: one plain gradient step, lr = 1.0, with g <- g / (||g||_col + 1);
- the initial weights W0 are learned by the outer loop; the step gives
  per-sample weights W* that are used for the current window only.

Here the tokens are the per-sensor latents of the current test window.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def inner_step(k: torch.Tensor, v: torch.Tensor,
               w1: torch.Tensor, w2: torch.Tensor,
               lr: float = 1.0, scale: float | None = None):
    """One hand-derived inner-loop GD step. Faithful to upstream.

    Args:
        k: keys   [B, H, N, hd]  (B windows, H heads, N tokens/sensors)
        v: values [B, H, N, hd]
        w1, w2: inner weights [1, H, hd, hd] (learned init W0)
        lr: inner learning rate (eta), ViT3 default 1.0
        scale: loss scale; upstream uses equivalent_head_dim**-0.5 (=9**-0.5)
               and notes hd**-0.5 performs similarly. Default: hd**-0.5.
    Returns:
        (w1_star, w2_star): per-sample adapted weights [B, H, hd, hd]
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # forward (only the pieces the closed-form gradient needs)
    z1 = k @ w1
    z2 = k @ w2
    sig = torch.sigmoid(z2)
    a = z2 * sig  # SiLU(z2)

    # backward, hand-derived: e = dL/dv_hat with
    # L = -(v_hat * v).sum(-1).mean(tokens) * scale,  v_hat = z1 * a
    e = -v / float(v.shape[2]) * scale
    g1 = k.transpose(-2, -1) @ (e * a)
    g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

    # gradient normalization (upstream "clip"): g / (||g||_col + 1)
    g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
    g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

    return w1 - lr * g1, w2 - lr * g2


def inner_apply(q: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor):
    """Query pass through the adapted inner model: (q @ w1*) * SiLU(q @ w2*).

    q: [B, H, M, hd]; w1, w2: [B, H, hd, hd] (adapted) or [1, H, hd, hd].
    Returns [B, H, M, hd].
    """
    return (q @ w1) * F.silu(q @ w2)


def inner_loss(k: torch.Tensor, v: torch.Tensor,
               w1: torch.Tensor, w2: torch.Tensor,
               scale: float | None = None) -> torch.Tensor:
    """The explicit inner loss (for tests / diagnostics; the step itself
    uses the closed-form gradient and never materializes this)."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    v_hat = inner_apply(k, w1, w2)
    return -((v_hat * v).sum(-1).mean(-1) * scale)  # [B, H]
