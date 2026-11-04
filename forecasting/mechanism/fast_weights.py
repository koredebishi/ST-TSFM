"""Fast-weight adapter: the adapter used in the paper's TTRA experiment.

The adapted state is the weights of a small SwiGLU inner model
(mechanism/inner_loop.py), fit per window on the sensors' (k, v)
projections with one gradient step, then discarded.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mechanism.inner_loop import inner_apply, inner_step


class FastWeightAdapter(nn.Module):
    """Fast-weight cross-sensor residual (the ViT^3 inner model).

    forward(latents [B, N, d]) -> residual [B, N, d], zero at init.
    """

    def __init__(self, d_model: int = 768, n_heads: int = 4,
                 head_dim: int = 64, eta: float = 1.0,
                 inner_steps: int = 1):
        super().__init__()
        self.h, self.hd, self.eta = n_heads, head_dim, eta
        self.inner_steps = inner_steps
        inner = n_heads * head_dim
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)
        self.w1_0 = nn.Parameter(torch.randn(1, n_heads, head_dim, head_dim) * 0.02)
        self.w2_0 = nn.Parameter(torch.randn(1, n_heads, head_dim, head_dim) * 0.02)
        self.gate = nn.Parameter(torch.zeros(1))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.hd).transpose(1, 2)

    def adapted_weights(self, latents: torch.Tensor):
        k = self._heads(self.k(latents))
        v = self._heads(self.v(latents))
        w1, w2 = self.w1_0, self.w2_0
        for _ in range(self.inner_steps):
            w1, w2 = inner_step(k, v, w1, w2, lr=self.eta)
        return w1, w2

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        w1, w2 = self.adapted_weights(latents)
        q = self._heads(self.q(latents))
        o = inner_apply(q, w1, w2)
        B, _, N, _ = o.shape
        o = o.transpose(1, 2).reshape(B, N, self.h * self.hd)
        return self.gate * self.out(o)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
