"""Register-token adapter: an alternative TTRA design, not used in the paper.

The adapted state is K register tokens R [K x d] with a learned initial
value R0. For each test window they are re-fit by one gradient step on a
dot-product reconstruction loss, then discarded.

Per test window, with sensor latents x [B, N, d]:
    q_j = W_q x_j                            sensors are queries
    k_r, v_r = W_k R, W_v R                  registers are the memory
    o_j = softmax(q_j k^T / sqrt(hd)) v
    L = -mean_j <o_j, x_j> * scale           reconstruct the sensors
    R* = R0 - eta * dL/dR                    one step
    residual_j = gate * W_o o_j(R*)          gate starts at 0

R* is fit on all sensors, so each sensor's residual depends on the
others. With d = 768, K = 8 and inner dim 256 the adapter has about 0.8M
parameters, about 0.7% of the backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTRARegisters(nn.Module):
    """K test-time-adapted register tokens for a frozen backbone.

    forward(latents [B, N, d]) -> residual [B, N, d], zero at init.
    """

    def __init__(self, d_model: int = 768, n_registers: int = 8,
                 n_heads: int = 4, head_dim: int = 64,
                 eta: float = 1.0, inner_steps: int = 1):
        super().__init__()
        self.K, self.h, self.hd = n_registers, n_heads, head_dim
        self.eta, self.inner_steps = eta, inner_steps
        inner = n_heads * head_dim
        self.registers = nn.Parameter(torch.randn(1, n_registers, d_model) * 0.02)
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.hd).transpose(1, 2)  # [B,H,N,hd]

    def _read(self, latents: torch.Tensor, registers: torch.Tensor):
        """Sensors read the register memory. -> [B, N, h*hd]"""
        B = latents.shape[0]
        q = self._heads(self.q(latents))                       # [B,H,N,hd]
        k = self._heads(self.k(registers.expand(B, -1, -1)))   # [B,H,K,hd]
        v = self._heads(self.v(registers.expand(B, -1, -1)))
        att = torch.softmax(q @ k.transpose(-2, -1) * self.hd ** -0.5, dim=-1)
        o = att @ v                                            # [B,H,N,hd]
        return o.transpose(1, 2).reshape(B, latents.shape[1], self.h * self.hd)

    def inner_loss(self, latents: torch.Tensor,
                   registers: torch.Tensor) -> torch.Tensor:
        """Negative dot-product reconstruction of the bag (ViT3 family).

        Reconstruction target is the latent projected to the inner width
        via W_q (so the dot product lives in one space); scale = hd**-0.5.
        """
        o = self._read(latents, registers)                     # [B,N,inner]
        tgt = self.q(latents)                                  # [B,N,inner]
        scale = self.hd ** -0.5
        return -((o * tgt).sum(-1).mean(-1) * scale).sum()

    def adapt(self, latents: torch.Tensor) -> torch.Tensor:
        """One (or inner_steps) GD step(s) on the registers, per window.

        Returns per-window adapted registers [B, K, d]. Differentiable
        w.r.t. R0 and projections (meta-learning path, ViT3 Insight 1).
        """
        B = latents.shape[0]
        R = self.registers.expand(B, -1, -1)
        for _ in range(self.inner_steps):
            R = R.requires_grad_() if not R.requires_grad else R
            loss = self.inner_loss(latents, R)
            (g,) = torch.autograd.grad(loss, R, create_graph=self.training)
            g = g / (g.norm(dim=-1, keepdim=True) + 1.0)  # ViT3 grad-norm
            R = R - self.eta * g
        return R

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        R_star = self.adapt(latents) if self.inner_steps > 0 \
            else self.registers.expand(latents.shape[0], -1, -1)
        o = self._read(latents, R_star)
        return self.gate * self.out(o)  # zero at init: attach is a no-op

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
