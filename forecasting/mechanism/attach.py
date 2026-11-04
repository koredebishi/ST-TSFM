"""Attach the test-time adapter (TTRA) to frozen Chronos-2.

Builds mechanism/chronos_backbone.py with an adapter. The adapter used in
the paper is the fast-weight inner model (mechanism/fast_weights.py,
variant="fast_weight"). The register-token version (mechanism/registers.py,
variant="token") is an alternative design and is not reported.

This is the evaluation path (stock predict, no gradients). Training uses
ChronosBackbone.forward_train.
"""

from __future__ import annotations

from mechanism.chronos_backbone import D_MODEL, ChronosBackbone
from mechanism.fast_weights import FastWeightAdapter
from mechanism.registers import TTRARegisters


def build_ttra(variant: str = "fast_weight", n_heads: int = 4,
               head_dim: int = 64, eta: float = 1.0, inner_steps: int = 1,
               n_registers: int = 8, device: str = "cuda"):
    """Returns (backbone, adapter): frozen Chronos-2 with the TTRA adapter
    injected on forecast_embeds (gate=0 at init => attach is a no-op).

    variant="fast_weight" (used in the paper) | "token" (alternative design).
    """
    if variant == "fast_weight":
        adapter = FastWeightAdapter(
            d_model=D_MODEL, n_heads=n_heads, head_dim=head_dim,
            eta=eta, inner_steps=inner_steps).to(device)
    elif variant == "token":
        adapter = TTRARegisters(
            d_model=D_MODEL, n_registers=n_registers, n_heads=n_heads,
            head_dim=head_dim, eta=eta, inner_steps=inner_steps).to(device)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    backbone = ChronosBackbone(adapter=adapter, device=device)
    return backbone, adapter
