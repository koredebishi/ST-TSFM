"""Wrapper around frozen Chronos-2. The only place the Chronos-2 API is called.

From the installed chronos package (chronos2/model.py, pipeline.py):
- encode() gives hidden_states (batch, n_ctx+1+n_out, d_model); the
  forecast embeddings are hidden_states[:, -n_out:], which go to
  output_patch_embedding (the quantile head);
- d_model = 768, output_patch_size = 16 (config.json);
- predict() runs under torch.no_grad(), so it is for evaluation only.

The adapter is added with a forward_pre_hook on
model.output_patch_embedding: its residual is added to the forecast
embeddings before the head. With the adapter gate at 0 the residual is 0
and the output is identical to plain Chronos-2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

D_MODEL = 768          # config.json, asserted at load
MODEL_ID = "amazon/chronos-2"


class ChronosBackbone:
    """Frozen Chronos-2 with an optional residual injected on the head input.

    adapter: callable mapping forecast_embeds [B_win, N_sensors, d] ->
    residual of the same shape, or None for stock behaviour. One predict
    call = one window; its N sensors are the Chronos batch (attach_spec §3).
    """

    def __init__(self, adapter: nn.Module | None = None, device: str = "cuda"):
        from chronos import BaseChronosPipeline

        try:
            self.pipe = BaseChronosPipeline.from_pretrained(
                MODEL_ID, device_map=device, dtype=torch.bfloat16)
        except TypeError:
            self.pipe = BaseChronosPipeline.from_pretrained(
                MODEL_ID, device_map=device)
        self.model = self.pipe.model  # Chronos2Model

        # freeze the entire backbone (assert no trainable backbone params)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # sourced-fact assert: head exists and d_model matches
        assert hasattr(self.model, "output_patch_embedding"), \
            "sourced API changed: no output_patch_embedding head"
        assert self.model.chronos_config.output_patch_size == 16, \
            "sourced fact changed: output_patch_size != 16"

        self.adapter = adapter
        self._hook_handle = None
        self._last_forecast_embeds_shape = None
        self._last_residual_absmax = None  # for the deterministic no-op proof
        self.train_mode = False  # eval detaches the residual; train keeps grad

    # ---- the injection hook (eval path) --------------------------------

    def _pre_hook(self, module, args):
        """forward_pre_hook on output_patch_embedding.

        args[0] = forecast_embeds, shape (batch=N_sensors, n_out, d_model).
        Returns modified args with the adapter residual added. gate=0 in the
        adapter => residual 0 => identical output (no-op proof).
        """
        fe = args[0]
        self._last_forecast_embeds_shape = tuple(fe.shape)
        # sourced-fact assert: head input is (batch, n_out, d_model)
        assert fe.shape[-1] == D_MODEL, \
            f"sourced fact changed: head input last dim {fe.shape[-1]} != {D_MODEL}"
        if self.adapter is None:
            return None
        # one window: batch axis = sensors -> [1 window, N sensors, d].
        # adapter runs in fp32 (its params are fp32; backbone is bf16).
        # enable_grad makes the hook adapter-agnostic: the fast-weight adapter
        # (analytical inner step) needs no autograd at eval,
        # but the token ablation's autograd inner step does. predict() is
        # no_grad (pipeline.py:735); we re-enable locally and detach the
        # result for the eval path (no outer/backbone grad). The grad-enabled
        # training forward is forward_train().
        with torch.enable_grad():
            bag = fe.transpose(0, 1).float()  # (n_out, N, d); n_out=1 here
            residual = self.adapter(bag)      # (n_out, N, d)
        if not self.train_mode:
            residual = residual.detach()      # eval: no outer/backbone grad
        # record exact injected magnitude: gate=0 => exactly 0 => true no-op
        self._last_residual_absmax = float(residual.detach().abs().max())
        return (fe + residual.transpose(0, 1).to(fe.dtype),) + tuple(args[1:])

    def _ensure_hook(self):
        if self.adapter is not None and self._hook_handle is None:
            self._hook_handle = self.model.output_patch_embedding.\
                register_forward_pre_hook(self._pre_hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ---- eval forward (stock predict path, no_grad) --------------------

    def predict_quantiles(self, context: torch.Tensor, prediction_length: int,
                          quantile_levels=(0.5,)):
        """context: (N_sensors, ctx_len). Returns stock pipeline output with
        the adapter residual injected (or stock if adapter is None)."""
        self._ensure_hook()
        return self.pipe.predict_quantiles(
            context, prediction_length=prediction_length,
            quantile_levels=list(quantile_levels))

    # ---- train forward (grad-enabled; reuses model.forward, NOT predict) --

    def forward_train(self, context: torch.Tensor, num_output_patches: int = 1):
        """Forward pass with gradients, for training the adapter.

        The "side door": call `model.forward` DIRECTLY (sourced model.py:618)
        instead of `predict` (pipeline.py:735 wraps it in no_grad). This
        reuses the model's OWN tail (encode -> head -> rearrange -> unscale,
        model.py:710-756) verbatim — nothing reimplemented. The hook injects
        the residual WITH grad (train_mode), the backbone stays frozen
        (requires_grad=False), so gradients reach only the adapter.

        context: (N_sensors, context_length), RAW values (the model
        instance-normalizes internally). Returns quantile_preds
        (N, num_quantiles, num_output_patches*output_patch_size), grad-enabled.
        """
        self._ensure_hook()
        prev = self.train_mode
        self.train_mode = True
        try:
            out = self.model(context=context, num_output_patches=num_output_patches)
        finally:
            self.train_mode = prev
        return out.quantile_preds

    def n_backbone_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def median_index(self) -> int:
        return list(self.model.chronos_config.quantiles).index(0.5)
