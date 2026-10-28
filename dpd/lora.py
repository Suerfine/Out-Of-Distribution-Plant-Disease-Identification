"""LoRA (low-rank adaptation) for the CLIP image encoder.

A frozen weight W is used as W + (alpha / r) * B @ A, with A (r x in) and B (out x r) trainable and B starting at
zero, so training starts from exactly the pretrained model. It is attached with torch.nn.utils.parametrize, which
substitutes the adapted weight wherever the module reads it; that matters because nn.MultiheadAttention reads
in_proj_weight / out_proj.weight directly instead of calling a Linear layer.
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

# Weight names inside an open_clip ResidualAttentionBlock (attention q/k/v + output, MLP in + out).
TARGETS = {
    "attn_qkv": ("attn", "in_proj_weight"),
    "attn_out": ("attn.out_proj", "weight"),
    "mlp_in": ("mlp.c_fc", "weight"),
    "mlp_out": ("mlp.c_proj", "weight"),
}


class LoRADelta(nn.Module):
    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, W):
        return W + (self.B @ self.A).to(W.dtype) * self.scale


def apply_lora(visual, last_n_blocks: int, rank: int, alpha: float, targets: list[str]) -> list[nn.Parameter]:
    """Freeze the whole image encoder, then add LoRA to `targets` in its last `last_n_blocks` blocks.
    Returns the new trainable parameters. Earlier blocks get no gradient, so autograd stores nothing for them."""
    for p in visual.parameters():
        p.requires_grad_(False)
    params = []
    for block in visual.transformer.resblocks[-last_n_blocks:]:
        for t in targets:
            module_path, name = TARGETS[t]
            module = block.get_submodule(module_path)
            out_f, in_f = getattr(module, name).shape
            delta = LoRADelta(out_f, in_f, rank, alpha).to(getattr(module, name).device)
            parametrize.register_parametrization(module, name, delta)
            params += [delta.A, delta.B]
    return params


def lora_state(visual) -> dict:
    """Only the LoRA tensors (A, B per adapted weight), keyed by their module path."""
    return {k: v.detach().cpu() for k, v in visual.state_dict().items() if ".parametrizations." in k and k.endswith((".A", ".B"))}


def load_lora(visual, state: dict, last_n_blocks: int, rank: int, alpha: float, targets: list[str]):
    """Rebuild the adapters on a fresh encoder and load saved LoRA tensors (for evaluation / the app)."""
    apply_lora(visual, last_n_blocks, rank, alpha, targets)
    missing = set(state) - set(visual.state_dict())
    if missing:
        raise KeyError(f"LoRA tensors not found in the model: {sorted(missing)[:3]}")
    visual.load_state_dict(state, strict=False)
