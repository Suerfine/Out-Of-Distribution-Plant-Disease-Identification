"""Compositional prompt learning on a frozen CLIP (in the style of CSP, Nayak et al. 2023).

Each crop and each disease gets ONE learnable word vector, initialised from CLIP's own embedding of its name.
A pair's prompt is a fixed sentence with two placeholder words swapped for those vectors, e.g.
    "a photo of [CROP] leaves with [DISEASE]: <fixed symptom description>."
so an unseen pair's prompt is assembled from vectors that were trained on other pairs. Both CLIP encoders stay
frozen; gradients flow through the frozen text encoder into the 55 + 175 vectors only.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpd.clip_utils import display_name, load_descriptions

PLACEHOLDER = "x"  # single BPE token; the tokenizer lowercases, and no prompt text contains a standalone "x"


class CompositionalPrompts(nn.Module):
    def __init__(self, clip_model, tokenizer, crop_names: list[str], disease_names: list[str]):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.placeholder_id = int(tokenizer([PLACEHOLDER])[0, 1])
        self.crop_vecs = nn.Parameter(self._init_vectors(crop_names))
        self.disease_vecs = nn.Parameter(self._init_vectors(disease_names))
        self.groups = {}

    @torch.no_grad()
    def _init_vectors(self, names: list[str]) -> torch.Tensor:
        """Mean of the token embeddings of each name (names are often several BPE tokens)."""
        tokens = self.tokenizer(names)
        emb = self.clip.token_embedding(tokens.to(self.clip.token_embedding.weight.device))
        vecs = []
        for t, e in zip(tokens, emb):
            n = int((t != 0).sum())  # [SOT] name... [EOT]
            vecs.append(e[1 : n - 1].mean(0))
        return torch.stack(vecs).float()

    @torch.no_grad()
    def add_group(self, name: str, texts: list[str], crop_idx: list[int | None], disease_idx: list[int | None]):
        """Register a set of prompt sentences. Each text has a placeholder for the crop and/or disease (in that order)
        wherever crop_idx / disease_idx is not None. Token embeddings are precomputed; only placeholders change."""
        device = self.clip.token_embedding.weight.device
        tokens = self.tokenizer(texts).to(device)
        if (tokens[:, -1] != 0).any():
            raise ValueError(f"Group '{name}': some prompts exceed CLIP's 77-token context")
        eot = tokens.argmax(dim=-1)  # EOT has the largest id (open_clip 'argmax' pooling)
        length = int(eot.max()) + 1  # causal mask: tokens after EOT cannot affect it, so truncate
        tokens = tokens[:, :length]
        crop_pos, disease_pos = [], []
        for t, c, d in zip(tokens, crop_idx, disease_idx):
            pos = (t == self.placeholder_id).nonzero().flatten().tolist()
            expected = (c is not None) + (d is not None)
            if len(pos) != expected:
                raise ValueError(f"Group '{name}': expected {expected} placeholders, found {len(pos)}")
            crop_pos.append(pos[0] if c is not None else -1)
            disease_pos.append(pos[-1] if d is not None else -1)
        as_t = lambda xs: torch.tensor([-1 if x is None else x for x in xs], device=device)
        self.groups[name] = {
            "base": self.clip.token_embedding(tokens).float(),
            "eot": eot,
            "crop_idx": as_t(crop_idx), "crop_pos": torch.tensor(crop_pos, device=device),
            "disease_idx": as_t(disease_idx), "disease_pos": torch.tensor(disease_pos, device=device),
        }

    def encode(self, name: str) -> torch.Tensor:
        """L2-normalised text embeddings for every sentence in a group (differentiable w.r.t. the vectors)."""
        g = self.groups[name]
        x = g["base"].clone()
        rows = torch.arange(len(x), device=x.device)
        m = g["crop_pos"] >= 0
        x[rows[m], g["crop_pos"][m]] = self.crop_vecs[g["crop_idx"][m]]
        m = g["disease_pos"] >= 0
        x[rows[m], g["disease_pos"][m]] = self.disease_vecs[g["disease_idx"][m]]
        length = x.shape[1]
        x = x + self.clip.positional_embedding[:length]
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask[:length, :length])
        x = self.clip.ln_final(x)
        x = x[rows, g["eot"]] @ self.clip.text_projection
        return F.normalize(x.float(), dim=-1)


def build_model(cfg, clip_model, tokenizer, pairs):
    """CompositionalPrompts with the 'pairs', 'crops' and 'diseases' prompt groups described by the config."""
    names, prompts = cfg["names"], cfg["prompts"]
    u2s = names["underscore_to_space"]
    crops = pairs.drop_duplicates("crop_id").sort_values("crop_id")
    diseases = pairs.drop_duplicates("disease_id").sort_values("disease_id")
    assert (crops["crop_id"].to_numpy() == np.arange(len(crops))).all()
    assert (diseases["disease_id"].to_numpy() == np.arange(len(diseases))).all()
    model = CompositionalPrompts(
        clip_model,
        tokenizer,
        [display_name(c, names["crop_overrides"], u2s) for c in crops["crop"]],
        [display_name(d, names["disease_overrides"], u2s) for d in diseases["disease"]],
    )
    desc = load_descriptions(prompts)
    fill = lambda t, disease: t.format(crop=PLACEHOLDER, disease=PLACEHOLDER, description=desc.get(disease, ""))
    model.add_group(
        "pairs",
        [fill(prompts["pair_template"], d) for d in pairs["disease"]],
        pairs["crop_id"].tolist(),
        pairs["disease_id"].tolist(),
    )
    model.add_group("crops", [prompts["crop_template"].format(crop=PLACEHOLDER)] * len(crops), crops["crop_id"].tolist(), [None] * len(crops))
    model.add_group(
        "diseases",
        [fill(prompts["disease_template"], d) for d in diseases["disease"]],
        [None] * len(diseases),
        diseases["disease_id"].tolist(),
    )
    return model
