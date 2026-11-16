"""Compositional prompt learning on a frozen CLIP (in the style of CSP, Nayak et al. 2023).

Each crop and each disease gets ONE learnable word vector, initialised from CLIP's own embedding of its name.
A pair's prompt is a fixed sentence with two placeholder words swapped for those vectors, e.g.
    "a photo of [CROP] leaves with [DISEASE]: <fixed symptom description>."
so an unseen pair's prompt is assembled from vectors that were trained on other pairs. Both CLIP encoders stay
frozen; gradients flow through the frozen text encoder into the 55 + 175 vectors only.

Learned context (CoOp, Zhou et al. 2022; prompts.learned_context): n_ctx learnable vectors are put at the start of
every sentence (right after the start token), one set per prompt group (pairs / crops / diseases), shared by all
classes in that group. Initialised from the token embeddings of an init phrase (e.g. "a photo of", n_ctx = its
token count) or randomly (N(0, 0.02), with n_ctx given). With prompts.learn_primitives false, the crop and disease
names are written into the sentences as ordinary text and stay fixed, i.e. plain CoOp; with it true (default),
context and primitive vectors are both learned (CoOp + CSP).

A description may also be a list (e.g. [taxonomy, visual symptoms], as in zero-shot v3.6): the class then gets one
sentence per entry, and its embedding is the renormalised mean of those sentences' embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpd.clip_utils import display_name, load_descriptions

PLACEHOLDER = "x"  # single BPE token; the tokenizer lowercases, and no prompt text contains a standalone "x"
CTX_PLACEHOLDER = "*"  # single BPE token for learned-context slots; no prompt text contains "*"


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
        self.ctx = nn.ParameterDict()  # learned context per group (CoOp); empty = fixed sentences only
        self.groups = {}

    @torch.no_grad()
    def add_context(self, name: str, init: str | None, n_ctx: int | None) -> int:
        """Learned context vectors for group `name`, from the token embeddings of `init` or random N(0, 0.02).
        Returns n_ctx."""
        emb = self.clip.token_embedding
        if init:
            tokens = self.tokenizer([init])[0]
            n = int((tokens != 0).sum())
            vecs = emb(tokens[1 : n - 1].to(emb.weight.device)).float()
            assert n_ctx in (None, len(vecs)), f"n_ctx {n_ctx} != {len(vecs)} tokens of init {init!r}"
        else:
            vecs = torch.randn(n_ctx, emb.weight.shape[1], device=emb.weight.device) * 0.02
        self.ctx[name] = nn.Parameter(vecs.clone())
        return len(vecs)

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
    def add_group(self, name: str, texts: list[str], crop_idx: list[int | None], disease_idx: list[int | None],
                  owner: list[int] | None = None):
        """Register a set of prompt sentences. Each text has a placeholder for the crop and/or disease (in that order)
        wherever crop_idx / disease_idx is not None. Token embeddings are precomputed; only placeholders change.
        owner[i] is the output row of sentence i (several sentences per row are averaged); default one row each."""
        device = self.clip.token_embedding.weight.device
        tokens = self.tokenizer(texts).to(device)
        if (tokens[:, -1] != 0).any():
            raise ValueError(f"Group '{name}': some prompts exceed CLIP's 77-token context")
        n_ctx = len(self.ctx[name]) if name in self.ctx else 0
        if n_ctx:
            ctx_id = int(self.tokenizer([CTX_PLACEHOLDER])[0, 1])
            if not (tokens[:, 1 : 1 + n_ctx] == ctx_id).all() or (tokens[:, 1 + n_ctx :] == ctx_id).any():
                raise ValueError(f"Group '{name}': every sentence must start with exactly {n_ctx} context slots")
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
            "owner": None if owner is None else torch.tensor(owner, device=device),
            "n_ctx": n_ctx,
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
        if g["n_ctx"]:
            x[:, 1 : 1 + g["n_ctx"]] = self.ctx[name].to(x.dtype)
        length = x.shape[1]
        x = x + self.clip.positional_embedding[:length]
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask[:length, :length])
        x = self.clip.ln_final(x)
        x = F.normalize((x[rows, g["eot"]] @ self.clip.text_projection).float(), dim=-1)
        if g["owner"] is not None:  # renormalised mean of each row's sentences
            n_rows = int(g["owner"].max()) + 1
            x = F.normalize(x.new_zeros(n_rows, x.shape[1]).index_add(0, g["owner"], x), dim=-1)
        return x


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
    learn_prim = prompts.get("learn_primitives", True)
    if not learn_prim:  # plain CoOp: names are fixed text, the primitive vectors are unused
        model.crop_vecs.requires_grad_(False)
        model.disease_vecs.requires_grad_(False)
    crop_text = lambda c: PLACEHOLDER if learn_prim else display_name(c, names["crop_overrides"], u2s)
    disease_text = lambda d: PLACEHOLDER if learn_prim else display_name(d, names["disease_overrides"], u2s)
    lc = prompts.get("learned_context")
    prefix = {}
    for group in ("pairs", "crops", "diseases"):
        n = model.add_context(group, lc.get("init"), lc.get("n_ctx")) if lc else 0
        prefix[group] = " ".join([CTX_PLACEHOLDER] * n) + " " if n else ""

    def sentences(group, template, crop_names, disease_names):
        """Every filled sentence, with its output row (one row per class; several if its description is a list)."""
        texts, owner = [], []
        for row, (c, d) in enumerate(zip(crop_names, disease_names)):
            ds = desc.get(d, "") if d is not None else ""
            for one in ds if isinstance(ds, list) else [ds]:
                texts.append(prefix[group] + template.format(
                    crop=crop_text(c) if c is not None else "", disease=disease_text(d) if d is not None else "",
                    description=one))
                owner.append(row)
        return texts, owner

    expand = lambda values, owner: [values[i] if learn_prim else None for i in owner]
    texts, owner = sentences("pairs", prompts["pair_template"], pairs["crop"].tolist(), pairs["disease"].tolist())
    model.add_group(
        "pairs", texts, expand(pairs["crop_id"].tolist(), owner), expand(pairs["disease_id"].tolist(), owner),
        owner if len(owner) > len(pairs) else None,
    )
    texts, owner = sentences("crops", prompts["crop_template"], crops["crop"].tolist(), [None] * len(crops))
    model.add_group("crops", texts, expand(crops["crop_id"].tolist(), owner), [None] * len(texts))
    texts, owner = sentences("diseases", prompts["disease_template"], [None] * len(diseases), diseases["disease"].tolist())
    model.add_group(
        "diseases", texts, [None] * len(texts), expand(diseases["disease_id"].tolist(), owner),
        owner if len(owner) > len(diseases) else None,
    )
    if not learn_prim:  # fixed names: two classes with the same text would be indistinguishable
        for group in ("pairs", "crops", "diseases"):
            g = model.groups[group]
            rows = g["base"].flatten(1)
            if g["owner"] is None and len(torch.unique(rows, dim=0)) < len(rows):
                raise ValueError(f"Group '{group}': duplicate prompt sentences with fixed names")
    return model
