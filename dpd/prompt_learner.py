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


def _text_tower(clip_model):
    """Return the text tower: clip_model.text for CustomTextCLIP (e.g. SigLIP), else clip_model."""
    return clip_model.text if hasattr(clip_model, "text") else clip_model


class CompositionalPrompts(nn.Module):
    def __init__(self, clip_model, tokenizer, crop_names: list[str], disease_names: list[str]):
        super().__init__()
        self.clip = clip_model
        self.text_tower = _text_tower(clip_model)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        if hasattr(tokenizer, "tokenizer"):  # HFTokenizer (e.g. SigLIP)
            hf = tokenizer.tokenizer
            self.bos_id = hf.bos_token_id
            self.eos_id = hf.eos_token_id
            self.pad_id = hf.pad_token_id
            self.pool_type = getattr(self.text_tower, "pool_type", "last")
            t_empty = tokenizer([""])[0]
            t_ab = tokenizer(["a b"])[0]
            self.ctx_offset = 1 if (self.bos_id is not None or (int(t_ab[0]) == int(t_empty[0]) and int(t_empty[0]) not in (self.eos_id, self.pad_id))) else 0
            unk_id = hf.unk_token_id
            end_ab = self._content_end(t_ab)
            candidates = ["platypus", "quasar", "zeppelin", "narwhal", "wombat", "xylophone", "zephyr", "~", "|", "^", "@", "#"]
            if hasattr(hf, "get_vocab"):
                vocab_words = [k[1:] for k in hf.get_vocab() if k.startswith("▁") and 6 <= len(k) <= 10 and k[1:].isascii() and k[1:].isalpha() and k[1:].islower()]
                candidates.extend(vocab_words[:200])
            self.placeholder = None
            self.placeholder_id = None
            for cand in candidates:
                t_cand = tokenizer([f"a {cand} b"])[0]
                end_cand = self._content_end(t_cand)
                if end_cand != end_ab + 1:
                    continue
                diff = (t_cand[:end_ab] != t_ab[:end_ab]).nonzero().flatten()
                ins = int(diff[0]) if len(diff) else end_ab
                pid = int(t_cand[ins])
                if pid != unk_id and torch.equal(t_cand[ins + 1 : end_cand], t_ab[ins:end_ab]):
                    t_colon = tokenizer([f"a {cand}: b"])[0]
                    if int((t_colon == pid).sum()) == 1:
                        self.placeholder = cand
                        self.placeholder_id = pid
                        break
            if self.placeholder is None:
                raise RuntimeError("Could not find a single-token placeholder for this HFTokenizer")
        else:  # OpenAI CLIP SimpleTokenizer
            self.bos_id = getattr(tokenizer, "sot_token_id", None)
            self.eos_id = getattr(tokenizer, "eot_token_id", None)
            self.pad_id = 0
            self.ctx_offset = 1
            self.pool_type = getattr(self.text_tower, "pool_type", "argmax")
            self.placeholder = PLACEHOLDER
            self.placeholder_id = int(tokenizer([self.placeholder])[0, self.ctx_offset])
        self.ctx_placeholder = CTX_PLACEHOLDER
        self.crop_vecs = nn.Parameter(self._init_vectors(crop_names))
        self.disease_vecs = nn.Parameter(self._init_vectors(disease_names))
        self.ctx = nn.ParameterDict()  # learned context per group (CoOp); empty = fixed sentences only
        self.groups = {}

    def _content_end(self, tokens_1d: torch.Tensor) -> int:
        stop_ids = {i for i in (self.eos_id, self.pad_id) if i is not None}
        for pos in range(self.ctx_offset, len(tokens_1d)):
            if int(tokens_1d[pos]) in stop_ids:
                return max(pos, self.ctx_offset + 1)
        return len(tokens_1d)

    @torch.no_grad()
    def add_context(self, name: str, init: str | None, n_ctx: int | None) -> int:
        """Learned context vectors for group `name`, from the token embeddings of `init` or random N(0, 0.02).
        Returns n_ctx."""
        emb = self.text_tower.token_embedding
        if init:
            tokens = self.tokenizer([init])[0]
            end = self._content_end(tokens)
            vecs = emb(tokens[self.ctx_offset : end].to(emb.weight.device)).float()
            assert n_ctx in (None, len(vecs)), f"n_ctx {n_ctx} != {len(vecs)} tokens of init {init!r}"
        else:
            vecs = torch.randn(n_ctx, emb.weight.shape[1], device=emb.weight.device) * 0.02
        self.ctx[name] = nn.Parameter(vecs.clone())
        return len(vecs)

    @torch.no_grad()
    def _init_vectors(self, names: list[str]) -> torch.Tensor:
        """Mean of the token embeddings of each name (names are often several tokens)."""
        tokens = self.tokenizer(names)
        emb = self.text_tower.token_embedding(tokens.to(self.text_tower.token_embedding.weight.device))
        vecs = []
        for t, e in zip(tokens, emb):
            end = self._content_end(t)
            vecs.append(e[self.ctx_offset : end].mean(0))
        return torch.stack(vecs).float()

    @torch.no_grad()
    def add_group(self, name: str, texts: list[str], crop_idx: list[int | None], disease_idx: list[int | None],
                  owner: list[int] | None = None):
        """Register a set of prompt sentences. Each text has a placeholder for the crop and/or disease (in that order)
        wherever crop_idx / disease_idx is not None. Token embeddings are precomputed; only placeholders change.
        owner[i] is the output row of sentence i (several sentences per row are averaged); default one row each."""
        device = self.text_tower.token_embedding.weight.device
        tokens = self.tokenizer(texts).to(device)
        if hasattr(self.tokenizer, "tokenizer"):
            hf = self.tokenizer.tokenizer
            if any(len(hf.encode(t, add_special_tokens=True)) > tokens.shape[1] for t in texts):
                raise ValueError(f"Group '{name}': some prompts exceed the model's {tokens.shape[1]}-token context")
        elif (tokens[:, -1] != 0).any():
            raise ValueError(f"Group '{name}': some prompts exceed CLIP's 77-token context")
        n_ctx = len(self.ctx[name]) if name in self.ctx else 0
        if n_ctx:
            ctx_id = int(self.tokenizer([CTX_PLACEHOLDER])[0, self.ctx_offset])
            c_start, c_end = self.ctx_offset, self.ctx_offset + n_ctx
            if not (tokens[:, c_start:c_end] == ctx_id).all() or (tokens[:, c_end:] == ctx_id).any():
                raise ValueError(f"Group '{name}': every sentence must start with exactly {n_ctx} context slots")
        eot = tokens.argmax(dim=-1)  # EOT has the largest id in OpenAI CLIP ('argmax' pooling)
        if self.pool_type == "argmax" and self.text_tower.attn_mask is not None:
            length = int(eot.max()) + 1  # causal mask: tokens after EOT cannot affect it, so truncate
            tokens = tokens[:, :length]
        crop_pos, disease_pos = [], []
        for idx_row, (t, c, d) in enumerate(zip(tokens, crop_idx, disease_idx)):
            pos = (t == self.placeholder_id).nonzero().flatten().tolist()
            expected = (c is not None) + (d is not None)
            if len(pos) != expected:
                raise ValueError(
                    f"Group '{name}': expected {expected} placeholders "
                    f"(str={self.placeholder!r}, id={self.placeholder_id}), found {len(pos)} in {texts[idx_row]!r}"
                )
            crop_pos.append(pos[0] if c is not None else -1)
            disease_pos.append(pos[-1] if d is not None else -1)
        as_t = lambda xs: torch.tensor([-1 if x is None else x for x in xs], device=device)
        self.groups[name] = {
            "base": self.text_tower.token_embedding(tokens).float(),
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
            x[:, self.ctx_offset : self.ctx_offset + g["n_ctx"]] = self.ctx[name].to(x.dtype)
        length = x.shape[1]
        x = x + self.text_tower.positional_embedding[:length]
        mask = self.text_tower.attn_mask[:length, :length] if self.text_tower.attn_mask is not None else None
        x = self.text_tower.transformer(x, attn_mask=mask)
        x = self.text_tower.ln_final(x)
        if self.pool_type == "last":
            pooled = x[:, -1]
        elif self.pool_type == "mean":
            pooled = x.mean(dim=1)
        else:
            pooled = x[rows, g["eot"]]
        proj = self.text_tower.text_projection
        if proj is not None:
            pooled = proj(pooled) if isinstance(proj, nn.Module) else (pooled @ proj)
        x = F.normalize(pooled.float(), dim=-1)
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
    crop_text = lambda c: model.placeholder if learn_prim else display_name(c, names["crop_overrides"], u2s)
    disease_text = lambda d: model.placeholder if learn_prim else display_name(d, names["disease_overrides"], u2s)
    lc = prompts.get("learned_context")
    prefix = {}
    for group in ("pairs", "crops", "diseases"):
        n = model.add_context(group, lc.get("init"), lc.get("n_ctx")) if lc else 0
        prefix[group] = " ".join([model.ctx_placeholder] * n) + " " if n else ""

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
