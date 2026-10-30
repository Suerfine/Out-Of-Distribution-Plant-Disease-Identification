"""Patch-level disease scoring for the CLIP image encoder (v7+).

Lesions are small, so the CLS embedding (dominated by leaf shape, i.e. crop) barely moves when a few spots appear.
Here each disease's text embedding is also matched against every patch token (projected into CLIP's joint space the
same way as CLS), and the mean of its top-k patch similarities is that disease's patch score. A pair's score is
cosine(CLS, pair text) + lambda * patch score of the pair's disease, so the 333-pair evaluation and gamma
calibration are unchanged. Enabled by `model.patch_disease = {"top_k": k, "lambda": l}`; absent means CLS only.
"""

import torch
import torch.nn.functional as F


def image_features(clip_model, x: torch.Tensor, disease_text: torch.Tensor | None, top_k: int | None):
    """(L2-normalised CLS embedding, per-disease patch scores or None). `disease_text`: normalised (n_diseases, d)."""
    if disease_text is None:
        return F.normalize(clip_model.encode_image(x).float(), dim=-1), None
    visual = clip_model.visual
    # For OpenAI CLIP ViTs ln_post is applied to all tokens before pooling, so patch tokens only need the projection.
    assert visual.attn_pool is None and not visual.final_ln_after_pool, "patch scoring assumes OpenAI-style ViT pooling"
    visual.output_tokens = True
    try:
        pooled, tokens = visual(x)
    finally:
        visual.output_tokens = False
    cls = F.normalize(pooled.float(), dim=-1)
    patches = F.normalize((tokens @ visual.proj).float(), dim=-1)  # (B, P, d)
    sims = torch.einsum("bpd,kd->bkp", patches, disease_text.float())  # (B, n_diseases, P)
    return cls, sims.topk(top_k, dim=-1).values.mean(-1)


def combine(cls_scores, patch_scores, pair_disease, lam: float):
    """Pair scores: CLS-vs-pair-text cosine plus lambda times the patch score of each pair's disease."""
    if patch_scores is None:
        return cls_scores
    return cls_scores + lam * patch_scores[:, pair_disease]
