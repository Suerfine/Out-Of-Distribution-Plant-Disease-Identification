"""Compositional contrastive model: an image is matched to the embedding of a (crop, disease) pair.

    image -> backbone -> f -> Linear -> z                      (unit length)
    pair (c, d):  p(c, d) = normalise( e_c + e_d + MLP([e_c ; e_d]) )     e_c, e_d: learned crop / disease embeddings
    score(x, c, d) = s * z . p(c, d)                            s: learned temperature scale

The pair embedding starts as the sum of its two parts, so it begins as independent crop and disease evidence, and the MLP
learns how a disease looks on a particular crop. An unseen pair is scored with the same function, built from the embeddings
its crop and its disease learned in other pairs. The loss is an InfoNCE / softmax over the pair embeddings (CLIP-style,
with learned label embeddings instead of a text encoder); see train/train_compositional.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpd.cnn import HEAD_SIZES, build_backbone


class CompositionalNet(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, pair_crop: np.ndarray, pair_disease: np.ndarray, emb_dim: int = 256,
                 hidden: int = 512, head_dropout: float = 0.2, drop_path_rate: float = 0.0, init_scale: float = 14.3):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained, drop_path_rate)
        self.img = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(dim, emb_dim))
        self.crop_emb = nn.Embedding(HEAD_SIZES["crop"], emb_dim)
        self.dis_emb = nn.Embedding(HEAD_SIZES["disease"], emb_dim)
        nn.init.normal_(self.crop_emb.weight, std=0.1)
        nn.init.normal_(self.dis_emb.weight, std=0.1)
        self.compose = nn.Sequential(nn.Linear(2 * emb_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, emb_dim))
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(init_scale))))
        self.register_buffer("pair_crop", torch.as_tensor(pair_crop, dtype=torch.long))
        self.register_buffer("pair_disease", torch.as_tensor(pair_disease, dtype=torch.long))

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.") and n != "log_scale"]

    def pair_embeddings(self) -> torch.Tensor:
        c, d = self.crop_emb(self.pair_crop), self.dis_emb(self.pair_disease)
        return F.normalize(c + d + self.compose(torch.cat([c, d], dim=-1)), dim=-1)

    def forward(self, x) -> torch.Tensor:
        """(images x pairs) scores, in the order of dpd.data.pair_table()."""
        z = F.normalize(self.img(self.backbone(x)), dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * z @ self.pair_embeddings().T


@torch.no_grad()
def pair_scores(model: CompositionalNet, loader, device: str, amp: bool) -> np.ndarray:
    """Scores of every image in an ImageLabels loader against all pairs (images x 333)."""
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()
