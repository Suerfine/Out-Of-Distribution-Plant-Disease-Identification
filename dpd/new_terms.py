"""Opt-in training terms for the unseen-pair search (all sit on top of the held-out pairs loss in train/train_supcon.py).

kind "compositor": a pair classifier head (333 outputs) on the backbone features; its log-probability is added to the
    crop + disease score with a learned positive weight (adapted from the sparse linear compositor of the unified
    open-world CZSL framework, Jayasekara et al., WACV 2025).
kind "como": the pair head plus composed outliers, adapted from COMO (Jing et al., IJCAI 2025). Primitive features
    (crop part, disease part) are composed by addition. Composed features of pairs that are not training pairs are
    outliers: the outlier-exposure loss pushes the pair head's prediction on them towards uniform over the training
    pairs, and the discriminability loss makes composed features of the real absent pairs classify as those pairs.
    The paper's test-time seen/unseen gate and fusion weights are replaced by the shared gamma calibration and one
    fixed fusion weight.
kind "featsyn": feature synthesis for absent pairs. The same primitive features are composed into synthetic features
    of the absent pairs and the crop and disease heads are trained to read the composed crop and disease.
kind "calib": a learned instance-level seen/unseen detector on three score statistics (top score, margin, entropy),
    trained with the held-out pairs as positives; at scoring time each image's seen pairs are lowered by
    scale * P(unseen). Replaces the single gamma by a per-image shift (cf. multiple-criteria calibration, 2024).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, i), nn.ReLU(inplace=True), nn.Linear(i, o))


def score_stats(z: torch.Tensor, energy: bool = False) -> torch.Tensor:
    """(N, P) scores with -1e9 on masked pairs -> (N, 3): top score, margin to the second, entropy of the softmax;
    with energy, a 4th statistic: the log-sum-exp of the scores (the negative energy score)."""
    top2 = z.topk(2, dim=-1).values
    p = F.softmax(z, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    stats = [top2[:, 0] / 10.0, (top2[:, 0] - top2[:, 1]) / 10.0, ent / 5.0]
    if energy:
        stats.append(torch.logsumexp(z, dim=-1) / 10.0)
    return torch.stack(stats, dim=-1)


class NewTerms(nn.Module):
    def __init__(self, ncfg, dim, n_pairs, n_crop, n_dis):
        super().__init__()
        self.cfg, self.kind = ncfg, ncfg["kind"]
        assert self.kind in ("compositor", "como", "featsyn", "calib"), self.kind
        if self.kind in ("compositor", "como"):
            self.comp = nn.Linear(dim, n_pairs)
        if self.kind == "compositor":
            self.fuse = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) = 1
        if self.kind in ("como", "featsyn"):
            self.pc, self.pd = mlp(dim, dim), mlp(dim, dim)
            self.register_buffer("bank_c", torch.zeros(n_crop, dim))
            self.register_buffer("bank_d", torch.zeros(n_dis, dim))
            self.register_buffer("ok_c", torch.zeros(n_crop, dtype=torch.bool))
            self.register_buffer("ok_d", torch.zeros(n_dis, dtype=torch.bool))
        if self.kind == "calib":
            self.energy = bool(ncfg.get("energy"))
            # detector inputs: "basic" = top score, margin, entropy (+ energy); "topk" = the k largest seen-pair probabilities,
            # sorted (the gate input of COSMO, Atzmon & Chechik, CVPR 2019); use_feat adds a 16-d projection of the features
            self.stats_mode, self.topk = ncfg.get("stats", "basic"), int(ncfg.get("topk", 10))
            n_in = self.topk if self.stats_mode == "topk" else (4 if self.energy else 3)
            self.fproj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 16)) if ncfg.get("use_feat") else None
            n_in += 16 if self.fproj is not None else 0
            self.det = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))

    # ---- shared with scoring ----
    def fusion_weight(self):
        if self.kind == "compositor":
            return F.softplus(self.fuse)
        if self.kind == "como":
            return torch.tensor(float(self.cfg.get("fusion_weight", 1.0)))
        return None

    @torch.no_grad()
    def comp_logp(self, f):
        """(N, P) log-softmax of the pair head over all pairs, or None."""
        return F.log_softmax(self.comp(f).float(), dim=-1) if hasattr(self, "comp") else None

    def _det_in(self, z, f=None):
        if self.stats_mode == "topk":
            x = F.softmax(z, dim=-1).topk(self.topk, dim=-1).values
        else:
            x = score_stats(z, self.energy)
        if self.fproj is not None:
            x = torch.cat([x, self.fproj(f.detach().float())], dim=-1)
        return x

    @torch.no_grad()
    def p_unseen(self, scores_seen_only: torch.Tensor, f=None) -> torch.Tensor:
        return torch.sigmoid(self.det(self._det_in(scores_seen_only, f))).squeeze(-1)

    # ---- training ----
    @torch.no_grad()
    def _update_banks(self, vc, vd, crop, dis, m=0.99):
        for bank, ok, v, y in ((self.bank_c, self.ok_c, vc, crop), (self.bank_d, self.ok_d, vd, dis)):
            for c in y.unique():
                mean = v[y == c].mean(0)
                bank[c] = mean if not ok[c] else m * bank[c] + (1 - m) * mean
                ok[c] = True

    def _synth(self, vc, vd, crop, dis, absent_idx, pair_crop, pair_dis, n):
        """Synthetic features of n random absent pairs: class means of the crop and disease parts plus the residual of a
        random batch image, so they vary like real features."""
        u = absent_idx[torch.randint(len(absent_idx), (n,), device=vc.device)]
        cu, du = pair_crop[u], pair_dis[u]
        good = self.ok_c[cu] & self.ok_d[du]
        u, cu, du = u[good], cu[good], du[good]
        if len(u) == 0:
            return None
        k = torch.randint(len(vc), (len(u),), device=vc.device)
        x = (self.bank_c[cu] + (vc[k] - self.bank_c[crop[k]].where(self.ok_c[crop[k]][:, None], vc[k]))
             + self.bank_d[du] + (vd[k] - self.bank_d[dis[k]].where(self.ok_d[dis[k]][:, None], vd[k])))
        return x, u, cu, du

    def step(self, f, labels, y_pair, cand, hold, keep, zraw, pair_of, pair_crop, pair_dis, cls):
        """Extra loss and the (N, P) term added to the pair score. `zraw` = log P(crop) + log P(disease) per pair."""
        crop, dis = labels["crop"], labels["disease"]
        zero = f.new_zeros(())
        extra = 0.0
        if self.kind in ("compositor", "como"):
            lc = self.comp(f)
            ce = F.cross_entropy(lc.masked_fill(~cand, -1e9), y_pair, reduction="none")
            loss = (ce * keep).sum() / keep.sum().clamp_min(1.0)
            extra = self.fusion_weight().to(f.device) * F.log_softmax(lc, dim=-1)
        else:
            loss = zero
        if self.kind in ("como", "featsyn"):
            vc, vd = self.pc(f), self.pd(f)
            loss = loss + F.cross_entropy(cls["crop"](vc), crop) + F.cross_entropy(cls["disease"](vd), dis)
            self._update_banks(vc.detach(), vd.detach(), crop, dis)
            absent_idx = (~cand).nonzero().squeeze(-1)
            syn = self._synth(vc.detach(), vd.detach(), crop, dis, absent_idx, pair_crop, pair_dis, len(f) // 4) if len(absent_idx) else None
            if self.kind == "featsyn" and syn is not None:
                x, u, cu, du = syn
                loss = loss + self.cfg.get("weight", 1.0) * (F.cross_entropy(cls["crop"](x), cu) + F.cross_entropy(cls["disease"](x), du))
            if self.kind == "como":
                # outlier exposure: composed features of pairs that are not training pairs -> uniform over the training pairs
                perm = torch.randperm(len(f), device=f.device)
                pij = pair_of[crop, dis[perm]]
                outl = (pij < 0) | ~cand[pij.clamp_min(0)]
                if outl.any():
                    lo = self.comp((vc + vd[perm])[outl])
                    logp = F.log_softmax(lo[:, cand], dim=-1)
                    loss = loss + self.cfg.get("w_oe", 1.0) * (-logp.mean(-1)).mean()
                if syn is not None:  # discriminability: composed absent-pair features classify as those pairs
                    x, u, cu, du = syn
                    loss = loss + self.cfg.get("w_dc", 1.0) * F.cross_entropy(self.comp(x).masked_fill(cand, -1e9), u)
        if self.kind == "calib":
            seen_now = cand & ~hold
            z = zraw.detach().masked_fill(~seen_now, -1e9)
            tgt = hold[y_pair].float()
            logit = self.det(self._det_in(z, f)).squeeze(-1)
            pw = torch.tensor((1 - tgt.mean().item()) / max(tgt.mean().item(), 1e-3), device=f.device).clamp(1.0, 20.0)
            loss = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pw)
        return self.cfg.get("weight", 1.0) * loss if self.kind != "featsyn" else loss, extra
