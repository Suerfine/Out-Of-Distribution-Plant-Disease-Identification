"""Feature-generating zero-shot learning (f-CLSWGAN, Xian et al. 2018) on frozen CLIP image embeddings.

The unseen pairs have no images, but they do have text. A conditional generator G(noise, text embedding) -> image
embedding is trained as a WGAN-GP on the seen pairs' real (cached, frozen) CLIP image embeddings, plus a
classification loss that keeps generated features recognisable as their pair. Then G is fed the unseen pairs' text
embeddings to synthesise image features for them, and an ordinary classifier over all pairs is trained on real seen
+ synthetic unseen features. No unseen image is ever used: only the unseen pairs' names / descriptions.

Everything works on L2-normalised 512-d features, so one GAN epoch takes seconds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, cond_dim: int, noise_dim: int, hidden: int, out_dim: int, normalize: bool):
        super().__init__()
        self.noise_dim, self.normalize = noise_dim, normalize
        self.net = nn.Sequential(nn.Linear(cond_dim + noise_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, out_dim))

    def forward(self, cond):
        z = torch.randn(len(cond), self.noise_dim, device=cond.device)
        x = self.net(torch.cat([z, cond], dim=1))
        return F.normalize(x, dim=-1) if self.normalize else x


class Critic(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_dim + cond_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x, cond):
        return self.net(torch.cat([x, cond], dim=1)).squeeze(1)


class CosineClassifier(nn.Module):
    """logits = scale * cos(x, w_pair); one weight vector per pair, optionally initialised from text embeddings."""

    def __init__(self, init: torch.Tensor, scale: float):
        super().__init__()
        self.weight = nn.Parameter(init.clone())
        self.scale = scale

    def forward(self, x):
        return self.scale * F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).T


def fit_classifier(x, y, allowed, init, ccfg, epochs, on_epoch=None, log=print, tag="classifier"):
    """Train a CosineClassifier with cross-entropy over the `allowed` pair columns only (the rest masked out).

    x (n x d), y (n,) pair ids, allowed (n_pairs,) bool, init (n_pairs x d); all tensors on one device.
    on_epoch(epoch, model) is called after every epoch (e.g. validation) and may return a dict to log.
    """
    model = CosineClassifier(init, ccfg["logit_scale"]).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=ccfg["lr"], weight_decay=ccfg["weight_decay"])
    mask = torch.where(allowed, 0.0, float("-inf")).to(x.device)
    n, bs = len(x), ccfg["batch_size"]
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=x.device)
        total, correct = 0.0, 0
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            logits = model(x[idx]) + mask
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            correct += (logits.argmax(1) == y[idx]).sum().item()
        record = {f"{tag}_epoch": epoch, "loss": round(total / n, 4), "train_acc": round(correct / n, 4)}
        if on_epoch:
            record.update(on_epoch(epoch, model) or {})
        log(" ".join(f"{k}={v}" for k, v in record.items()))
    return model


def gradient_penalty(critic, real, fake, cond):
    alpha = torch.rand(len(real), 1, device=real.device)
    mix = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grad = torch.autograd.grad(critic(mix, cond).sum(), mix, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_gan(x, y, cond, aux, gcfg, log=print):
    """WGAN-GP with a classification loss (f-CLSWGAN). x: real features, y: their pair ids, cond: (n_pairs x c)
    condition per pair, aux: a frozen classifier over the trained pairs (logits over all pairs) for the cls loss."""
    device = x.device
    g = Generator(cond.shape[1], gcfg["noise_dim"], gcfg["hidden"], x.shape[1], gcfg["normalize_output"]).to(device)
    d = Critic(x.shape[1], cond.shape[1], gcfg["hidden"]).to(device)
    betas = tuple(gcfg["betas"])
    opt_g = torch.optim.Adam(g.parameters(), lr=gcfg["lr"], betas=betas)
    opt_d = torch.optim.Adam(d.parameters(), lr=gcfg["lr"], betas=betas)
    for p in aux.parameters():
        p.requires_grad_(False)
    n, bs = len(x), gcfg["batch_size"]
    iters = int(np.ceil(n / bs))
    for epoch in range(1, gcfg["epochs"] + 1):
        sums = {"critic": 0.0, "wdist": 0.0, "gen_adv": 0.0, "gen_cls": 0.0}
        for _ in range(iters):
            for _ in range(gcfg["n_critic"]):
                idx = torch.randint(0, n, (bs,), device=device)
                real, c = x[idx], cond[y[idx]]
                with torch.no_grad():
                    fake = g(c)
                d_real, d_fake = d(real, c).mean(), d(fake, c).mean()
                loss_d = d_fake - d_real + gcfg["lambda_gp"] * gradient_penalty(d, real, fake, c)
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
            idx = torch.randint(0, n, (bs,), device=device)
            c = cond[y[idx]]
            fake = g(c)
            adv = -d(fake, c).mean()
            cls = F.cross_entropy(aux(fake), y[idx])
            loss_g = adv + gcfg["cls_weight"] * cls
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            sums["critic"] += loss_d.item() / iters
            sums["wdist"] += (d_real - d_fake).item() / iters
            sums["gen_adv"] += adv.item() / iters
            sums["gen_cls"] += cls.item() / iters
        log(f"gan_epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in sums.items()))
    return g.eval()


@torch.no_grad()
def synthesise(g, cond, counts: dict[int, int]):
    """`counts[pair_id]` generated features per pair -> (features, pair ids)."""
    xs, ys = [], []
    for p, k in counts.items():
        if k > 0:
            xs.append(g(cond[p].expand(k, -1)))
            ys.append(torch.full((k,), p, device=cond.device, dtype=torch.long))
    if not xs:
        return torch.empty(0, g.net[-1].out_features, device=cond.device), torch.empty(0, device=cond.device, dtype=torch.long)
    return torch.cat(xs), torch.cat(ys)
