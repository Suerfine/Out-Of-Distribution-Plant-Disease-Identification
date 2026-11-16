"""GAN test, step 2 (2026-11-16 night): feature generation for unseen pairs (f-CLSWGAN, Xian et al. CVPR 2018; the synthetic-feature
idea of FF-ViT) on the seed-42 control's backbone features. Rule fixed BEFORE any result:
  Generator G([noise 128, one-hot crop 55, one-hot disease 175]) -> 1024-d feature; critic D([feature, condition]) -> score.
  WGAN-GP (gradient penalty 10, 5 critic steps per generator step, Adam 1e-4, betas 0.5/0.999, batch 256, 3,000 generator steps)
  + a classification term: the frozen network heads must recognise the generated feature's crop and disease (weight beta).
  Then the two linear heads (started from the network's heads) are retrained for 3 epochs on real training features plus n_syn
  generated features per target pair (validation: the 20 pseudo-unseen pairs; test: the 10 real unseen pairs).
  References: the original heads, and the heads retrained the same way on real features only (separates the effect of retraining).
  Variants: n_syn in {50, 200} x beta in {0.01, 0.1}. Chosen by validation HM (crop weight 1.75, gamma 0-15 step 0.25), adopted only
  if it beats the real-only retrain by at least 0.5 validation HM. Test read once for the chosen variant and the two references:
  HM at the gamma chosen on validation, plus gamma-free curve measures (seen accuracy at 80/85/90% unseen, AUSUC).
Usage: python gan_test.py val | test <n_syn> <beta> <gamma_gan> <gamma_orig> <gamma_real>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[3] / "queue"
REPO = HERE.parent / "Out-Of-Distribution-Plant-Disease-Identification"
sys.path.insert(0, str(REPO))
from dpd.data import pair_table, read_metadata  # noqa: E402

G_DIR = HERE / "gan"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pairs = pair_table(read_metadata())
CID, DID, UN = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy(), pairs["is_unseen"].to_numpy()
PID = pairs["pair_id"].to_numpy()
IDX = {p: i for i, p in enumerate(PID)}
NC, ND, NZ, W = 55, 175, 128, 1.75
GAM = np.round(np.arange(0.0, 15.0, 0.25), 6)


def cond(c, d):
    return torch.cat([F.one_hot(c, NC).float(), F.one_hot(d, ND).float()], 1)


class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NZ + NC + ND, 4096), nn.LeakyReLU(0.2), nn.Linear(4096, 1024))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024 + NC + ND, 4096), nn.LeakyReLU(0.2), nn.Linear(4096, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], 1))


def heads_from(state):
    hs = {}
    for h, n in (("crop", NC), ("disease", ND)):
        lin = nn.Linear(1024, n).to(DEV)
        lin.weight.data.copy_(state[h]["weight"])
        lin.bias.data.copy_(state[h]["bias"])
        hs[h] = lin
    return hs


def train_gan(X, c, d, heads, beta, steps=3000, seed=0):
    torch.manual_seed(seed)
    X, c, d = (torch.as_tensor(a).to(DEV) for a in (X, c, d))
    G, D = Gen().to(DEV), Critic().to(DEV)
    og = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999))
    for p in (*heads["crop"].parameters(), *heads["disease"].parameters()):
        p.requires_grad_(False)
    n = len(X)
    for step in range(steps):
        for _ in range(5):
            i = torch.randint(0, n, (256,), device=DEV)
            x, cc = X[i], cond(c[i], d[i])
            fake = G(torch.randn(256, NZ, device=DEV), cc).detach()
            a = torch.rand(256, 1, device=DEV)
            mix = (a * x + (1 - a) * fake).requires_grad_(True)
            gp = torch.autograd.grad(D(mix, cc).sum(), mix, create_graph=True)[0]
            loss_d = D(fake, cc).mean() - D(x, cc).mean() + 10 * ((gp.norm(dim=1) - 1) ** 2).mean()
            od.zero_grad()
            loss_d.backward()
            od.step()
        i = torch.randint(0, n, (256,), device=DEV)
        fake = G(torch.randn(256, NZ, device=DEV), cond(c[i], d[i]))
        cls = F.cross_entropy(heads["crop"](fake), c[i]) + F.cross_entropy(heads["disease"](fake), d[i])
        loss_g = -D(fake, cond(c[i], d[i])).mean() + beta * cls
        og.zero_grad()
        loss_g.backward()
        og.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"    gan step {step}: critic {loss_d.item():.3f}, generator {loss_g.item():.3f}, cls {cls.item():.3f}", flush=True)
    return G


def synth(G, target_pairs, n_syn, seed=1):
    torch.manual_seed(seed)
    k = np.array([IDX[p] for p in target_pairs])
    c = torch.as_tensor(np.repeat(CID[k], n_syn)).to(DEV)
    d = torch.as_tensor(np.repeat(DID[k], n_syn)).to(DEV)
    with torch.no_grad():
        x = G(torch.randn(len(c), NZ, device=DEV), cond(c, d))
    return x, c, d


def retrain(state, X, c, d, extra=None, epochs=3, seed=2):
    torch.manual_seed(seed)
    hs = heads_from(state)
    X, c, d = (torch.as_tensor(a).to(DEV) for a in (X, c, d))
    if extra is not None:
        X, c, d = torch.cat([X, extra[0]]), torch.cat([c, extra[1]]), torch.cat([d, extra[2]])
    opt = torch.optim.Adam([*hs["crop"].parameters(), *hs["disease"].parameters()], 1e-4, weight_decay=0.0)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        for b in range(0, len(X), 64):
            i = perm[b:b + 64]
            xb = F.dropout(X[i], 0.2, training=True)
            loss = F.cross_entropy(hs["crop"](xb), c[i], label_smoothing=0.1) + F.cross_entropy(hs["disease"](xb), d[i], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return hs


@torch.no_grad()
def logps(hs, X):
    X = torch.as_tensor(X).to(DEV)
    return F.log_softmax(hs["crop"](X), 1).cpu().numpy(), F.log_softmax(hs["disease"](X), 1).cpu().numpy()


def hm(a, u):
    return 2 * a * u / (a + u + 1e-12)


def curve(lc_s, ld_s, ys, lc_u, ld_u, yu, seen_mask, hidden, gammas):
    """seen / unseen accuracy for each gamma (seen pairs penalised; hidden pairs out of the label space)."""
    out = []
    Ss = W * lc_s[:, CID] + ld_s[:, DID]
    Su = W * lc_u[:, CID] + ld_u[:, DID]
    Ss[:, hidden] = -1e9
    Su[:, hidden] = -1e9
    for g in gammas:
        a = ((Ss - g * seen_mask).argmax(1) == ys).mean()
        u = ((Su - g * seen_mask).argmax(1) == yu).mean()
        out.append((g, a, u))
    return np.array(out)


def summary(cv):
    h = hm(cv[:, 1], cv[:, 2])
    j = h.argmax()
    o = cv[np.argsort(cv[:, 2])]
    res = {"best_hm": h[j] * 100, "gamma": cv[j, 0], "seen": cv[j, 1] * 100, "unseen": cv[j, 2] * 100, "ausuc": np.trapezoid(o[:, 1], o[:, 2]) * 100}
    for t in (0.8, 0.85, 0.9):
        ok = cv[cv[:, 2] >= t]
        res[f"seen@{int(t * 100)}"] = ok[:, 1].max() * 100 if len(ok) else float("nan")
    return res


def show(name, r):
    print(f"  {name:34s} HM {r['best_hm']:6.2f} (gamma {r['gamma']:5.2f}, seen {r['seen']:.1f}, unseen {r['unseen']:.1f}) | AUSUC {r['ausuc']:.1f} | "
          f"seen at unseen>=80/85/90: {r['seen@80']:.1f} / {r['seen@85']:.1f} / {r['seen@90']:.1f}", flush=True)


def validate():
    z = np.load(G_DIR / "val_split1.npz")
    state = torch.load(G_DIR / "val_split1_cls.pt")
    hs0 = heads_from(state)
    lc, ld = logps(hs0, z["feat_vs"])
    print("heads reproduce saved log-probs: max diff", float(np.abs(lc - z["lc_vs"]).max()), flush=True)
    fit_pairs = z["fit_pairs"]
    seen_mask = np.isin(PID, fit_pairs).astype(float)
    hidden = UN.copy()
    target = np.setdiff1d(PID[~UN], fit_pairs)  # the pseudo-unseen pairs
    ys = np.array([IDX[p] for p in z["pair_vs"]])
    yu = np.array([IDX[p] for p in z["pair_vu"]])
    print(f"fit pairs {len(fit_pairs)}, pseudo-unseen target pairs {len(target)}", flush=True)

    def score(hs):
        a = logps(hs, z["feat_vs"])
        b = logps(hs, z["feat_vu"])
        return summary(curve(a[0], a[1], ys, b[0], b[1], yu, seen_mask, hidden, GAM))

    results = {"original heads": score(hs0)}
    results["retrained, real only"] = score(retrain(state, z["feat_fit"], z["crop_fit"], z["dis_fit"]))
    for beta in (0.01, 0.1):
        G = train_gan(z["feat_fit"], z["crop_fit"], z["dis_fit"], heads_from(state), beta)
        for n_syn in (50, 200):
            hs = retrain(state, z["feat_fit"], z["crop_fit"], z["dis_fit"], synth(G, target, n_syn))
            results[f"GAN n_syn {n_syn}, beta {beta}"] = score(hs)
    print("\nVALIDATION (split 1, crop weight 1.75):")
    for k, r in results.items():
        show(k, r)
    gans = {k: r for k, r in results.items() if k.startswith("GAN")}
    best = max(gans, key=lambda k: gans[k]["best_hm"])
    gain = gans[best]["best_hm"] - results["retrained, real only"]["best_hm"]
    print(f"\nchosen: {best}; gain over real-only retrain {gain:+.2f} (rule needs >= +0.50) -> {'ADOPT' if gain >= 0.5 else 'NOT ADOPTED'}")
    json.dump({k: {kk: float(vv) for kk, vv in r.items()} for k, r in results.items()}, open(G_DIR / "val_results.json", "w"), indent=1)


def test(n_syn, beta, g_gan, g_orig, g_real):
    z = np.load(G_DIR / "final_s42.npz")
    state = torch.load(G_DIR / "final_s42_cls.pt")
    hs0 = heads_from(state)
    lc, ld = logps(hs0, z["feat_ts"])
    print("heads reproduce saved log-probs: max diff", float(np.abs(lc - z["lc_ts"]).max()), flush=True)
    seen_mask = (~UN).astype(float)
    hidden = np.zeros(len(PID), bool)
    ys = np.array([IDX[p] for p in z["pair_ts"]])
    yu = np.array([IDX[p] for p in z["pair_tu"]])
    G = train_gan(z["feat_train"], z["crop_train"], z["dis_train"], heads_from(state), beta)
    variants = {"original heads": (hs0, g_orig),
                "retrained, real only": (retrain(state, z["feat_train"], z["crop_train"], z["dis_train"]), g_real),
                f"GAN n_syn {n_syn}, beta {beta}": (retrain(state, z["feat_train"], z["crop_train"], z["dis_train"], synth(G, PID[UN], n_syn)), g_gan)}
    print("\nTEST (seed-42 control, crop weight 1.75; HM at the gamma chosen on validation; the curve measures need no gamma):")
    out = {}
    for k, (hs, g) in variants.items():
        a, b = logps(hs, z["feat_ts"]), logps(hs, z["feat_tu"])
        np.savez_compressed(G_DIR / f"test_parts_{k.split(',')[0].replace(' ', '_')}.npz", lc_ts=a[0], ld_ts=a[1], lc_tu=b[0], ld_tu=b[1], y_ts=z['pair_ts'], y_tu=z['pair_tu'])
        cv = curve(a[0], a[1], ys, b[0], b[1], yu, seen_mask, hidden, np.round(np.arange(0.0, 15.0, 0.25), 6))
        r = summary(cv)
        at = cv[np.argmin(np.abs(cv[:, 0] - g))]
        r.update(hm_at_val_gamma=hm(at[1], at[2]) * 100, val_gamma=g, seen_at=at[1] * 100, unseen_at=at[2] * 100,
                 gamma0_seen=cv[0, 1] * 100, gamma0_unseen=cv[0, 2] * 100)
        out[k] = r
        print(f"  {k:34s} HM at validation gamma {g}: {r['hm_at_val_gamma']:.2f} (seen {r['seen_at']:.1f}, unseen {r['unseen_at']:.1f}) | "
              f"gamma 0: seen {r['gamma0_seen']:.1f}, unseen {r['gamma0_unseen']:.1f} | AUSUC {r['ausuc']:.1f} | "
              f"seen at unseen>=80/85/90: {r['seen@80']:.1f} / {r['seen@85']:.1f} / {r['seen@90']:.1f}", flush=True)
    json.dump({k: {kk: float(vv) for kk, vv in r.items()} for k, r in out.items()}, open(G_DIR / "test_results.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "val":
        validate()
    else:
        test(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
