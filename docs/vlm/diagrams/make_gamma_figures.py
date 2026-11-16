"""Measured figures for docs/vlm/Gamma_calibration_explained.docx, from a finished run's saved test scores.

    .venv/Scripts/python docs/vlm/diagrams/make_gamma_figures.py [runs/vlm/v5.5]

Writes gamma_gap_histogram.{svg,png}, gamma_tradeoff_curve.{svg,png} and gamma_stats.json to docs/vlm/diagrams/gamma/.
The PNGs are screenshots of the SVGs taken with headless Edge (no internet needed).
Colour: dark grey = seen, blue = unseen, amber = harmonic mean / the chosen gamma. Diagnostic only: test scores are
shown to explain the mechanism, never to choose gamma.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.evaluate import ausuc, seen_unseen_curve, top1_over_gammas  # noqa: E402

OUT = ROOT / "docs" / "vlm" / "diagrams" / "gamma"
SEEN, UNSEEN, HM, INK, RULE = "#495057", "#2F6FD0", "#E08A00", "#1B1F24", "#CED4DA"
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
W, H, L, R, T, B = 900, 430, 70, 30, 56, 62  # canvas and plot margins


def text(x, y, s, size=13, anchor="middle", color=INK, weight="normal", rotate=None):
    rot = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}"{rot}>{s}</text>'


def frame(title, xlabel, ylabel, x0, x1, y0, y1, xticks, yticks, xfmt, yfmt):
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    sy = lambda v: H - B - (v - y0) / (y1 - y0) * (H - T - B)  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Arial">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>', text(L, 26, title, 16, "start", weight="bold")]
    for v in yticks:
        out += [f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{RULE}" stroke-width="1"/>',
                text(L - 8, sy(v) + 4, yfmt(v), 12, "end")]
    for v in xticks:
        out += [f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{H - B}" y2="{H - B + 5}" stroke="{INK}"/>',
                text(sx(v), H - B + 20, xfmt(v), 12)]
    out += [f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="{INK}"/>',
            text((L + W - R) / 2, H - 14, xlabel, 13), text(18, (T + H - B) / 2, ylabel, 13, rotate=-90)]
    return out, sx, sy


def vline(sx, x, label, color, y_text, dash="5,4", anchor="start"):
    dx = 5 if anchor == "start" else -5
    return [f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H - B}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>',
            text(sx(x) + dx, y_text, label, 12, anchor, color, "bold")]


def gap_histogram(ds, du, g_sel):
    x0, x1, step = -0.10, 0.30, 0.01
    edges = np.arange(x0, x1 + 1e-9, step)
    hs = np.histogram(np.clip(ds, x0, x1 - 1e-9), edges)[0] / len(ds)
    hu = np.histogram(np.clip(du, x0, x1 - 1e-9), edges)[0] / len(du)
    top = 0.16
    out, sx, sy = frame("Gap between the best seen-pair score and the best unseen-pair score, per test image",
                        "gap d = best seen score - best unseen score (cosine units)", "share of images",
                        x0, x1, 0, top, np.arange(x0, x1 + 1e-9, 0.05), np.arange(0, top + 1e-9, 0.04),
                        lambda v: f"{v:+.2f}".replace("+0.00", "0").replace("-0.00", "0"), lambda v: f"{v:.0%}")
    bw = sx(step) - sx(0)
    for h, color in ((hs, SEEN), (hu, UNSEEN)):
        for e, v in zip(edges[:-1], h):
            if v > 0:
                out.append(f'<rect x="{sx(e) + 0.5:.1f}" y="{sy(v):.1f}" width="{bw - 1:.1f}" height="{sy(0) - sy(v):.1f}" fill="{color}" fill-opacity="0.62"/>')
    out += vline(sx, 0.0, "gamma = 0: border at d = 0", INK, T + 12, anchor="end")
    out += vline(sx, g_sel, f"gamma = {g_sel:g} (chosen on validation)", HM, T + 12, dash="none")
    out += [text(sx(g_sel) - 8, T + 42, "&#8592; unseen pair wins", 12, "end", HM),
            text(sx(g_sel) + 8, T + 42, "seen pair wins &#8594;", 12, "start", HM)]
    lx = W - R - 250
    out += [f'<rect x="{lx}" y="{T + 60}" width="14" height="14" fill="{UNSEEN}" fill-opacity="0.62"/>', text(lx + 22, T + 72, f"unseen-test images (n = {len(du):,})", 12, "start"),
            f'<rect x="{lx}" y="{T + 82}" width="14" height="14" fill="{SEEN}" fill-opacity="0.62"/>', text(lx + 22, T + 94, f"seen-test images (n = {len(ds):,})", 12, "start")]
    return "\n".join(out + ["</svg>"])


def tradeoff_curve(gs, seen, unseen, hm, g_sel):
    out, sx, sy = frame("Seen accuracy, unseen accuracy and harmonic mean as gamma grows", "gamma (subtracted from every seen-pair score, cosine units)",
                        "Top-1 accuracy", gs[0], gs[-1], 0.2, 0.9, np.arange(0, gs[-1] + 1e-9, 0.01), np.arange(0.2, 0.91, 0.1),
                        lambda v: f"{v:.2f}", lambda v: f"{v:.0%}")
    out += vline(sx, g_sel, f"chosen on validation: {g_sel:g}", HM, T + 12, dash="none", anchor="end" if g_sel > 0.06 else "start")
    # (values, colour, label, line width, grid index of the label, label offset in px): placed where no other line passes
    for y, color, name, width, at, dy in ((seen, SEEN, "seen", 2.5, 3, -9), (unseen, UNSEEN, "unseen", 2.5, 20, -12), (hm, HM, "harmonic mean", 3.5, 35, 22)):
        pts = " ".join(f"{sx(g):.1f},{sy(v):.1f}" for g, v in zip(gs, y))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        out.append(text(sx(gs[at]), sy(y[at]) + dy, name, 12.5, "middle", color, "bold"))
    i = int(np.argmin(np.abs(gs - g_sel)))
    out.append(f'<circle cx="{sx(gs[i]):.1f}" cy="{sy(hm[i]):.1f}" r="5.5" fill="{HM}" stroke="#fff" stroke-width="1.5"/>')
    out.append(text(sx(gs[i]) + 9, sy(hm[i]) - 9, f"HM {hm[i]:.1%}", 12.5, "start", HM, "bold"))
    return "\n".join(out + ["</svg>"])


def to_png(svg: Path, png: Path):
    from PIL import Image, ImageChops

    edge = next(p for p in EDGE if Path(p).exists())
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "p.html"
        page.write_text(f'<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:20px;background:#fff">{svg.read_text(encoding="utf-8")}</body></html>', encoding="utf-8")
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={W + 60},{H + 60}",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--screenshot={png}", page.as_uri()], check=True, capture_output=True, timeout=120)
    img = Image.open(png).convert("RGB")
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    img.crop((max(box[0] - 12, 0), max(box[1] - 12, 0), min(box[2] + 12, img.width), min(box[3] + 12, img.height))).save(png)


def main():
    run = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/vlm/v5.5")
    z, metrics = np.load(run / "scores.npz"), json.loads((run / "metrics.json").read_text())
    pairs = pair_table(read_metadata())
    um = pairs["is_unseen"].to_numpy()
    ss, su, ls, lu = z["test_seen_scores"], z["test_unseen_scores"], z["test_seen_labels"], z["test_unseen_labels"]
    g_sel = float(metrics["gamma"])
    gs = np.round(np.arange(0, 0.1001, 0.0025), 4)
    seen, unseen = top1_over_gammas(ss, ls, um, gs), top1_over_gammas(su, lu, um, gs)
    hm = 2 * seen * unseen / np.maximum(seen + unseen, 1e-12)
    ds, du = (s[:, ~um].max(1) - s[:, um].max(1) for s in (ss, su))

    OUT.mkdir(parents=True, exist_ok=True)
    for name, svg in (("gamma_gap_histogram", gap_histogram(ds, du, g_sel)), ("gamma_tradeoff_curve", tradeoff_curve(gs, seen, unseen, hm, g_sel))):
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
        to_png(OUT / f"{name}.svg", OUT / f"{name}.png")
    closed_s = float((np.flatnonzero(~um)[ss[:, ~um].argmax(1)] == ls).mean())
    closed_u = float((np.flatnonzero(um)[su[:, um].argmax(1)] == lu).mean())
    stats = {
        "run": run.name, "gamma_selected": g_sel, "gamma_best_on_test_diagnostic": float(gs[hm.argmax()]), "hm_best_on_test_diagnostic": float(hm.max()),
        "ausuc": ausuc(seen_unseen_curve(ss, ls, su, lu, pairs)), "closed_world": {"seen": closed_s, "unseen": closed_u, "hm": 2 * closed_s * closed_u / (closed_s + closed_u)},
        "gap_median": {"seen_images": float(np.median(ds)), "unseen_images": float(np.median(du))},
        "share_with_seen_pair_on_top_at_gamma0": {"seen_images": float((ds > 0).mean()), "unseen_images": float((du > 0).mean())},
        "sweep": [{"gamma": float(g), "seen": float(a), "unseen": float(b), "hm": float(h), "seen_images_sent_to_unseen": float((ds < g).mean()),
                   "unseen_images_sent_to_unseen": float((du < g).mean())} for g, a, b, h in zip(gs, seen, unseen, hm)],
    }
    (OUT / "gamma_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "sweep"}, indent=1))


if __name__ == "__main__":
    main()
