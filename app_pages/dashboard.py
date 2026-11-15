from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dpd.data import pair_table
from dpd.evaluate import ausuc, evaluate, seen_unseen_curve
from dpd.inference import list_runs
from dpd.ui import KIND, OUTCOMES, backbone, best_run_index, load_metadata, run_label, test_predictions


@st.cache_data(max_entries=32)
def top1_breakdown(run_dir: str, gamma: float) -> pd.DataFrame:
    """Pair / crop / disease Top-1 on both test splits, with the run's gamma and without calibration (gamma 0).

    Recomputed from the run's saved scores.npz with dpd/evaluate.py, so it matches metrics.json exactly.
    Crop and disease accuracy are read off the top-1 predicted pair.
    """
    z = np.load(Path(run_dir) / "scores.npz")
    pairs = pair_table(load_metadata())
    rows = []
    for label, g in (("calibrated", gamma), ("uncalibrated", 0.0)):
        r = evaluate(z["test_seen_scores"], z["test_seen_labels"], z["test_unseen_scores"], z["test_unseen_labels"], pairs, g, (1,))
        for split, name in (("test_seen", "Seen"), ("test_unseen", "Unseen")):
            rows.append({"setting": label, "split": name, "pair": r[split]["top1"],
                         "crop": r[split]["crop_top1"], "disease": r[split]["disease_top1"]})
    return pd.DataFrame(rows)


@st.cache_data(max_entries=64)
def tradeoff(run_dir: str) -> tuple[pd.DataFrame, float]:
    """The run's seen-unseen curve over every calibration gamma (thinned to at most 400 points for plotting) and
    its AUSUC, both from the saved test scores.npz."""
    z = np.load(Path(run_dir) / "scores.npz")
    curve = seen_unseen_curve(z["test_seen_scores"], z["test_seen_labels"], z["test_unseen_scores"], z["test_unseen_labels"],
                              pair_table(load_metadata()))
    keep = np.unique(np.linspace(0, len(curve) - 1, 400).round().astype(int))
    return curve.iloc[keep].reset_index(drop=True), ausuc(curve)


runs = list_runs()
if not runs:
    st.info("No finished runs in `runs/`. Train or evaluate a model first.", title="No model yet", icon=":material/model_training:")
    st.stop()

pct = lambda label, **kw: st.column_config.ProgressColumn(label, format="percent", min_value=0.0, max_value=1.0, **kw)
gamma_of = lambda r: float(r["metrics"].get("gamma", 0.0))

# runs/<family> folder -> label; anything else (runs/baseline) is a baseline.
FAMILY_LABELS = {
    "vlm": "proposed (VLM)",
    "cnn": "proposed (CNN)",
    "cropnorm": "proposed (crop norm)",
    "multitask": "proposed (multi-task)",
    "disentangle": "proposed (disentangled)",
    "contrastive": "proposed (contrastive)",
    "ssl": "proposed (SSL)",
}

table = pd.DataFrame(
    [
        {
            "model": r["version"],
            "family": FAMILY_LABELS.get(r["family"], "baseline"),
            "method": KIND.get(r["method"], r["method"].replace("_", " ")),
            "backbone": backbone(r),
            "seen_top1": r["metrics"]["test_seen"]["top1"],
            "unseen_top1": r["metrics"]["test_unseen"]["top1"],
            "hm_top1": r["metrics"]["harmonic_mean_top1"],
            "ausuc": tradeoff(str(r["dir"]))[1],
            "seen_top5": r["metrics"]["test_seen"]["top5"],
            "unseen_top5": r["metrics"]["test_unseen"]["top5"],
            "hm_top5": r["metrics"]["harmonic_mean_top5"],
            "seen_unseen_ratio": (r["metrics"]["test_seen"]["top1"] / r["metrics"]["test_unseen"]["top1"]
                                  if r["metrics"]["test_unseen"]["top1"] else None),
            "seen_to_unseen": r["metrics"]["test_seen"]["predicted_as_unseen"],
            "unseen_to_unseen": r["metrics"]["test_unseen"]["predicted_as_unseen"],
            "gamma": gamma_of(r),
            "seen_crop": r["metrics"]["test_seen"]["crop_top1"],
            "seen_disease": r["metrics"]["test_seen"]["disease_top1"],
            "unseen_crop": r["metrics"]["test_unseen"]["crop_top1"],
            "unseen_disease": r["metrics"]["test_unseen"]["disease_top1"],
            "unseen_disease_g0": top1_breakdown(str(r["dir"]), gamma_of(r))
            .query("setting == 'uncalibrated' and split == 'Unseen'")["disease"].iloc[0],
        }
        for r in runs
    ]
)

best_of = lambda t: t.loc[t["hm_top1"].idxmax()] if len(t) else None
best = best_of(table)
best_vlm, best_base = best_of(table[table["family"] != "baseline"]), best_of(table[table["family"] == "baseline"])
with st.container(horizontal=True):
    st.metric("Models", len(table), border=True,
              help=f"{(table['family'] != 'baseline').sum()} proposed runs and {(table['family'] == 'baseline').sum()} baselines")
    st.metric("Best overall (HM Top-1)", f"{best['hm_top1']:.1%}", border=True, help=f"{best['model']} · {best['method']}")
    for label, b in (("Best proposed", best_vlm), ("Best baseline", best_base)):
        if b is not None:
            st.metric(f"{label} (HM Top-1)", f"{b['hm_top1']:.1%}", border=True, help=f"{b['model']} · {b['method']} · {b['backbone']}")
    if best_vlm is not None and best_base is not None:
        st.metric("Proposed − baseline", f"{best_vlm['hm_top1'] - best_base['hm_top1']:+.1%}", border=True,
                  help="Best proposed HM Top-1 minus the best baseline's, in percentage points")

family_color = alt.Color("family:N", title="Family", scale=alt.Scale(domain=[*FAMILY_LABELS.values(), "baseline"]))
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**Top-1 accuracy by model**")
    metric_names = {"seen_top1": "Seen", "unseen_top1": "Unseen", "hm_top1": "Harmonic mean"}
    long = table.melt(id_vars=["model", "family"], value_vars=list(metric_names), var_name="metric", value_name="accuracy")
    long["metric"] = long["metric"].map(metric_names)
    order = table.sort_values("hm_top1", ascending=False)["model"].tolist()
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=240)),
            xOffset=alt.XOffset("metric:N", sort=list(metric_names.values())),
            y=alt.Y("accuracy:Q", axis=alt.Axis(format="%"), title="Top-1"),
            color=alt.Color("metric:N", sort=list(metric_names.values()), title=None, legend=alt.Legend(orient="top")),
            tooltip=["model", "family", "metric", alt.Tooltip("accuracy:Q", format=".1%")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, alt="Seen, unseen and harmonic-mean Top-1 per model, best first")
with c2, st.container(border=True):
    st.markdown("**Seen vs unseen Top-1**")
    base = alt.Chart(table).encode(
        x=alt.X("seen_top1:Q", axis=alt.Axis(format="%"), title="Seen Top-1", scale=alt.Scale(zero=False, padding=20)),
        y=alt.Y("unseen_top1:Q", axis=alt.Axis(format="%"), title="Unseen Top-1", scale=alt.Scale(zero=False, padding=20)),
    )
    points = base.mark_circle(size=120).encode(
        color=family_color.legend(orient="top"),
        tooltip=["model", "method", "backbone", alt.Tooltip("seen_top1:Q", format=".1%"),
                 alt.Tooltip("unseen_top1:Q", format=".1%"), alt.Tooltip("hm_top1:Q", format=".1%", title="HM Top-1")],
    )
    labels = base.mark_text(dx=9, align="left", fontSize=11).encode(text="model:N")
    st.altair_chart((points + labels).properties(height=320), alt="Seen against unseen Top-1 for every model")
    st.caption("Top-right is best. The harmonic mean rewards models that do well on both, not just one.")

with st.container(border=True):
    st.markdown("**Seen / unseen trade-off over every calibration γ (AUSUC)**")
    by_hm = table.sort_values("hm_top1", ascending=False)
    default = list(dict.fromkeys([*by_hm["model"].head(3), *by_hm.loc[by_hm["family"] == "baseline", "model"].head(1)]))
    picked = st.multiselect("Models", by_hm["model"].tolist(), default=default, key="dash_ausuc_models")
    if picked:
        dirs = {r["version"]: str(r["dir"]) for r in runs}
        auc = {m: tradeoff(dirs[m])[1] for m in picked}
        name = lambda m: f"{m} (AUSUC {auc[m]:.3f})"
        curves = pd.concat([tradeoff(dirs[m])[0].assign(model=name(m)) for m in picked])
        chosen = table[table["model"].isin(picked)].assign(model=lambda d: d["model"].map(name))
        color = alt.Color("model:N", title=None, sort=[name(m) for m in picked], legend=alt.Legend(orient="top", labelLimit=320))
        axes = dict(
            x=alt.X("seen:Q", title="Seen Top-1", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("unseen:Q", title="Unseen Top-1", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        )
        lines = alt.Chart(curves).mark_line(interpolate="step-after").encode(
            **axes, color=color, order="gamma:Q",
            tooltip=["model", alt.Tooltip("gamma:Q", format=".4f", title="γ"),
                     alt.Tooltip("seen:Q", format=".1%"), alt.Tooltip("unseen:Q", format=".1%")],
        )
        dots = alt.Chart(chosen.rename(columns={"seen_top1": "seen", "unseen_top1": "unseen"})).mark_point(
            size=110, filled=True, stroke="white", strokeWidth=1).encode(
            **axes, color=color,
            tooltip=["model", alt.Tooltip("gamma:Q", format=".4f", title="validation γ"),
                     alt.Tooltip("seen:Q", format=".1%"), alt.Tooltip("unseen:Q", format=".1%"),
                     alt.Tooltip("hm_top1:Q", format=".1%", title="HM Top-1")],
        )
        st.altair_chart(alt.layer(lines, dots).properties(height=380),
                        alt="Seen against unseen Top-1 for each selected model as calibration gamma varies")
    st.caption("Each line sweeps the calibration γ from favouring seen pairs (right, unseen 0%) to favouring unseen "
               "pairs (top, seen 0%); the dot is the model's reported operating point (γ chosen on validation), whose "
               "harmonic mean is the headline number. **AUSUC** is the area under the line: 1 means perfect on both at "
               "once, and it does not depend on the chosen γ. Unseen Top-1 is a 10-pair problem once an image is "
               "assigned to the unseen side, while seen is 323 pairs, so the curves reach higher on the unseen axis.")

with st.container(border=True):
    st.markdown("**How each model's test predictions go wrong**")
    split_pick = st.segmented_control(
        "Test split", ["Both", "Seen", "Unseen"], default="Both", required=True, key="dash_outcome_split"
    )
    rows = []
    for r in runs:
        p = test_predictions(str(r["dir"]), r["method"], gamma_of(r))
        if split_pick != "Both":
            p = p[p["split"] == f"test_{split_pick.lower()}"]
        share = p["outcome"].value_counts(normalize=True)
        rows += [{"model": r["version"], "outcome": o, "share": float(share.get(o, 0.0)), "rank": i} for i, o in enumerate(OUTCOMES)]
    out = pd.DataFrame(rows)
    chart = (
        alt.Chart(out)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of test images"),
            color=alt.Color("outcome:N", sort=OUTCOMES, title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=OUTCOMES, range=["#2e9e5b", "#f2a33a", "#e0644f", "#8c2d2d"])),
            order=alt.Order("rank:Q"),
            tooltip=["model", "outcome", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=34 * len(runs) + 40)
    )
    st.altair_chart(chart, alt="Share of test images per model that are correct, wrong crop, wrong disease or both wrong")
    st.caption("Top-1 prediction against the true pair. *Wrong crop*: right disease on the wrong crop; "
               "*wrong disease*: right crop with the wrong disease.")

with st.container(border=True):
    st.markdown("**All models**")
    st.dataframe(
        table.sort_values("hm_top1", ascending=False),
        hide_index=True,
        alt="Test metrics of every finished model run",
        column_config={
            "model": st.column_config.TextColumn("Model", pinned=True),
            "family": "Family",
            "method": "Method",
            "backbone": "Backbone",
            "seen_top1": pct("Seen Top-1"),
            "unseen_top1": pct("Unseen Top-1"),
            "hm_top1": pct("HM Top-1"),
            "ausuc": st.column_config.NumberColumn("AUSUC", format="%.3f",
                                                   help="Area under the seen-unseen Top-1 curve over every calibration γ; independent of the chosen γ"),
            "seen_top5": pct("Seen Top-5"),
            "unseen_top5": pct("Unseen Top-5"),
            "hm_top5": pct("HM Top-5"),
            "seen_unseen_ratio": st.column_config.NumberColumn("Seen : unseen", format="%.2f", help="Seen Top-1 / unseen Top-1; 1.00 is balanced"),
            "seen_to_unseen": pct("Seen → unseen", help="Seen-test images predicted as an unseen pair (always wrong)"),
            "unseen_to_unseen": pct("Unseen → unseen", help="Unseen-test images predicted as some unseen pair"),
            "seen_crop": pct("Seen crop"),
            "seen_disease": pct("Seen disease"),
            "unseen_crop": pct("Unseen crop"),
            "unseen_disease": pct("Unseen disease"),
            "unseen_disease_g0": pct("Unseen disease (γ=0)",
                                     help="Disease accuracy on unseen images without calibration: how well disease knowledge transfers to a new crop"),
            "gamma": st.column_config.NumberColumn("γ", format="%.4f", help="Calibration subtracted from seen-pair scores"),
        },
    )
    st.caption("All rows use the same protocol: every test image ranked against all 333 pairs (dpd/evaluate.py).")

st.subheader("Model details")
run = st.selectbox("Model", runs, index=best_run_index(runs), format_func=run_label, key="dash_run")
metrics = run["metrics"]
if run["cfg"].get("_note"):
    st.caption(run["cfg"]["_note"])
with st.container(horizontal=True):
    st.metric("Seen Top-1", f"{metrics['test_seen']['top1']:.1%}", border=True,
              help="Accuracy on the 2,000 test images of the 323 seen pairs, ranked against all 333 pairs.")
    st.metric("Unseen Top-1", f"{metrics['test_unseen']['top1']:.1%}", border=True,
              help="Accuracy on the 1,448 test images of the 10 pairs never seen in training.")
    st.metric("HM Top-1", f"{metrics['harmonic_mean_top1']:.1%}", border=True,
              help="Harmonic mean of seen and unseen Top-1: high only when both are high.")
    st.metric("HM Top-5", f"{metrics['harmonic_mean_top5']:.1%}", border=True,
              help="Harmonic mean of seen and unseen Top-5 accuracy.")
    st.metric("Seen images → unseen pairs", f"{metrics['test_seen']['predicted_as_unseen']:.1%}", border=True,
              help="Share of seen-test images whose top prediction is one of the 10 unseen pairs (always a mistake). "
                   "Calibration γ raises this.")
    st.metric("Calibration γ", f"{gamma_of(run):g}", border=True, help="Subtracted from every seen pair's score")

c1, c2 = st.columns(2)
with c1, st.container(border=True, height="stretch"):
    st.markdown("**Top-1: pair, crop and disease, with and without calibration**")
    bd = top1_breakdown(str(run["dir"]), gamma_of(run))
    wide = bd.pivot(index="split", columns="setting", values=["pair", "crop", "disease"])
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    st.dataframe(
        wide.reset_index(),
        hide_index=True,
        alt="Pair, crop and disease Top-1 with and without calibration",
        column_config={
            "split": "Test split",
            "pair_calibrated": pct("Pair (γ)"), "pair_uncalibrated": pct("Pair (γ=0)"),
            "crop_calibrated": pct("Crop (γ)"), "crop_uncalibrated": pct("Crop (γ=0)"),
            "disease_calibrated": pct("Disease (γ)"), "disease_uncalibrated": pct("Disease (γ=0)"),
        },
        column_order=["split", "pair_calibrated", "pair_uncalibrated", "crop_calibrated", "crop_uncalibrated",
                      "disease_calibrated", "disease_uncalibrated"],
    )
    st.caption("Crop and disease accuracy are read off the top-1 predicted pair. Without calibration, unseen accuracy "
               "usually drops sharply: the model favours pairs it was trained on.")
with c2, st.container(border=True, height="stretch"):
    st.markdown("**Most frequent confusions**")
    pairs = pair_table(load_metadata())
    names = pairs["pair"].to_numpy()
    p = test_predictions(str(run["dir"]), run["method"], gamma_of(run))
    wrong = p[p["outcome"] != "Correct"]
    conf = (
        wrong.assign(true=names[wrong["true_id"]], predicted=names[wrong["pred_id"]])
        .groupby(["true", "predicted", "outcome"]).size().rename("images").reset_index()
        .sort_values("images", ascending=False).head(10)
    )
    st.dataframe(conf, hide_index=True, alt="Ten most frequent true-predicted pair confusions on both test splits")
    st.caption("Both test splits. The full seen-test confusion matrix is saved as `confusion_test_seen.csv` in the run folder.")
