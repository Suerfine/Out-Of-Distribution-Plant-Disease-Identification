// Builds docs/vlm/Gamma_calibration_explained.docx.  Usage: node build_gamma_docx.js <repo root>
// Run docs/vlm/diagrams/make_gamma_figures.py first: the measured figures and the sweep table come from its output.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, HeadingLevel, LevelFormat, Footer, PageNumber,
} = require("docx");

const ROOT = process.argv[2];
const DIAG = path.join(ROOT, "docs", "vlm", "diagrams");
const stats = JSON.parse(fs.readFileSync(path.join(DIAG, "gamma", "gamma_stats.json"), "utf8"));
const FONT = "Arial";
const W = 9026; // A4 content width in DXA with 1 inch margins
const GREY = "F1F3F5", BLUE = "E7F0FF", RULE = "ADB5BD";

// ---------- helpers (same look as build_breakdown_docx.js) ----------
function runs(text, base = {}) {
  const out = [];
  text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/).forEach((part) => {
    if (!part) return;
    if (part.startsWith("**")) out.push(new TextRun({ text: part.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (part.startsWith("*")) out.push(new TextRun({ text: part.slice(1, -1), italics: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: part, font: FONT, ...base }));
  });
  return out;
}
const P = (text, opt = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 288 }, ...opt });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const B = (items) => items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 276 } }));
let listNo = 0;
const N = (items) => { listNo += 1; const inst = listNo; return items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "numbers", level: 0, instance: inst }, spacing: { after: 60, line: 276 } })); };
const KEY = (text) => new Paragraph({
  children: runs(text), spacing: { before: 120, after: 160, line: 288 }, indent: { left: 200, right: 200 },
  shading: { type: ShadingType.CLEAR, fill: BLUE },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2F6FD0", space: 8 } },
});
const CODE = (lines) => lines.map((t, i) => new Paragraph({
  children: [new TextRun({ text: t, font: "Consolas", size: 19 })], indent: { left: 200, right: 200 },
  shading: { type: ShadingType.CLEAR, fill: GREY }, spacing: { after: i === lines.length - 1 ? 160 : 0, line: 276 }, keepNext: i < lines.length - 1,
}));
const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, opt = {}) {
  const paras = (Array.isArray(text) ? text : [text]).map((t) => new Paragraph({ children: runs(String(t), { size: opt.size || 19, ...(opt.bold ? { bold: true } : {}) }), spacing: { after: 40, line: 264 } }));
  return new TableCell({ children: paras, width: { size: width, type: WidthType.DXA }, borders,
    shading: opt.fill ? { type: ShadingType.CLEAR, fill: opt.fill } : undefined, margins: { top: 60, bottom: 60, left: 100, right: 100 } });
}
function table(headers, rows, widths, opt = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const hdr = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { fill: GREY, bold: true, size: opt.size })) });
  const body = rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], { size: opt.size, fill: opt.mark && opt.mark(r) ? BLUE : undefined })) }));
  return [new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] }), new Paragraph({ spacing: { after: 160 }, children: [] })];
}
function fig(n, file, caption) {
  const data = fs.readFileSync(file);
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  const width = 600, height = Math.round((h / w) * width);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data, transformation: { width, height }, altText: { title: `Figure ${n}`, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(`**Figure ${n}.** ${caption}`, { size: 18, color: "495057" }) }),
  ];
}
const pct = (v) => (100 * v).toFixed(1) + "%";

// ---------- numbers from the measured sweep ----------
const sel = stats.gamma_selected, best = stats.gamma_best_on_test_diagnostic;
const at = (g) => stats.sweep.find((r) => Math.abs(r.gamma - g) < 1e-9);
const s0 = at(0), sSel = at(sel), sBest = at(best);
const shown = [0, 0.02, 0.04, best, 0.06, sel, 0.09].filter((g, i, a) => a.indexOf(g) === i).sort((a, b) => a - b);
const cw = stats.closed_world;

// ---------- content ----------
const c = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? c.push(...x) : c.push(x)));

add(
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Gamma Calibration Explained", font: FONT, size: 44, bold: true })] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "What the calibration constant does to the scores, how it is chosen in this project, and where the method comes from", font: FONT, size: 22, color: "495057" })] }),
  KEY("**In one paragraph.** Every trained model in this project scores seen pairs higher than unseen pairs, because only seen pairs were ever the correct answer during training. **Gamma** is one constant that is subtracted from the score of every seen pair before the highest score is taken. It moves the border between 'seen' and 'unseen' predictions and changes nothing else. The method is called **calibrated stacking** and comes from **Chao et al. (2016)**. Gamma is chosen on a validation split that imitates unseen pairs, never on the test sets."),
  P(`All measured numbers below are for **${stats.run}** (the best single VLM on validation), computed from its saved test scores. They are shown to explain the mechanism. Gamma itself was never chosen on them.`),
);

// 1
add(H1("1. The problem gamma solves"),
  P("At test time each image is ranked against all 333 pairs at once: 323 seen pairs and 10 unseen pairs. The model is not told which group the image belongs to. This is the **generalised zero-shot** setting."),
  P("During training, the pair loss is a cross-entropy over the **seen pairs only**. An unseen pair has no training image, so it can never be the correct class. Two things follow:"),
  ...B([
    "The text embedding of every seen pair is **pulled towards** real image embeddings, many times. The text embedding of an unseen pair is never pulled towards any image. It is built from a crop vector and a disease vector that were trained inside *other* pairs.",
    "The model is never shown an image whose correct answer is an unseen pair, so it is **never penalised** for giving a seen pair a high score on such an image."]),
  P(`The outcome is a score advantage for seen pairs on *every* image, including images of unseen pairs. Without any correction, ${stats.run} gets **${pct(s0.seen)}** on seen-test images but only **${pct(s0.unseen)}** on unseen-test images (harmonic mean ${pct(s0.hm)}). On ${pct(stats.share_with_seen_pair_on_top_at_gamma0.unseen_images)} of the unseen-test images the top-scoring pair is a seen pair, so the correct answer cannot win.`),
  P("The CNN baseline shows the same effect more strongly: the independent ResNet-50 gets 8.2% unseen accuracy and a harmonic mean of 14.5% without calibration, and 59.7% with it. So this bias is not specific to CLIP. It comes from training on seen classes only."),
);

// 2
add(H1("2. The rule"),
  P("Let f_c(x) be the model's score for image x and pair c. For the VLM this is the cosine similarity between the image embedding and the pair's text embedding. Let S be the set of seen pairs. The prediction is:"),
  KEY("**prediction = argmax over all 333 pairs c of  [ f_c(x) - gamma x 1(c is in S) ]**"),
  P("1(c is in S) is 1 for a seen pair and 0 for an unseen pair. This is Equation 2 of Chao et al. (2016), who named it *calibrated stacking*. They describe three cases:"),
  ...B([
    "**gamma = 0**: nothing is changed. All pairs are ranked by their raw score (the paper calls this *direct stacking*).",
    "**gamma very large**: every seen pair is pushed below every unseen pair, so every image is given an unseen pair.",
    "**gamma very negative**: the opposite. Unseen pairs can never win, as in a normal classifier.",
  ]),
  P("In the code this is one line in dpd/evaluate.py. unseen_mask is True for the 10 unseen pairs, so (~unseen_mask) is 1 for seen pairs:"),
  ...CODE(["def calibrate(scores, unseen_mask, gamma):", "    return scores - gamma * (~unseen_mask)[None, :]"]),
  ...fig(1, path.join(DIAG, "breakdown", "fig07_calibration.png"), "Calibration on one example and on the whole test set (v5.5). The bar values in the top half are illustrative; the table in the figure is measured."),
);

// 3
add(H1("3. What gamma does inside the scores"),
  H2("3.1 It is a threshold on one number per image"),
  P("Subtracting the same constant from all seen pairs does not change which seen pair is best, and it does not touch the unseen pairs at all. So for each image only two candidates matter: its **best seen pair** and its **best unseen pair**. Define the gap:"),
  KEY("**d(x) = best seen score - best unseen score.**   The image is given its best unseen pair when **d(x) < gamma**, and its best seen pair otherwise."),
  P("That is the whole mechanism. Gamma is a threshold on d. With gamma = 0 an unseen pair wins only if it already has the highest raw score. Raising gamma lets an unseen pair win even when a seen pair scores up to gamma higher."),
  H2("3.2 What gamma does not do"),
  ...B([
    "It does **not** change the network. No weight is updated. It is applied after the model has produced its scores.",
    "It does **not** change the order among seen pairs or among unseen pairs. If the model confuses two seen diseases, gamma cannot fix that.",
    "It does **not** look at the image. Every image gets the same shift. This is its main limit (Sections 3.3 and 6).",
  ]),
  H2("3.3 What the gaps look like in a real model"),
  ...fig(2, path.join(DIAG, "gamma", "gamma_gap_histogram.png"), `The gap d for every test image of ${stats.run} (measured). Images left of a vertical line are given an unseen pair at that gamma.`),
  P(`Figure 2 shows the gap d of every test image. It shows why a correction is needed and why one constant can do most of the job. For seen-test images the median gap is **${stats.gap_median.seen_images.toFixed(3)}**: a seen pair wins clearly, as it should. For unseen-test images the median gap is **${stats.gap_median.unseen_images.toFixed(3)}**, which is still above zero: a seen pair wins, but only by a little. The two groups sit at different places on the d axis, so a threshold between them separates most images correctly.`),
  H2("3.4 How large is gamma, really?"),
  P(`The gamma chosen for ${stats.run} is ${sel}. That looks small, but CLIP multiplies cosines by a logit scale of 100 before the softmax. So ${sel} in cosine units is ${(sel * 100).toFixed(1)} logit units, and e^${(sel * 100).toFixed(1)} is about ${Math.round(Math.exp(sel * 100)).toLocaleString("en-US")}. In probability terms, calibration divides the probability of every seen pair by roughly that factor before comparing. This matches the size of the bias: training made seen pairs that much more 'likely' in the model's eyes.`),
  P("Gamma is always in the units of the score it is subtracted from. For the VLM that is cosine units (grid 0 to 0.1, step 0.0025). For the CNN baselines the pair score is log P(crop) + log P(disease), so gamma is in log-probability units (grid 0 to 10, step 0.1; the independent ResNet-50 uses 6.5). The two are not comparable as numbers, but they do the same job."),
);

// 4
add(H1("4. The trade-off gamma controls"),
  P("Raising gamma moves images from seen pairs to unseen pairs. Unseen accuracy goes up because unseen images can now reach their correct pair. Seen accuracy goes down because some seen images are moved to an unseen pair, which is always wrong for them. The harmonic mean H = 2 x S x U / (S + U) is highest where the two are balanced. All accuracies are Top-1."),
  ...fig(3, path.join(DIAG, "gamma", "gamma_tradeoff_curve.png"), `Seen accuracy, unseen accuracy and harmonic mean of ${stats.run} on the test sets for every gamma on the grid (measured, diagnostic only).`),
  ...table(["Gamma", "Seen", "Unseen", "Harmonic mean", "Seen images moved to an unseen pair", "Note"],
    shown.map((g) => { const r = at(g); return [String(g), pct(r.seen), pct(r.unseen), pct(r.hm), pct(r.seen_images_sent_to_unseen),
      g === 0 ? "No calibration" : g === sel ? "Chosen on validation (reported)" : g === best ? "Best on test (diagnostic)" : ""]; }),
    [900, 1200, 1300, 1400, 2026, 2200], { mark: (r) => r[5].startsWith("Chosen") }),
  P(`Plotting seen accuracy against unseen accuracy for every possible gamma gives the **seen-unseen curve**. The area under it (**AUSUC**, also from Chao et al.) summarises the model without depending on which gamma was picked. For ${stats.run} it is ${stats.ausuc.toFixed(3)}. The harmonic mean is one point on that curve; AUSUC is the whole curve.`),
);

// 5
add(H1("5. How gamma is chosen in this project"),
  P("Gamma is a hyperparameter. Choosing it on the test sets would leak test information into the result. But choosing it needs images of 'unseen' pairs, and the training set has none. The solution is to create them:"),
  ...N([
    "**Hold out 20 seen pairs completely** (at most 40 images each). Their crop and their disease still appear in other training pairs, exactly like the real unseen pairs. To the model they are unseen. These are the *pseudo-unseen* pairs.",
    "**Hold out 15% of the images** of the remaining pairs as validation-seen images.",
    "**Train on the rest.** After every epoch, score the validation images against the validation label space, and compute the harmonic mean for every gamma on the grid.",
    "**Keep the (epoch, gamma) with the best validation harmonic mean.** From v5.8 on this is the mean over 3 different splits.",
    "**Refit** a fresh model on all 8,000 training images for that many epochs, apply that gamma, and evaluate on the test sets once.",
  ]),
  ...fig(4, path.join(DIAG, "breakdown", "fig08_validation.png"), "The two-stage procedure used by every trained run. Gamma is fixed in stage 1 and only applied in stage 2."),
  P("Chao et al. tune their models in the same spirit: their supplementary material splits the training classes into folds and treats one fold's classes as 'pseudo-unseen' in each round. This project uses that idea to pick gamma itself."),
);

// 6
add(H1("6. Limits measured in this project"),
  ...B([
    `**The validation gamma is not the test-best gamma.** For ${stats.run} validation chose ${sel}; on the test sets the harmonic mean would peak at ${best} (${pct(sBest.hm)} instead of the reported ${pct(sSel.hm)}). At ${sel}, ${pct(sSel.seen_images_sent_to_unseen)} of seen-test images are moved to an unseen pair. The reported number is the honest one: the ${best} value is known only by looking at the test sets.`,
    "**The validation must mirror the test.** In v4 the validation ranked images against 30 untrained pairs (the 20 held-out and the 10 real unseen pairs), while the test has 10. More untrained competitors need a larger gamma, so gamma was over-estimated (0.045) and 53% of seen-test images were moved to unseen pairs. v4.1 removed the real unseen pairs from the validation label space; gamma dropped to 0.0375 and the harmonic mean rose from 43.6% to 52.9% (the learning rate was lowered in the same step).",
    "**Twenty held-out pairs is a small sample.** In v4.2 three different splits asked for gamma 0.0425, 0.0275 and 0.0375. Averaging over splits reduces this noise.",
    `**One constant cannot separate the groups image by image.** The two histograms in Figure 2 overlap, so any threshold misplaces some images. If seen images were ranked among seen pairs only and unseen images among unseen pairs only, ${stats.run} would reach ${pct(cw.seen)} and ${pct(cw.unseen)} (harmonic mean ${pct(cw.hm)}). The best any single gamma can do on test is ${pct(sBest.hm)}. That gap of ${(100 * (cw.hm - sBest.hm)).toFixed(1)} points is the cost of deciding seen-versus-unseen with one global constant, and it is the same pattern for the later models (v10.3: 76.0% against 67.9%).`,
  ]),
);

// 7
add(H1("7. Where the method comes from"),
  ...table(["Source", "What it contributes", "How this project uses it"], [
    ["**Chao, Changpinyo, Gong and Sha (2016)**. *An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild.* ECCV 2016. **Primary source.**",
      "Shows that in the generalised setting, unseen-class images are almost always classified as a seen class. Proposes **calibrated stacking** (subtract a calibration factor gamma from seen-class scores, their Eq. 2) and the **AUSUC** metric (area under the seen-unseen accuracy curve).",
      "The rule in dpd/evaluate.py is their Eq. 2, with the same symbol gamma. seen_unseen_curve() and ausuc() implement their curve and its area."],
    ["**Xian, Lampert, Schiele and Akata (2019)**. *Zero-Shot Learning: A Comprehensive Evaluation of the Good, the Bad and the Ugly.* IEEE TPAMI 41(9).",
      "Proposes the **harmonic mean** of seen and unseen accuracy as the generalised zero-shot score, because an arithmetic mean is dominated by a high seen accuracy.",
      "The headline metric. One difference: they average accuracy per class; this project reports per-image Top-1 and keeps the per-class mean as a check."],
    ["**Purushwalkam, Nickel, Gupta and Ranzato (2019)**. *Task-Driven Modular Networks for Zero-Shot Compositional Learning.* ICCV 2019.",
      "Brings the same calibration to **compositional** zero-shot learning (attribute + object pairs, the same structure as crop + disease). They add a scalar bias to the unseen pairs, citing Chao et al., and report the area under the resulting curve.",
      "Supports using the method for pairs. Adding a bias to unseen pairs and subtracting gamma from seen pairs are the same thing: only the difference between the two groups matters."],
    ["**Nayak, Yu and Bach (2023)**. *Learning to Compose Soft Prompts for Compositional Zero-Shot Learning.* ICLR 2023.",
      "The prompt-learning method (CSP) that the VLM's crop and disease vectors follow. It is evaluated with the same calibration bias, citing Chao et al.",
      "Shows calibration is the standard protocol for the kind of model built here."],
  ], [2700, 3300, 3026], { size: 18 }),
  H2("7.1 One difference to state clearly"),
  P(`The compositional papers above sweep the bias on the **test** set and report the best harmonic mean on the curve. This project fixes gamma on **validation** before the test sets are touched, which is stricter. For ${stats.run} the two conventions give ${pct(sBest.hm)} (best on the test curve) and ${pct(sSel.hm)} (reported). When comparing with published numbers, say which convention is used.`),
  H2("7.2 References"),
  ...N([
    "W.-L. Chao, S. Changpinyo, B. Gong and F. Sha, \"An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild,\" in *Computer Vision - ECCV 2016*, Lecture Notes in Computer Science, vol. 9906, Springer, 2016. doi: 10.1007/978-3-319-46475-6_4. arXiv:1605.04253.",
    "Y. Xian, C. H. Lampert, B. Schiele and Z. Akata, \"Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly,\" *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 41, no. 9, pp. 2251-2265, 2019. doi: 10.1109/TPAMI.2018.2857768. arXiv:1707.00600.",
    "S. Purushwalkam, M. Nickel, A. Gupta and M. Ranzato, \"Task-Driven Modular Networks for Zero-Shot Compositional Learning,\" in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 3592-3601. doi: 10.1109/ICCV.2019.00369. arXiv:1905.05908.",
    "N. V. Nayak, P. Yu and S. H. Bach, \"Learning to Compose Soft Prompts for Compositional Zero-Shot Learning,\" in *International Conference on Learning Representations (ICLR)*, 2023. arXiv:2204.03574.",
  ]),
  H2("7.3 A sentence you can adapt for the report"),
  KEY("\"Because the models are trained on seen pairs only, their scores are biased towards seen pairs. We correct this with calibrated stacking [Chao et al., 2016]: a constant gamma is subtracted from the score of every seen pair before ranking. Gamma is selected on a validation split in which whole seen pairs are held out as pseudo-unseen pairs, and is never tuned on the test sets.\""),
);

// 8
add(H1("8. Where it is in the code"),
  ...table(["File", "What it does"], [
    ["dpd/evaluate.py: calibrate()", "Subtracts gamma from the seen-pair columns of the score matrix. Used by every model."],
    ["dpd/evaluate.py: top1_over_gammas(), hm_top1_over_gammas()", "Accuracy and harmonic mean for a whole grid of gammas at once, using the gap d from Section 3.1."],
    ["dpd/evaluate.py: seen_unseen_curve(), ausuc()", "The seen-unseen curve and the area under it."],
    ["dpd/splits.py", "Builds the validation split with pseudo-unseen pairs."],
    ["train/train.py, train/train_lora.py, train/train_cnn.py (stage 1)", "Compute the validation harmonic mean for every (epoch, gamma) and keep the best. The grid is validation.gamma_grid in the config."],
    ["train/ensemble.py", "Chooses the mixing weight and gamma together on validation, on log-probability scores."],
    ["runs/.../metrics.json", "Stores the chosen gamma, the calibrated result, and the uncalibrated harmonic mean (harmonic_mean_top1_gamma0)."],
    ["dpd/inference.py", "Applies the stored gamma when a saved run predicts a new image."],
    ["docs/vlm/diagrams/make_gamma_figures.py", "Regenerates Figures 2 and 3 and the numbers in this document from a run's saved scores."],
  ], [3600, W - 3600]),
);

// 9
add(H1("9. Short answers for the Q&A"),
  ...table(["Question", "Answer"], [
    ["What is gamma?", "A constant subtracted from the score of every seen pair before the highest score is taken. It cancels the score advantage that seen pairs get from training."],
    ["Why are seen pairs favoured?", "The training loss only ever has a seen pair as the correct answer. Seen-pair text embeddings are pulled towards images; unseen-pair embeddings never are, and the model is never penalised for scoring a seen pair high on an unseen image."],
    ["Is gamma learned by the network?", "No. It is a hyperparameter applied to the finished scores. No weight changes."],
    ["Is choosing gamma cheating?", "No. It is chosen on held-out training pairs that imitate unseen pairs. The test sets are used once, after gamma is fixed."],
    ["Does the model know which pairs are unseen?", "The evaluation knows which 10 pairs have no training images; that is part of the task definition. It does not know which *images* are unseen: every image gets the same shift and is ranked against all 333 pairs."],
    ["Why not a different gamma per image or per crop?", "A single constant is the published, standard method and has one value to fit on a small validation set. Its cost is measured in Section 6. A crop-conditional calibration is future work; it was not validated in this project."],
    ["Why does the harmonic mean rise so much with calibration?", `The harmonic mean rewards balance. Without calibration ${stats.run} is ${pct(s0.seen)} seen and ${pct(s0.unseen)} unseen (${pct(s0.hm)}). With gamma ${sel} it is ${pct(sSel.seen)} and ${pct(sSel.unseen)} (${pct(sSel.hm)}): seen falls by ${Math.round(100 * (s0.seen - sSel.seen))} points, unseen rises by ${Math.round(100 * (sSel.unseen - s0.unseen))}.`],
    ["Where does the method come from?", "Calibrated stacking, Chao et al., ECCV 2016. The harmonic mean is from Xian et al. (2019)."],
  ], [2900, W - 2900]),
);

// ---------- document ----------
const heading = (size, before, after) => ({ run: { font: FONT, size, bold: true, color: "1B1F24" }, paragraph: { spacing: { before, after }, keepNext: true } });
const doc = new Document({
  creator: "VLM pipeline", title: "Gamma Calibration Explained",
  styles: { default: { document: { run: { font: FONT, size: 21 } }, heading1: heading(30, 360, 160), heading2: heading(24, 240, 120) } },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 320 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Gamma calibration explained  |  page ", PageNumber.CURRENT], font: FONT, size: 16, color: "868E96" })] })] }) },
    children: c,
  }],
});
const out = path.join(ROOT, "docs", "vlm", "Gamma_calibration_explained.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length); });
