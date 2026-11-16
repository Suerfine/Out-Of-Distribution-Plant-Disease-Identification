// Builds docs/vlm/VLM_technical_breakdown.docx.  Usage: node build.js <repo root>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, HeadingLevel, LevelFormat, TableOfContents, PageBreak, Footer, PageNumber,
} = require("docx");

const ROOT = process.argv[2];
const FIG = path.join(ROOT, "docs", "vlm", "diagrams", "breakdown");
const FONT = "Arial";
const W = 9026; // A4 content width in DXA with 1 inch margins
const GREY = "F1F3F5", BLUE = "E7F0FF", RULE = "ADB5BD";

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold** and *italic* inside a string
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
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], pageBreakBefore: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const B = (items) => items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 276 } }));
let listNo = 0;
const N = (items, ref = "numbers") => { listNo += 1; const inst = listNo; return items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: ref, level: 0, instance: inst }, spacing: { after: 60, line: 276 } })); };
const KEY = (text) => new Paragraph({
  children: runs(text), spacing: { before: 120, after: 160, line: 288 }, indent: { left: 200, right: 200 },
  shading: { type: ShadingType.CLEAR, fill: BLUE },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2F6FD0", space: 8 } },
});
const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, opt = {}) {
  const paras = (Array.isArray(text) ? text : [text]).map((t) => new Paragraph({ keepNext: !!opt.keep, children: runs(String(t), { size: opt.size || 19, bold: opt.bold }), spacing: { after: 40, line: 264 }, alignment: opt.align }));
  return new TableCell({ children: paras, width: { size: width, type: WidthType.DXA }, borders,
    shading: opt.fill ? { type: ShadingType.CLEAR, fill: opt.fill } : undefined, margins: { top: 60, bottom: 60, left: 100, right: 100 } });
}
function table(headers, rows, widths, opt = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const hdr = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { fill: GREY, bold: true, size: opt.size })) });
  const body = rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], { size: opt.size })) }));
  return [new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] }), new Paragraph({ spacing: { after: 160 }, children: [] })];
}
// One experiment: what changed / result / why / how sure
function exp(title, change, result, why, sure) {
  const w = [1500, W - 1500];
  const row = (k, v, keep = true) => new TableRow({ cantSplit: true, children: [cell(k, w[0], { bold: true, keep }), cell(v, w[1], { keep })] });
  const head = new TableRow({ cantSplit: true, children: [new TableCell({ columnSpan: 2, width: { size: W, type: WidthType.DXA }, borders,
    shading: { type: ShadingType.CLEAR, fill: GREY }, margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ keepNext: true, children: runs(title, { bold: true, size: 20 }) })] })] });
  const rowsOut = [head, row("What changed", change), row("Result", result), row("Why (mechanism)", why, !!sure)];
  if (sure) rowsOut.push(row("How sure", sure, false));
  return [new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: w, rows: rowsOut }), new Paragraph({ spacing: { after: 160 }, children: [] })];
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function fig(n, name, caption) {
  const { w, h, data } = pngSize(path.join(FIG, name + ".png"));
  const width = 600, height = Math.round((h / w) * width);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data, transformation: { width, height }, altText: { title: `Figure ${n}`, description: caption, name: name } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(`**Figure ${n}.** ${caption}`, { size: 18, color: "495057" }) }),
  ];
}

// ---------- content ----------
const c = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? c.push(...x) : c.push(x)));

// Title page
add(
  new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "VLM Pipeline: Technical Breakdown", font: FONT, size: 52, bold: true })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Out-of-distribution plant disease identification (DPD-Small)", font: FONT, size: 28 })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "How each stage of the CLIP pipeline works inside the network, what every experiment changed, and why the numbers moved", font: FONT, size: 22, color: "495057" })] }),
  P("**Purpose.** This is a working document to draw the report from, and to prepare for the question-and-answer session. It is longer than the report allows on purpose. Every experiment has the same four parts: what changed, the result, the mechanism, and how sure we are of that mechanism."),
  P("**How to read the numbers.** *Validation HM* is the harmonic mean on a validation split carved from the training set; every keep-or-drop decision was made on it. *Test HM* is the harmonic mean on the two official test sets, reported once per run and never used to decide. All accuracies are Top-1 unless stated."),
  P("**How to read the colours in the figures.** Grey is pretrained CLIP that stays frozen. Blue is what this project trains. Green is data going in. Amber is a score or a decision. Red is a loss, which exists only during training."),
  P("**Honesty labels.** *Measured* means an experiment in this project directly supports the statement. *Likely* means it is the most reasonable explanation but no experiment isolated it. Keep that distinction in the report and in the Q&A."),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
);

// 1
add(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "1. The task in one page", font: FONT })] }),
  P("Every class is a **pair**: one crop and one disease, for example *tomato + early blight*. There are 55 crops, 175 diseases and 333 pairs. Training images exist for 323 pairs (the **seen** pairs). The other 10 pairs are **unseen**: the model never gets an image of them, but it does get images of the same crop with other diseases and of the same disease on other crops."),
  ...table(["Split", "Images", "Pairs", "Used for"], [
    ["Train", "8,000", "323 seen", "Training and validation"],
    ["Test seen", "2,000", "323 seen", "Seen accuracy"],
    ["Test unseen", "1,448", "10 unseen", "Unseen accuracy"]], [2000, 1500, 1800, 3726]),
  P("At test time every image is ranked against **all 333 pairs at once** (the generalised zero-shot setting). The model is not told whether an image is from a seen or an unseen pair. The headline number is the **harmonic mean** of seen accuracy S and unseen accuracy U:"),
  KEY("**H = 2 x S x U / (S + U)**.  The harmonic mean is high only when both S and U are high. A model with S = 70% and U = 5% gets H = 9%, not 37%. This is why a model that ignores unseen pairs scores badly even if it is excellent on seen pairs."),
  H2("1.1 What makes this dataset hard"),
  ...B([
    "**Very few images per pair.** The median seen pair has 7 training images, and 139 pairs have 5 or fewer. A network with millions of free weights can memorise 7 photos easily.",
    "**The disease is a small part of the picture.** Leaf shape (the crop) fills the image; a lesion may cover a few percent of the pixels.",
    "**The unseen test set is uneven.** *tomato + early blight* is 472 of the 1,448 unseen images (33%). The two largest unseen pairs are 51% of the unseen set, so they dominate unseen accuracy.",
    "**Most diseases appear on one crop only.** 133 of the 175 diseases occur with a single crop in training. For those, the model cannot separate 'what this disease looks like' from 'what this crop looks like'. Only 42 diseases are seen on two or more crops, and all 10 unseen pairs use diseases from that group.",
    "**Field photos from 46 source datasets.** Accuracy differs strongly by source (Section 10.3)."]),
  H2("1.2 Why a vision-language model is a natural fit"),
  P("A normal classifier has one output unit per class. A class with no training image has an output unit that was never trained, so it can never win. A vision-language model (VLM) has **no fixed output units**. A class is a *sentence*, turned into a vector by a text encoder. To add a class you write a new sentence. An unseen pair therefore has a meaningful class vector on day one, built from words whose meaning the model already knows. Figure 4 shows this for *tomato + early blight*."),
);

// 2
add(H1("2. The pipeline end to end"),
  ...fig(1, "fig01_pipeline", "The whole pipeline. Grey boxes are pretrained CLIP and never change. Blue boxes are the only trained parts."),
  P("Follow one photo through Figure 1:"),
  ...N([
    "The photo is resized so its shorter side is 224 pixels and the centre 224 x 224 square is kept. It is cut into 14 x 14 = 196 squares of 16 x 16 pixels (**patches**).",
    "The **image encoder** (a Vision Transformer, ViT-B/16, 12 blocks, about 86 million weights) turns the 196 patches plus one extra *class token* into one vector of 512 numbers, the **image embedding v**. It is scaled to length 1.",
    "Each of the 333 pairs is written as a sentence (a **prompt**), for example *a photo of tomato leaves with early blight: brown target spots with concentric rings and yellow halos on older leaves.* The **text encoder** (a 12-block Transformer) turns each sentence into a vector of 512 numbers, the **text embedding t**, also of length 1.",
    "The **score** of a pair is the cosine similarity between v and t. Because both have length 1, this is just the dot product v . t, a number between -1 and 1. Higher means the image and the sentence 'point the same way'.",
    "**Calibration**: a constant gamma is subtracted from the scores of the 323 seen pairs (Section 6).",
    "The pair with the highest score is the prediction. Top-5 counts a hit if the true pair is among the five highest."]),
  KEY("**Everything in this project is one of three things:** (a) changing the *text* that goes into the text encoder, (b) training a *small number of extra weights* while CLIP stays frozen, or (c) changing *how scores are turned into a decision*. No experiment retrains CLIP itself."),
  H2("2.1 Where each stage lives in the code"),
  ...table(["Stage", "Versions", "Script", "What is trained"], [
    ["Zero-shot", "v1 - v3.9", "train/zeroshot.py", "Nothing"],
    ["Prompt learning", "v4 - v4.3", "train/train.py + dpd/prompt_learner.py", "230 word vectors (118k numbers)"],
    ["Prompt learning + LoRA", "v5 - v5.13, v7 - v10, v12, v13", "train/train_lora.py + dpd/lora.py", "Word vectors + LoRA matrices (0.4M - 4.7M)"],
    ["Feature generation", "v11 - v11.4", "train/train_feature_gen.py + dpd/feature_gen.py", "Generator, critic, cosine classifier"],
    ["Ensemble", "v6 - v6.4", "train/ensemble.py", "Nothing new (one mixing weight)"],
    ["Evaluation (shared by all)", "all", "dpd/evaluate.py", "-"]], [2000, 1900, 3026, 2100], { size: 18 }),
);

// 3
add(H1("3. CLIP mechanics you need to be able to explain"),
  H2("3.1 How CLIP was pretrained, and why that gives zero-shot ability"),
  P("CLIP was trained by OpenAI on about 400 million (image, caption) pairs from the web. In each training batch the image encoder embeds the images and the text encoder embeds the captions. The loss (a **contrastive loss**) pulls each image embedding towards the embedding of its own caption and pushes it away from the other captions in the batch. After training, an image and a sentence that describe the same thing have a high cosine similarity. That shared space is what lets us classify with sentences."),
  P("Two consequences matter for this project:"),
  ...B([
    "CLIP knows what is common on the web. *Tomato leaf* is common. *Cercospora leaf spot on coffee* is not. So CLIP starts out good at crops and poor at diseases (measured in v1: crop correct 43%, disease correct 8% on test seen).",
    "Only the **final pooled outputs** (the class token for images, the end token for text) were ever compared in the loss. The individual patch outputs were never trained to match text. This explains why patch-level scoring (v7) was noisy."]),
  H2("3.2 The image encoder (ViT-B/16)"),
  ...N([
    "**Patch embedding.** Each 16 x 16 x 3 patch is flattened and multiplied by one learned matrix, giving a vector of 768 numbers. There are 196 of them.",
    "**Class token and positions.** One extra learned vector (the class token, CLS) is put in front, giving 197 vectors. A learned **position embedding** is added to each so the network knows where each patch was.",
    "**12 Transformer blocks.** Each block has two parts. *Self-attention*: every vector looks at every other vector and takes a weighted average of them; the weights (attention) are computed from the vectors themselves by three matrices called query, key and value (q, k, v). *MLP*: a small two-layer network applied to each vector separately (768 to 3072 to 768). Both parts are added back to their input (a residual connection).",
    "**Read-out.** After block 12 only the class token is kept. It is normalised and multiplied by a projection matrix to 512 numbers. That is v."]),
  P("The important point: **the class token is the only thing the classifier sees**. It can only contain what attention moved into it. If attention spends most of its weight on leaf outline and background, a small lesion contributes little to v. LoRA (Section 7) changes exactly the attention and MLP matrices of the later blocks, so it changes what reaches the class token."),
  H2("3.3 The text encoder"),
  ...N([
    "**Tokenising.** The sentence is split into word pieces (tokens) from a fixed vocabulary of 49,408 pieces. A start token and an end token are added. The limit is 77 tokens; our longest prompts stay under it (the code refuses longer ones rather than silently cutting them).",
    "**Token embedding.** Each token is replaced by its row in an embedding table: a vector of 512 numbers. A position embedding is added.",
    "**12 Transformer blocks with a causal mask.** Same block design as the image side, but each token may only attend to tokens on its **left**. So the end token is the only one that has seen the whole sentence.",
    "**Read-out.** The output at the end token is normalised and projected to 512 numbers. That is t."]),
  P("Because of the causal mask, **word order matters** and the end token is a *summary of the whole sentence*. Words do not have fixed vectors after the first layer: each word's vector is mixed with the words before it. This is why a learned word vector only works inside the sentence it was trained in (measured in v5.6)."),
  H2("3.4 From cosine to probability: temperature, softmax, cross-entropy"),
  P("Cosine similarities are small numbers. For this model the score of the right pair is typically around 0.3 and the wrong ones around 0.2. To train, they are turned into probabilities:"),
  ...N([
    "**Scale.** Multiply every cosine by 100 (CLIP's own learned *logit scale*, the inverse of its temperature). 0.30 and 0.25 become 30 and 25.",
    "**Softmax.** probability of class i = exp(score i) / sum over all classes of exp(score). A gap of 5 after scaling means the first class is exp(5), about 150 times, more probable.",
    "**Cross-entropy loss.** loss = -log(probability given to the true class). It is 0 when the model is certain and right, and large when the true class gets a low probability."]),
  KEY("**Why the factor 100 matters in every later section.** A cosine difference of only 0.01 becomes 1 unit after scaling, which is a factor of 2.7 in probability. So a calibration constant of gamma = 0.07 is not small: it is 7 units, a factor of about 1,100. And two prompts whose embeddings differ by 0.01 in cosine are clearly separable to the model."),
  P("The gradient of cross-entropy has a simple form: for each class, *(predicted probability - 1 if it is the true class, else 0)*. So training **pulls the image embedding and the true class's text embedding together** and **pushes apart the image and every wrong class in proportion to how much probability that wrong class received**. Wrong classes the model already scores low are almost ignored; the wrong classes that are *nearly* winning get the strongest push. This one rule explains most results below."),
);

// 4
add(H1("4. Stage A: zero-shot (v1 to v3.9). Only the text changes"),
  P("Nothing is trained in this stage. Both encoders are used as downloaded. The only thing that changes is the sentence written for each pair. This stage answers: *what does CLIP already know, and which kind of text helps?*"),
  ...fig(2, "fig02_prompts", "The prompt for the same pair at each zero-shot step, and the harmonic mean on test."),
  ...exp("v1. Plain CLIP, raw class names", "One sentence per pair using the dataset's raw names: *a photo of {crop} leaves with {disease}.*",
    "HM 5.4%. Seen 6.4%, unseen 4.6%. Chance is 0.3%. Crop correct 43% (seen) but disease correct only 8%.",
    ["CLIP's web training made it good at recognising plant species and poor at disease names. Many names are rare or meaningless to it (*sigatoka*, *esca*, *xamthomonas*). A word the tokenizer does not know is broken into fragments, and the resulting text embedding carries little meaning.",
     "Predictions also collapse onto a few **hub prompts**: one prompt (*soybean bacterial blight*) attracted 99 seen and 129 unseen images. In a high-dimensional space some vectors sit close to many others; without training nothing corrects this."],
    "Measured (crop vs disease accuracy; hub counts)."),
  ...exp("v2. Clean names and five sentence templates", "Typos fixed, Latin pest names replaced by common names (*eriosoma lanigerum* to *woolly apple aphids*), *paddy* to *rice*. Five different sentence templates per pair; the five text embeddings are averaged.",
    "HM 5.6% (+0.2). Top-5 HM 22.2% (+3.8). Seen disease correct 10.5% (from 8.2%).",
    ["Common names are words CLIP met during pretraining, so their embeddings carry meaning. Averaging five templates removes the accident of one particular wording: each template's embedding is the class meaning plus some wording-specific noise, and the noise partly cancels in the mean.",
     "The gain shows in Top-5 rather than Top-1: the right pair moves up the ranking but is still rarely first, because the *name* alone still says little about what the disease looks like."],
    "Measured."),
  ...exp("v3. Symptom descriptions (the knowledge base)", "Each prompt gets a short visual description of the disease after a colon: colour, lesion shape, margins, halos, powder or pustules. One description per disease, about 8 words.",
    "HM 10.2% (from 5.6%). Unseen 14.6% (from 4.6%). *grape + black rot* rose from 8% to 59%.",
    ["The description puts **visual words** into the sentence (*brown*, *concentric rings*, *yellow halo*, *powdery*). CLIP learned these words from millions of captions, so its text encoder maps them near the image features of such patterns. The end-token embedding now encodes *what to look for*, not only a name.",
     "The gain is uneven because it depends on whether the description is distinctive. *Black rot* (tan spots, dark border, tiny black dots) is unlike its neighbours and jumped; *tomato early blight* stayed near 0% because its description is close to many other leaf-spot diseases."],
    "Measured, and confirmed by the control v3.1."),
  ...exp("v3.1. Control: random text instead of descriptions", "The same prompts, but each description is replaced by random letter strings of the same length.",
    "HM 5.5%. The whole v3 gain disappears (back to the v2 level).",
    "This rules out the alternative explanation that *any* extra text helps (longer prompts, more tokens). Only the **meaning** of the description produces the gain. This is the cleanest evidence in the project that the language side contributes real information.",
    "Measured. This is a proper control experiment and worth citing in the report."),
  ...exp("v3.2. Each descriptor scored separately, scores averaged", "The description is split into separate short descriptors; the image is scored against each, and the scores are averaged without re-normalising.",
    "HM 3.8% (worse). But ranking among the 10 unseen pairs alone is as good as v3 (49.0% vs 48.5%).",
    "The mean of several length-1 vectors pointing in different directions is **shorter than 1**. A class with more, or more varied, descriptors gets a shorter mean vector and therefore systematically lower scores. The information is the same; the *scale between classes* is broken. Zero-shot has no training step that could fix a per-class scale error.",
    "Measured (the unseen-only diagnostic isolates ranking from scale)."),
  ...exp("v3.3 to v3.6. Taxonomy text versus symptom text", "v3.3: the dataset's official taxonomic description (pathogen name) instead of symptoms. v3.4 / v3.5: both in one sentence, in either order. v3.6: both as **two separate prompts** whose embeddings are averaged.",
    "v3.3 HM 8.0%; v3.4 8.4%; v3.5 8.5%; **v3.6 11.9%** (best zero-shot; Top-5 HM 38.1%).",
    ["Taxonomy names help seen pairs and ranking; symptoms help unseen pairs. They are complementary. But when both are put into **one sentence** the taxonomy dominates whichever comes first (v3.4 and v3.5 give the same result), because the sentence is compressed into one end-token vector and the rare Latin tokens take most of it.",
     "Encoding them **separately** and averaging the two vectors gives each exactly half the weight by construction. That is why v3.6 keeps the unseen accuracy of v3 and adds the seen accuracy of v3.3."],
    "Measured for the order test and the separate-prompt result. *Why* Latin tokens dominate one sentence is a likely explanation, not tested."),
  ...exp("v3.7 / v3.8. Longer descriptions", "The 8-word descriptions are replaced by 21-word ones.",
    "v3.8 HM 7.9% (from 10.2%); v3.7 10.1% (from 11.9%). Seen improves slightly, unseen halves.",
    "Long descriptions share generic phrases (*leaves yellowing, drying, dropping*). The text embeddings of different diseases become **more alike**, so the one distinctive cue that decided an unseen pair is diluted. Short and distinctive beats long and complete.",
    "Measured result; the 'shared phrases' cause is likely."),
  ...exp("v3.9. BioCLIP backbone", "Same prompts, but the backbone is BioCLIP (CLIP further trained on 10 million biology images with taxonomic captions).",
    "HM 3.3% (from 10.2%). Even crop accuracy fell to 23%.",
    "BioCLIP's **text encoder** was trained on captions made of Latin names and taxonomic ranks. Our prompts are common-name sentences with symptom descriptions, which is text unlike anything it was trained on. A biology-specific *image* encoder does not help if the *text* side cannot read the prompts.",
    "Measured result; cause is likely (consistent with v4.3)."),
  KEY("**What Stage A established.** (1) Symptom text carries real, usable information (v3 vs v3.1). (2) Short, distinctive descriptions transfer best. (3) Without training, scores are not comparable across classes, so absolute accuracy stays near 10%. The next stage fixes (3)."),
);

// 5
add(H1("5. Stage B: prompt learning (v4 to v4.3). Training 118,000 numbers"),
  H2("5.1 The mechanism"),
  P("The idea follows Compositional Soft Prompting (CSP, Nayak et al. 2023). CLIP stays completely frozen. In each prompt, the crop word and the disease word are replaced by **learnable vectors**: one vector of 512 numbers per crop (55) and one per disease (175). That is 230 x 512 = **117,760 trainable numbers**, against roughly 150 million frozen ones."),
  ...fig(3, "fig03_text_encoder", "The learned vectors enter at the token-embedding level. Everything after that is frozen CLIP."),
  P("Step by step, for one training batch of 256 images (v4) or 64 images (v5 on):"),
  ...N([
    "Every prompt is tokenised once. Its token embeddings are stored. At the [CROP] and [DISEASE] positions the stored embedding is overwritten with the current crop vector and disease vector.",
    "The sequence goes through the 12 frozen text blocks. The end-token output is projected and normalised. This is done for the pair prompts, the crop prompts and the disease prompts.",
    "The scores 100 x cos(v, t) are computed and the three cross-entropy losses are added (Figure 5).",
    "Backpropagation computes how the loss would change if each number in the crop and disease vectors changed. The gradient passes **through** the frozen text blocks (they are part of the computation) but only the 230 vectors are updated.",
    "Each vector starts from CLIP's own embedding of the class name (the mean of its token embeddings), so training starts exactly from the zero-shot model and moves away from it gradually."]),
  ...fig(5, "fig05_loss", "The three losses. The image embedding is shared; each head compares it with a different set of prompts."),
  P("**Why three losses and not one.** The pair loss alone would let a crop vector and a disease vector 'co-adapt': the pair *tomato + late blight* could be fitted by a tomato vector that secretly encodes late blight. Then reusing the tomato vector with *early blight* would fail. The crop prompt (*a photo of [CROP] leaves.*) and the disease prompt (*a photo of leaves with [DISEASE]: symptoms.*) each contain only one learned vector, so the crop and disease losses force each vector to be correct **on its own**. That is what makes them reusable."),
  ...fig(4, "fig04_composition", "Composition: the unseen pair's prompt is built from two vectors that were trained by other pairs."),
  P("**Why not train the whole text encoder?** With a median of 7 images per pair, 63 million free text weights would memorise the training pairs. Worse, the frozen encoder is what *combines* a crop word and a disease word into a sentence meaning. If it is retrained on seen pairs only, it can forget how to combine words it never saw together. Keeping it frozen keeps the composition rule that unseen pairs depend on."),
  H2("5.2 The experiments"),
  ...exp("v4. First trained model", "230 learned vectors on frozen CLIP, v3's prompt with symptom description. Learning rate 1e-3, up to 40 epochs. Calibration constant gamma introduced, chosen on validation.",
    "HM **43.6%** (from 10.2%). Seen 30.1%, unseen 79.1%, with gamma 0.045. Without calibration: seen 50.8%, unseen 26.6%, HM 34.9%.",
    ["This is the largest single jump in the project (+33 points) and it comes from **alignment**. Before training, each text embedding sits wherever CLIP's general knowledge put it. Cross-entropy moves every pair's text embedding towards the image embeddings of *this dataset's* photos of that pair and away from the photos of other pairs. That removes the hub prompts and makes scores comparable between classes, which was the zero-shot ceiling.",
     "Unseen pairs gain too (26.6% without calibration, from 14.6%) although they have no images, because their prompts are built from vectors that seen pairs trained.",
     "Selection picked **epoch 3 of 40**. Validation accuracy on held-out pairs peaked in epochs 1 to 3 and then fell from 39% to 15% while seen accuracy kept rising. The vectors were starting to memorise seen *combinations* instead of staying reusable parts: this is overfitting, visible only because the validation split contains held-out pairs."],
    "Measured. The validation curve is direct evidence of the overfitting described."),
  ...exp("v4.1. Validation that mirrors the test, lower learning rate", "Two changes to the validation split: the 10 real unseen pairs are hidden from the validation label space, and each held-out pair is capped at 40 images. Learning rate 1e-3 to 2e-4.",
    "HM **52.9%** (+9.3). Seen 44.3%, unseen 65.7%, gamma 0.0375, epoch 18.",
    ["In v4 the validation ranked images against 30 untrained pairs (20 held-out + the 10 real unseen) while the test has only 10. More untrained competitors need a larger gamma to win, so validation over-estimated gamma, and on test 53% of *seen* images were pushed to unseen pairs. With the split corrected, gamma dropped and the seen/unseen balance moved from 30/79 to 44/66. The harmonic mean rewards balance, so it rose.",
     "One held-out pair (*cassava mosaic*) was 544 of 1,340 validation images and so decided the validation score. The cap removes that.",
     "The smaller learning rate makes each update a fifth of the size. The validation curve became smooth (44 to 46% from epoch 5 on, instead of swinging between 22% and 36%), so the selected epoch is no longer a lucky spike."],
    "Measured result. **Caveat for Q&A:** two things changed at once (the split and the learning rate), so the +9.3 cannot be divided between them."),
  ...exp("v4.2. Three validation splits", "Stage 1 is repeated on 3 different validation splits; epoch and gamma are chosen on the mean.",
    "HM 51.8% (no gain). The three splits wanted gamma 0.0425, 0.0275 and 0.0375.",
    "The splits disagree, which confirms that one split is a noisy sample (20 pairs is small). But the average lands on the same gamma as v4.1, so the remaining calibration error is **systematic**, not random: held-out seen pairs seem to need a larger gamma than the real unseen pairs.",
    "Measured."),
  ...exp("v4.3. BioCLIP with the same recipe", "v4.2 with the BioCLIP backbone.",
    "HM 48.6% (v4.2: 51.8%). Seen is similar (43.7% vs 41.2%), unseen much lower (54.7% vs 69.7%).",
    "BioCLIP's image features fit the seen pairs equally well. What fails is **composing a new pair from two learned vectors**, which is done by the text encoder. This matches v3.9 and locates the weakness in BioCLIP's text side.",
    "Measured result; the localisation to the text encoder is supported by two experiments."),
);

// 6
add(H1("6. Calibration and model selection"),
  H2("6.1 Why trained models are biased towards seen pairs"),
  P("During training the pair loss is a softmax over the **seen pairs only** (an unseen pair has no image, so it can never be the true class). Two things follow from the gradient rule in Section 3.4:"),
  ...B([
    "Seen pair prompts are repeatedly **pulled towards** real image embeddings. Unseen pair prompts are never pulled towards anything.",
    "The model is never penalised for giving a seen pair a high score on an image that actually belongs to an unseen pair, because it never sees such an image."]),
  P("The result: for *any* image, the best seen-pair score tends to be higher than the best unseen-pair score. Without correction, v5.5 gets 67.6% on seen images but only 28.7% on unseen images."),
  H2("6.2 The fix: calibrated stacking"),
  ...fig(7, "fig07_calibration", "Calibration on one example and on the whole test set (v5.5). The bar values in the top half are illustrative; the table is measured."),
  P("One constant, **gamma**, is subtracted from every seen pair's score before taking the maximum (Chao et al. 2016). It does not change the order *among* seen pairs or *among* unseen pairs. It only moves the border between the two groups. A larger gamma moves more images to unseen pairs: unseen accuracy rises, seen accuracy falls. The harmonic mean peaks where the two are balanced."),
  P("Gamma is a **hyperparameter**, so it must be chosen without the test set. That needs a validation set that contains 'unseen' pairs, which is the next subsection."),
  H2("6.3 Choosing epoch and gamma without touching the test set"),
  ...fig(8, "fig08_validation", "The two-stage procedure used by every trained VLM run."),
  ...B([
    "**Pseudo-unseen pairs.** 20 seen pairs are removed from training completely, chosen so that their crop and their disease each still appear in other training pairs. To the model they are exactly like real unseen pairs.",
    "**Val-seen.** 15% of the images of the remaining pairs are held out (every pair keeps at least one training image).",
    "**Stage 1.** Train on the rest. After every epoch, score the validation images and compute the harmonic mean for each gamma on a grid (0 to 0.1 in steps of 0.0025). Keep the (epoch, gamma) with the best value.",
    "**Stage 2.** Train a *fresh* model on all 8,000 images for that number of epochs, apply that gamma, and evaluate on test once."]),
  P("**Why refit in stage 2?** The stage-1 model never saw the 20 held-out pairs or the validation images. Throwing away that data would hurt the tail pairs most (many have under 5 images)."),
  H2("6.4 How reliable is this validation? (be ready for this question)"),
  ...B([
    "**It is noisy.** The same v5.5 recipe scores 56.5%, 62.1% and 60.7% on three different splits: a range of about 6 points from the choice of held-out pairs alone.",
    "**Decision rule used.** Keep a change only if validation HM improves by more than 2 points. From v5.8 on, compare on the mean of 3 splits.",
    "**Single-split validation did not rank LoRA variants.** Across the 16 single-split LoRA runs, validation HM and test HM have a correlation of 0.05. Some variants rejected on validation (v9, v7.1, v13.1) are among the highest on test. This does *not* mean they should have been kept; it means differences of 1 to 3 points between LoRA variants cannot be resolved with one split.",
    "**Test numbers have their own error bar.** A bootstrap over test images gives about +/- 1.7 points on test HM for one model.",
    "**A known mismatch.** In stage 2 the learning-rate schedule (cosine decay) ends at the selected epoch, while in stage 1 it ended at the maximum epoch. So the refitted model is trained with a faster decay than the validated one. A config option (lr_schedule_epochs) now fixes this, but all reported runs used the old behaviour."]),
);

// 7
add(H1("7. Stage C: LoRA on the image encoder (v5 and its ablations)"),
  H2("7.1 Why the image side had to change"),
  P("With CLIP frozen, the image embedding v of a photo is fixed. Prompt learning can move the 333 text embeddings around, but if two photos (*tomato late blight* and *tomato early blight*) have nearly the same v, **no arrangement of text embeddings can separate them**. That was the situation after v4: seen disease accuracy 49%, against 63% for the ResNet-50 CNN baseline whose whole image network is trained. The bottleneck was the frozen image encoder."),
  P("Fully fine-tuning 86 million image weights on 8,000 images would overfit and would move the image embeddings away from the space the text encoder was aligned with. LoRA is the middle path."),
  H2("7.2 The mechanism"),
  ...fig(6, "fig06_lora", "Left: which blocks are adapted. Right: what happens inside one adapted weight matrix."),
  P("A weight matrix W in the network multiplies its input x. LoRA (Low-Rank Adaptation, Hu et al. 2022) leaves W frozen and adds a trained side path:"),
  KEY("**output = W x + (alpha / r) x B (A x)**.  A has r rows (it squeezes the 768-number input down to r numbers). B has r columns (it expands those r numbers back). r is the **rank**. With r = 16 and alpha = 32 the factor alpha / r is 2."),
  ...B([
    "**Low rank means few directions.** The update B A is a 768 x 768 matrix, but it is built from only 16 'input directions' and 16 'output directions'. It can change how the layer reacts to 16 patterns and nothing else. This is a strong built-in limit on overfitting.",
    "**B starts at zero, A starts random.** So at step 0 the update is exactly zero and the model is exactly pretrained CLIP. Training moves away from CLIP smoothly. (If both started at zero, the gradient to each would be zero and nothing would ever train.)",
    "**Which matrices.** In each adapted block: the attention q/k/v matrix, the attention output matrix, and both MLP matrices. Adapting attention changes *where the class token looks*. Adapting the MLP changes *what features are computed* from what it sees.",
    "**Which blocks.** The last 4 (v5) or last 8 (v5.3 on) of 12. Early blocks detect generic edges and textures that transfer well. Later blocks build task-specific meaning. Also, blocks before the first adapter need no gradient, which saves memory on a 6 GB GPU.",
    "**How many numbers.** Per block at rank 8: 98,304. v5: 4 blocks = **393k**. v5.3: 8 blocks at rank 16 = **1.57M**. v12: 12 blocks at rank 32 = **4.72M**. The encoder itself has 86M."]),
  P("**A side effect to mention.** In Stage B the image embeddings were computed once and cached, because the encoder was frozen. With LoRA they change every step, so every image is re-encoded every epoch. That makes **data augmentation** possible for the first time (random crops of 50 to 100% of the image, flips, mild brightness / contrast / saturation jitter; no hue jitter because colour is diagnostic). It also makes one run take about 45 minutes instead of seconds."),
  H2("7.3 The experiments"),
  ...exp("v5. LoRA in the last 4 blocks, rank 8", "v4.2's prompt model + LoRA (393k numbers) trained jointly. Images re-encoded each epoch with augmentation. One validation split, 15 epochs.",
    "HM **65.1%** (+12.2 over v4.1). Seen 53.4% (from 44.3%), unseen 83.2% (from 65.7%). First VLM above the ResNet-50 baseline (59.7%).",
    ["**Seen side:** seen disease accuracy rose from 49% to 58%. The adapted attention and MLP layers produce image embeddings in which diseases of the same crop are further apart, so the prompt vectors can separate them. Uncalibrated seen accuracy (67.1%) now equals the CNN baseline (67.2%).",
     "**Unseen side:** the gain came mainly from better *crop* recognition (unseen crop correct 86.6%) and cleaner seen/unseen separation, not from better disease transfer: unseen disease accuracy without calibration stayed at 21% (v4.1: 23%). With 10 unseen pairs over 9 crops, sending an image to 'unseen' and getting the crop right almost decides the pair."],
    "Measured. **Caveat:** LoRA and augmentation were introduced together, so their effects are not separated."),
  ...exp("v5.1. 336-pixel input", "Input 224 to 336 pixels (441 patches instead of 196). Everything else as v5.",
    "Validation 51.0% vs 51.2% (-0.2). Not kept.",
    "More patches should show lesions in more detail. But CLIP's position embeddings were learned for a 14 x 14 grid; for 21 x 21 they are interpolated, so the encoder sees patch positions it was never trained on, and each patch now covers a smaller part of the leaf than in pretraining. The possible gain in detail was cancelled.",
    "Measured result; the cause is likely."),
  ...exp("v5.2. Test-time augmentation (4 flips)", "At prediction, the embeddings of the image and its three flipped versions are averaged.",
    "Validation +0.4. Not kept.",
    "Averaging helps only if the views disagree. The embedding of a flipped leaf has cosine about 0.99 with the original: the model is already almost flip-invariant (flips were in the training augmentation), so the average adds no new information. It costs 4x inference time.",
    "Measured (the 0.99 cosine was measured)."),
  ...exp("v5.3 = v5.5. More LoRA capacity (8 blocks, rank 16)", "Last 8 blocks instead of 4, rank 16 instead of 8 (alpha doubled so the factor alpha / r stays 2). 1.57M numbers.",
    "Validation **56.9% vs 51.2% (+5.7)**: the only single change that passed the 2-point rule. Test HM 64.5% vs 65.1% (no gain). Without calibration: unseen 28.7% (from 16.2%), HM 40.3% (from 26.1%).",
    ["Capacity lets the image encoder change more: more blocks reshape the features earlier, higher rank allows more independent directions. It learns faster (best epoch 6 instead of 13) and then overfits sooner.",
     "The clear effect is **before** calibration: less seen bias and the best disease transfer to new crops of any model (32% uncalibrated unseen disease accuracy). After calibration the harmonic mean is the same as v5, because gamma was already compensating for the bias."],
    "Measured. Report honestly that the validation gain did not appear in test HM."),
  ...exp("v5.4. Class-balanced sampling", "Images drawn with probability proportional to 1 / sqrt(pair size), so rare pairs appear more often.",
    "Validation -2.2. Not kept.",
    "Pairs with 5 or fewer images were drawn about 4 times more often (16.7% of draws instead of 4.5%). The model sees the same few photos repeatedly and memorises them, while the common pairs, which make up most test images, get fewer updates.",
    "Measured result and sampling rates; the cause is likely."),
  ...exp("v5.6 / v5.7. Two-sentence prompts (v3.6's idea) in the trained model", "v5.6: v5.5's trained vectors scored with v3.6's two-sentence text, no retraining. v5.7: trained with the two sentences.",
    "v5.6 test HM 61.5% (worse than 64.5%). v5.7 validation 55.6% vs 56.9% (-1.3). Not kept.",
    ["v5.6 shows that a learned vector is only valid **in the sentence it was trained in**. The text encoder mixes each word with its context (Section 3.3), so the same vector in a different sentence produces a different, untrained embedding.",
     "v5.7 shows the zero-shot benefit of taxonomy text does not survive training. Once vectors and LoRA are fitted to the images, they already carry what the taxonomy sentence added."],
    "Measured."),
  ...exp("v5.8. The same model on three validation splits", "v5.5 with 3 splits. Not a new method: a measurement of validation noise.",
    "Mean validation 59.75%; per split 56.5 / 62.1 / 60.7. Test 64.2%.",
    "Which 20 pairs are held out changes the validation score by about 6 points. This is larger than the effect of any later change, and it is the reason all later comparisons use the same three splits.",
    "Measured."),
  P("v5.9 to v5.13 (alternating updates, multiple descriptions, separate crop / disease image branches, learned context words) are configured but **not yet run**. Do not report results for them."),
);

// 8
add(H1("8. Stage D: attempts beyond the base recipe (v7 to v13)"),
  P("After v5.5 the diagnosis was: **70 to 75% of seen disease errors have the crop right**. The model knows it is a tomato leaf and picks the wrong tomato disease. The experiments in this section each attack that from a different angle. **None passed the 2-point rule.** They are still worth explaining, because each tests a specific idea about how the network works."),
  H2("8.1 Looking at lesions: v7, v7.1, v7.2, v13"),
  ...fig(12, "fig12_lesion", "Two of the lesion-level ideas."),
  ...exp("v7. Patch-level disease score", "Each disease text embedding is also compared with all 196 patch outputs; the mean of the 8 best patches is added to the pair score.",
    "Validation +1.5 (inside noise). Test 62.2% (lower than 64.5%). Not kept.",
    "A lesion occupies a few patches, so matching text against patches should find it. But CLIP's pretraining only ever compared the **class token** with text (Section 3.1). Patch outputs were never trained to line up with text embeddings, so their similarities to disease text are noisy.",
    "Measured result; the cause is likely and well known for CLIP."),
  ...exp("v7.1. Closer crops in augmentation", "Random crops cover 25 to 100% of the image instead of 50 to 100%.",
    "Validation -1.6. Test 66.4% (higher). Not kept (decided on validation).",
    "Close-ups make lesions fill more of the frame, which should help disease recognition; but they can also cut the lesion out, or remove the leaf outline needed for the crop. Validation and test disagree in sign, which is what noise looks like.",
    "Inconclusive."),
  ...exp("v7.2. Extra loss against same-crop pairs only", "An additional cross-entropy that ranks the true pair only against other pairs of the same crop.",
    "Validation -0.5, test unchanged. No effect.",
    "The ordinary pair softmax **already contains** those same-crop pairs as wrong classes, and by the gradient rule they are the ones that get the strongest push because they score highest. The extra loss re-weights a signal that is already there.",
    "Measured result; the explanation follows directly from the loss."),
  ...exp("v13 / v13.1. Local-to-global consistency", "Each training image also gets a small crop (10 to 35% of the image). An extra loss makes the small crop's embedding match the whole image's embedding (which is treated as a fixed target). No labels used.",
    "v13 (weight 5) validation -0.4; v13.1 (weight 20) validation -1.2. Test 66.1% and 68.3%. Not kept.",
    ["To recognise the whole image from a small region, the encoder must encode local detail. In v13 the extra loss actually *rose* during training (0.069 to 0.080): at weight 5 it was too weak to compete with the supervised losses, so the idea was not really tested. At weight 20 it held (0.042 to 0.049), yet validation did not improve.",
     "Crop and disease accuracy rose by similar amounts, so there is no sign of a lesion-specific effect."],
    "Measured. v13.1 has the highest single-VLM test HM, but it was rejected on validation; do not present it as a selected model."),
  H2("8.2 Changing the text again: v8"),
  ...exp("v8. Contrastive descriptions", "33 diseases whose descriptions were near-duplicates in CLIP's text space (cosine above 0.92) were rewritten to lead with the distinguishing cue.",
    "Validation -0.1. No effect.",
    "In the trained model each disease has its own learned vector in the prompt, and the image encoder has been adapted. Those trained parts already separate the diseases; a small change in the fixed description text shifts the starting point but training absorbs it. Text wording matters a lot before training (Stage A) and little after.",
    "Measured result; cause is likely. **Missing control:** no trained run with random or no descriptions exists, so 'descriptions help the trained model' is not proven."),
  H2("8.3 Using what was already trained: v9"),
  ...exp("v9. Fusing crop and disease scores at prediction; scaling the LoRA update", "(1) Pair score = (pair + a x crop score + b x disease score) / (1 + a + b). (2) LoRA update multiplied by s between 0 and 1 (s = 0 is frozen CLIP). a, b, s chosen on validation.",
    "Validation +1.6, 95% interval [-0.2, +3.6]. Chose a = 4 (the edge of the grid), b = 1, s = 1. Test 67.7%. Not kept.",
    ["The crop and disease heads are trained every step but ignored at prediction. Adding them back gives each pair extra evidence from its parts, which is exactly the evidence that transfers to unseen pairs. The large crop weight says the crop head is the most reliable signal.",
     "Scaling LoRA down (moving back towards frozen CLIP) never helped: the adapted features are better than a blend."],
    "Measured, with a paired bootstrap interval that includes zero. The grid edge means the crop weight was not fully explored."),
  H2("8.4 Staying close to CLIP: v10, v10.1, v10.2, v10.3"),
  P("Fine-tuning can drift away from the pretrained space that unseen pairs rely on. Three regularisers were tested, one at a time."),
  ...exp("v10. Image consistency", "Extra loss: keep the LoRA image embedding close (L1 distance) to frozen CLIP's embedding of the same image. Weight 10.",
    "Validation +0.5. Test: seen 51.8%, unseen 85.6%. Not kept.",
    "It works as designed: pulling image features back towards CLIP trades seen accuracy for unseen accuracy. The harmonic mean does not gain from moving along that trade-off.",
    "Measured."),
  ...exp("v10.1 / v10.3. Text anchor", "Extra loss: keep the learned pair, crop and disease text embeddings close (L1) to their values at the start of training. Weight 25.",
    "One split: +1.4. **Three splits (v10.3): 61.15% vs 59.75% (+1.4), higher on every split.** Test 66.3% (seen 55.1%, unseen 83.3%).",
    "The starting text embeddings are CLIP's own name embeddings, which is what unseen pairs' prompts are made from. Without an anchor the vectors can drift to fit seen combinations (the overfitting seen in v4). The anchor lets them move, but penalises moving far, so they stay reusable.",
    "Measured. Below the 2-point rule, but the most consistent gain after v5.3, so v10.3 was used as the CLIP member of the final ensemble."),
  ...exp("v10.2. Weight averaging (EMA)", "Validation and the final model use a running average of the trained weights instead of the latest values.",
    "Validation +0.4. Not kept.",
    "Averaging over recent steps smooths out step-to-step noise. The validation curve did become smoother and peaked later, but its peak was no higher.",
    "Measured."),
  ...exp("v12. LoRA in all 12 blocks, rank 32", "v10.3 with 4.72M LoRA numbers instead of 1.57M.",
    "Validation +0.65 (one split only). Test 67.9%: seen 58.6% (+3.5), unseen 80.9% (-2.4).",
    "Same pattern as v5.3: more capacity buys seen accuracy (validation seen rose to 70 to 75%) but not transfer to held-out pairs, and the gamma needed keeps rising, which is the sign of growing seen bias.",
    "Measured."),
);

// 9
add(H1("9. Feature generation (v11 to v11.4)"),
  ...fig(9, "fig09_feature_gen", "Generating image embeddings for pairs that have no images."),
  P("This branch attacks the seen bias at its source. The bias exists because the classifier never trains on an unseen-pair example. Feature generation (f-CLSWGAN, Xian et al. 2018) **manufactures** such examples."),
  ...N([
    "A small **generator** network takes a pair's text embedding plus random noise and outputs a fake image embedding (512 numbers).",
    "A **critic** network scores how real an embedding looks for that pair. Generator and critic are trained against each other (a Wasserstein GAN with gradient penalty) on the seen pairs, where real embeddings exist. An extra classification loss keeps each fake embedding recognisable as its pair.",
    "The trained generator is given the **unseen** pairs' text embeddings and produces 300 fake embeddings for each.",
    "A cosine classifier over all 333 pairs is trained on real seen embeddings plus fake unseen ones."]),
  P("The noise matters: without it the generator would output one point per pair. With it, it outputs a *cloud*, imitating the natural spread of real photos."),
  ...table(["Version", "What changed", "Validation HM (3 splits)", "Test HM (seen / unseen)", "Reading"], [
    ["v11", "Generator on frozen CLIP features", "49.25 (v4.2: 46.57)", "57.8 (55.9 / 59.8)", "Kept over v4.2. Gain is on seen pairs; the classifier is far more balanced."],
    ["v11.2", "Same classifier, **no** generated features", "37.03", "41.9 (48.2 / 37.1)", "Control. The 12-point drop is what the generated features add."],
    ["v11.1", "Generator also conditioned on crop and disease text", "50.75", "59.7 (55.7 / 64.2)", "+1.5: the unseen condition reuses parts seen with other partners."],
    ["v11.3", "v11 on v10.3's LoRA features", "60.24", "64.9 (60.9 / 69.5)", "+11 from better features, but not above v10.3 itself (61.15)."],
    ["v11.4", "v11.1 on LoRA features", "61.85", "66.1 (62.1 / 70.8)", "Best single-model validation; +0.7 over v10.3, below the rule."]], [900, 2300, 1500, 1700, 2626], { size: 18 }),
  P("**Mechanism of the result.** v11.4 and v10.3 reach the same test HM (66.1 vs 66.3) with very different balance: 62 / 71 against 55 / 83. Generated features give the classifier real competition from unseen pairs during training, so fewer seen images are lost at calibration (18% of seen images moved to unseen pairs, against 27% for v10.3). But the fake embeddings are only as good as the generator's guess, so unseen accuracy itself is lower. The two approaches fail in different ways, which makes v11.4 a natural third ensemble member (not yet tried)."),
);

// 10
add(H1("10. The ensemble (v6 to v6.4) and where the final number comes from"),
  ...fig(10, "fig10_ensemble", "The final model: v10.3 and an independent ConvNeXt CNN combined at score level."),
  H2("10.1 The mechanism"),
  P("An ensemble averages the outputs of models that were trained differently. It helps when the models make **different mistakes**: where one is wrong and unsure, the other is often right and sure, and the average follows the confident one."),
  P("The two members here are as different as possible:"),
  ...B([
    "**CLIP + LoRA (v10.3).** Classes are sentences; one image encoder; knowledge from 400M web image-text pairs.",
    "**Independent CNN (ConvNeXt-Base).** Two separate networks, one predicting the crop (55 outputs) and one the disease (175 outputs), pretrained on ImageNet-22k. A pair's score is log P(crop) + log P(disease). An unseen pair gets a score because its crop output and its disease output were each trained."]),
  P("**Putting them on one scale.** CLIP gives cosines around 0.3; the CNN gives sums of log-probabilities around -5. Averaging those directly would be meaningless. Both are converted to **log-probabilities over the same 333 pairs** (CLIP: multiply by 100 then log-softmax; CNN: log-softmax of its pair scores). Then: final = w x CLIP + (1 - w) x CNN. The weight w and gamma are chosen on validation."),
  ...table(["Version", "Members", "w on CLIP", "Validation HM", "Test HM", "Seen / unseen"], [
    ["v6", "v5 + ResNet-50 CNN", "0.4", "53.1", "70.1", "59.8 / 84.8"],
    ["v6.1", "v5.5 + ResNet-50 CNN", "0.5", "58.9", "70.8", "60.7 / 84.9"],
    ["v6.2", "v5.7 + ResNet-50 CNN", "0.5", "58.2", "70.5", "59.8 / 85.8"],
    ["**v6.3**", "**v10.3 + ConvNeXt-Base CNN**", "0.3", "66.0", "**75.0**", "64.3 / 90.1"],
    ["v6.4", "v10.3 + ConvNeXt-Base 336 px", "0.2", "67.6", "74.6", "64.2 / 89.1"]], [900, 3000, 1000, 1400, 1100, 1626], { size: 18 }),
  H2("10.2 Why it improves by this much"),
  ...B([
    "**Members alone on test:** v10.3 66.3%, ConvNeXt 68.6%. **Together: 75.0%.** That is +6.4 over the better member.",
    "**Seen side.** Seen accuracy with no unseen competition is 67.5% for v10.3 and 74.2% for ConvNeXt; the ensemble reaches 74.6%. The CNN is the stronger seen-pair classifier and the ensemble keeps that.",
    "**Unseen side.** Among the 10 unseen pairs alone: v10.3 86.8%, ConvNeXt 91.2%, ensemble 95.1%. Here the two genuinely complement each other: the ensemble beats both.",
    "**Fewer seen images lost to calibration.** 27% of seen images are moved to an unseen pair for v10.3, 19% for ConvNeXt, 21% for the ensemble.",
    "v6.2 shows the limit: swapping the CLIP member for one with different prompts changed nothing. Ensembling gains come from members that differ in *kind*, not in detail."]),
  H2("10.3 An honest framing for the report"),
  KEY("**No single VLM beats the ConvNeXt independent CNN on test** (best single VLM 68.3%, ConvNeXt 68.6%; these are within each other's error bars). Against the *required* ResNet-50 baselines (59.7% independent, 56.7% multi-task) every LoRA model is clearly better. The defensible claim is: *the VLM is competitive alone and strongly complementary to a CNN; the combination is the proposed approach.*"),
  P("Per-source accuracy is a ready-made 'field robustness' result. For v10.3 on seen test images without calibration: 86% on the largest source (CDDM, close-up photos), 30% on PWV2 (in-the-wild photos, 233 images), 0 of 36 on RDD1 (rice; healthy leaves predicted as brown spot or blast). 24% of the unseen test set is from PWV2. The ConvNeXt CNN shows the same pattern (about 43% on PWV2), so this is a property of the data, not of the VLM."),
);

// 11
add(H1("11. What actually drove the improvement"),
  ...fig(11, "fig11_progress", "The path from 5.4% to 75.0% on the test sets."),
  ...table(["Lever", "Step", "Test HM gain", "Mechanism in one line", "Evidence"], [
    ["Learned crop and disease vectors", "v3 to v4", "+33.4", "Cross-entropy aligns every pair's text embedding with this dataset's image embeddings; shared vectors carry that to unseen pairs.", "Measured"],
    ["LoRA on the image encoder (+ augmentation)", "v4.1 to v5", "+12.2", "Lets the image embedding separate diseases within a crop; fixes the frozen-image bottleneck.", "Measured (two changes together)"],
    ["Validation that mirrors the test (+ lower learning rate)", "v4 to v4.1", "+9.3", "A correct gamma balances seen and unseen; the harmonic mean rewards balance.", "Measured (two changes together)"],
    ["Ensemble with a CNN", "v10.3 to v6.3", "+8.7 (+6.4 over ConvNeXt alone)", "Two different kinds of model make different mistakes; averaging log-probabilities follows the confident one.", "Measured"],
    ["Symptom descriptions", "v2 to v3", "+4.6", "Visual words in the prompt tell the text encoder what the disease looks like.", "Measured, with a control (v3.1)"],
    ["Calibration (gamma)", "inside every trained model", "+24 for v5.5 (40.3 to 64.5)", "Removes the score advantage that seen pairs get from training.", "Measured"]], [2000, 1100, 1300, 3300, 1326], { size: 18 }),
  P("**What did not move the result** (all within validation noise): higher resolution, test-time augmentation, class-balanced sampling, patch-level scoring, closer crops, same-crop negatives, rewritten descriptions, image-consistency loss, weight averaging, more LoRA capacity beyond 8 blocks, the self-supervised local-global loss. The common thread: **after v5.5 every single-VLM variant lands between 64.5% and 68.3% on test, a band about as wide as the measurement noise.**"),
  H2("11.1 Where the remaining error is"),
  ...B([
    "**Seen versus unseen competition costs about 9 points.** If seen images were ranked only among seen pairs and unseen images only among unseen pairs, v10.3 would reach HM 76.0% instead of 66.3%. Choosing gamma perfectly on test would only give 67.9%. So the loss is not a badly chosen gamma: one constant cannot separate the two groups image by image.",
    "**Calibration moves about a quarter of seen images to unseen pairs,** and roughly half of those were correct before calibration.",
    "**Rare pairs.** Seen pairs with 5 or fewer training images (207 test images) are right 10 to 24% of the time after calibration, against 62 to 67% for pairs with more than 50 images.",
    "**The hardest unseen pair is the largest.** *tomato + early blight* is 76% correct (v10.3). Its disease vector was trained by only 102 images, from two other crops."]),
);

// 12
add(H1("12. Likely questions and short answers"),
  ...[
    ["What is a vision-language model and why use one here?", "Two encoders, one for images and one for text, trained so that matching images and sentences get similar vectors. A class is a sentence, not an output unit, so a class with no training images still has a meaningful class vector. That is exactly the unseen-pair problem."],
    ["How does CLIP classify with no training?", "Write one sentence per class, embed all sentences, embed the image, pick the sentence with the highest cosine similarity."],
    ["What exactly did you train?", "Stage B: 230 word vectors (118k numbers), CLIP frozen. Stage C: those plus LoRA matrices in the last 8 image blocks (1.57M numbers). CLIP's own 150M weights never change."],
    ["Why does a symptom description help?", "It adds visual words the text encoder already links to image patterns. With random text of the same length the gain disappears (v3.1), so it is the meaning, not the length."],
    ["How can gradients update the vectors if the text encoder is frozen?", "Frozen means its weights are not updated. It is still part of the computation, so the chain rule passes the gradient through it to the input vectors, which are the trainable part."],
    ["Why can the model score a pair it never saw?", "The pair's prompt is built from a crop vector and a disease vector that were each trained by other pairs, combined by the frozen text encoder (Figure 4)."],
    ["Why three losses?", "The crop and disease losses force each vector to be right on its own, so it can be reused with a new partner."],
    ["What is LoRA and why is B zero at the start?", "A frozen weight W gets a trained low-rank addition B A. B = 0 makes the addition zero at step 0, so training starts exactly from pretrained CLIP. A is random so that B receives a non-zero gradient."],
    ["Why low rank?", "A rank-16 update can only change the layer along 16 directions: enough to adapt, too little to memorise 8,000 images."],
    ["Why is the model biased to seen pairs, and what is gamma?", "Only seen pairs are ever the correct answer in training, so only their prompts get pulled towards images. Gamma is a constant subtracted from seen-pair scores to cancel that advantage. It is chosen on validation."],
    ["Is choosing gamma cheating?", "No. It is chosen on held-out *training* pairs that imitate unseen pairs. The test sets are used once, after everything is fixed."],
    ["Why harmonic mean and not the average?", "It punishes imbalance. 70% seen with 5% unseen gives 9%, so a model cannot score well by ignoring unseen pairs."],
    ["Why did the 336-pixel input not help?", "Position embeddings were learned for a 14 x 14 grid and must be interpolated for 21 x 21, so the encoder works outside what it was trained on. (Likely explanation.)"],
    ["Why did the ensemble help so much?", "A sentence-based model and a two-head CNN make different mistakes. Averaged on a common log-probability scale, the combined model is right more often than either: unseen-only accuracy 95% against 87% and 91%."],
    ["What is the weakest part of your method?", "Validation noise. Differences of 1 to 3 points between LoRA variants could not be resolved, and on its own the VLM does not beat a strong CNN. We state this and present the ensemble as the method."],
    ["What would you do next?", "Try other CLIP checkpoints (only OpenAI CLIP and BioCLIP were tried), a three-member ensemble with the feature-generation model, and a crop-conditional calibration instead of one global gamma."],
  ].flatMap(([q, a]) => [P(`**Q. ${q}**`, { keepNext: true, spacing: { before: 120, after: 40 } }), P(a)]),
);

// 13
add(H1("13. Full results table"),
  P("Test numbers with the gamma chosen on validation. 'Val' is validation HM (mean over 3 splits where marked *). Zero-shot runs have no validation. All in percent, Top-1."),
  ...table(["Version", "Change in one line", "Val", "Test HM", "Seen", "Unseen", "Kept?"], [
    ["v1", "Zero-shot, raw names", "-", "5.4", "6.4", "4.6", "reference"],
    ["v2", "Clean names, 5 templates", "-", "5.6", "7.2", "4.6", "yes"],
    ["v3", "Symptom descriptions", "-", "10.2", "7.8", "14.6", "yes"],
    ["v3.1", "Random text control", "-", "5.5", "4.4", "7.4", "control"],
    ["v3.2", "Descriptors scored separately", "-", "3.8", "7.2", "2.6", "no"],
    ["v3.3", "Taxonomy text only", "-", "8.0", "9.8", "6.7", "no"],
    ["v3.4 / v3.5", "Taxonomy + symptoms, one sentence", "-", "8.4 / 8.5", "9.5", "7.6", "no"],
    ["v3.6", "Taxonomy + symptoms, two prompts", "-", "11.9", "10.3", "14.1", "best zero-shot"],
    ["v3.7 / v3.8", "Longer descriptions", "-", "10.1 / 7.9", "12.0 / 8.9", "8.7 / 7.1", "no"],
    ["v3.9", "BioCLIP, zero-shot", "-", "3.3", "3.1", "3.5", "no"],
    ["v4", "Learned crop / disease vectors", "35.8", "43.6", "30.1", "79.1", "yes"],
    ["v4.1", "Validation fix, lr 2e-4", "45.8", "52.9", "44.3", "65.7", "yes"],
    ["v4.2", "3 validation splits", "46.6*", "51.8", "41.2", "69.7", "procedure"],
    ["v4.3", "BioCLIP, trained", "48.1*", "48.6", "43.7", "54.7", "no"],
    ["v5", "LoRA, 4 blocks, rank 8", "51.2", "65.1", "53.4", "83.2", "yes"],
    ["v5.1", "336 px", "51.0", "63.7", "52.7", "80.7", "no"],
    ["v5.2", "Flip TTA", "51.6", "63.9", "51.1", "85.5", "no"],
    ["v5.3 = v5.5", "LoRA, 8 blocks, rank 16", "56.9", "64.5", "52.2", "84.3", "yes"],
    ["v5.4", "Class-balanced sampling", "49.0", "62.3", "49.6", "83.8", "no"],
    ["v5.7", "Two-sentence prompts, trained", "55.6", "66.5", "56.7", "80.3", "no"],
    ["v5.8", "v5.5 on 3 splits", "59.75*", "64.2", "52.1", "83.5", "reference"],
    ["v7", "Patch-level disease score", "58.4", "62.2", "49.6", "83.4", "no"],
    ["v7.1", "Closer crops", "55.3", "66.4", "56.2", "81.3", "no"],
    ["v7.2", "Same-crop negatives", "56.4", "64.6", "52.3", "84.6", "no"],
    ["v8", "Contrastive descriptions", "56.8", "66.4", "55.6", "82.5", "no"],
    ["v9", "Crop / disease score fusion", "58.3", "67.7", "56.6", "84.3", "no"],
    ["v10", "Image consistency loss", "57.4", "64.5", "51.8", "85.6", "no"],
    ["v10.1", "Text anchor", "58.3", "65.4", "54.3", "82.3", "see v10.3"],
    ["v10.2", "Weight averaging (EMA)", "57.3", "64.6", "53.0", "82.9", "no"],
    ["v10.3", "Text anchor, 3 splits", "61.15*", "66.3", "55.1", "83.3", "ensemble member"],
    ["v11", "Feature generation, frozen features", "49.25*", "57.8", "55.9", "59.8", "yes (vs v4.2)"],
    ["v11.2", "No generation (control)", "37.0*", "41.9", "48.2", "37.1", "control"],
    ["v11.4", "Generation on LoRA features", "61.85*", "66.1", "62.1", "70.8", "no (+0.7)"],
    ["v12", "LoRA, 12 blocks, rank 32", "61.8*", "67.9", "58.6", "80.9", "no"],
    ["v13 / v13.1", "Local-global consistency", "56.5 / 55.7", "66.1 / 68.3", "53.4 / 58.5", "87.0 / 82.2", "no"],
    ["v6", "v5 + ResNet-50 CNN", "53.1", "70.1", "59.8", "84.8", "yes"],
    ["v6.1", "v5.5 + ResNet-50 CNN", "58.9", "70.8", "60.7", "84.9", "yes"],
    ["**v6.3**", "**v10.3 + ConvNeXt CNN**", "66.0", "**75.0**", "64.3", "90.1", "**final**"],
    ["v6.4", "v10.3 + ConvNeXt 336 px", "67.6", "74.6", "64.2", "89.1", "equal"],
    ["Baseline", "Independent CNN, ResNet-50", "-", "59.7", "54.8", "65.5", "required"],
    ["Baseline", "Multi-task CNN, ResNet-50", "-", "56.7", "50.6", "64.5", "required"],
    ["Reference", "Independent CNN, ConvNeXt-Base", "-", "68.6", "64.5", "73.3", "ensemble member"]],
    [1150, 3000, 900, 1100, 900, 976, 1000], { size: 17 }),
  P("*The val column of v8 comes from the config note: its run folder was later overwritten by another run, so its test files no longer exist on disk.*"),
);

// 14
add(H1("14. Glossary"),
  ...table(["Term", "Plain meaning in this project"], [
    ["Ablation", "An experiment that changes exactly one thing, to measure the effect of that thing alone."],
    ["Attention (self-attention)", "The operation inside a Transformer block where each vector takes a weighted average of all the others. The weights say 'how much should I look at you'."],
    ["Augmentation", "Random changes to a training image (crop, flip, brightness) so the network sees a slightly different picture each epoch and cannot memorise pixels."],
    ["AUSUC", "Area under the seen-unseen curve: seen accuracy against unseen accuracy for every possible gamma. One number that does not depend on which gamma was chosen."],
    ["Backbone", "The large pretrained network a method is built on. Here: OpenAI CLIP ViT-B/16."],
    ["Backpropagation / gradient", "The calculation of how much the loss would change if each trainable number changed a little. Training moves each number a small step in the direction that lowers the loss."],
    ["Bootstrap interval", "Re-sampling the evaluation images many times to see how much a score would vary by chance. Gives an error bar."],
    ["Calibration (calibrated stacking)", "Subtracting a constant gamma from seen-pair scores so that unseen pairs can compete."],
    ["Causal mask", "The rule in CLIP's text encoder that a token may only attend to tokens on its left."],
    ["Class token (CLS)", "An extra vector added to the image patches. After the last block it is the summary of the image and becomes the image embedding."],
    ["CLIP", "Contrastive Language-Image Pre-training. The vision-language model used here: an image encoder and a text encoder trained on 400M image-caption pairs."],
    ["Composition / compositional zero-shot learning", "Recognising a new *combination* (crop + disease) of parts that were each seen in other combinations."],
    ["Contrastive loss", "CLIP's pretraining loss: pull matching image-caption pairs together, push non-matching ones apart."],
    ["Cosine similarity", "The dot product of two vectors of length 1. Ranges from -1 to 1; higher means more similar direction."],
    ["Cross-entropy", "The classification loss: minus the log of the probability the model gave to the correct class."],
    ["CSP (Compositional Soft Prompting)", "The method Stage B follows: one learnable vector per primitive (here crop and disease), placed inside a fixed sentence."],
    ["Embedding", "A list of numbers (a vector) that represents an image, a word or a sentence. Here 512 numbers."],
    ["Ensemble", "Combining the outputs of several models into one prediction."],
    ["Epoch", "One pass over all training images."],
    ["Fine-tuning", "Continuing to train a pretrained network on new data."],
    ["Frozen", "Weights that are used in the computation but not updated by training."],
    ["Gamma", "The calibration constant. Around 0.07 for the CLIP models (cosine units), around 6.7 for the ensemble (log-probability units)."],
    ["GAN / WGAN-GP", "Generative adversarial network: a generator that makes fake samples and a critic that judges them, trained against each other. WGAN-GP is a stable variant."],
    ["Generalised zero-shot", "The test setting where the model must choose among seen *and* unseen classes together."],
    ["Harmonic mean (HM)", "2 x S x U / (S + U) for seen accuracy S and unseen accuracy U. High only if both are high."],
    ["Held-out / pseudo-unseen pair", "A seen pair removed from training on purpose, to act as a stand-in for an unseen pair during validation."],
    ["Hub prompt", "A class prompt whose embedding is close to many images, so it wins far too often."],
    ["Hyperparameter", "A setting chosen by the experimenter rather than learned (learning rate, rank, gamma, number of epochs)."],
    ["Learning rate", "The size of each training step."],
    ["Logit / logit scale", "A score before softmax. The logit scale (100) is the number cosines are multiplied by to make them logits."],
    ["LoRA (Low-Rank Adaptation)", "Adding a small trainable matrix product B A to a frozen weight matrix."],
    ["MLP", "A small two-layer network. Each Transformer block contains one."],
    ["Overfitting", "Fitting the training examples so closely that performance on new examples gets worse."],
    ["Pair", "One crop combined with one disease: a class in this task. 333 in total."],
    ["Patch", "A 16 x 16 pixel square of the image. The image encoder treats each as one input token."],
    ["Position embedding", "A learned vector added to each token so the network knows its place in the sequence or grid."],
    ["Primitive", "One part of a pair: a crop or a disease."],
    ["Prompt", "The sentence that describes a class to the text encoder."],
    ["Prompt learning", "Training some of the vectors that make up the prompt instead of writing all of it by hand."],
    ["Rank", "The number of independent directions a matrix can change. LoRA's r."],
    ["Seen / unseen pair", "A pair with / without training images."],
    ["Softmax", "The function that turns a list of scores into probabilities that sum to 1."],
    ["Temperature", "1 / logit scale. A low temperature (0.01 here) makes the softmax sharp."],
    ["Test-time augmentation (TTA)", "Predicting on several altered copies of the test image and averaging."],
    ["Token", "A word piece. Text is split into tokens before entering the text encoder (maximum 77)."],
    ["Top-1 / Top-5", "Correct if the true class is the highest / among the five highest scores."],
    ["Transformer block", "One layer of the encoders: self-attention followed by an MLP, each with a residual connection."],
    ["Validation split", "Data taken out of the training set to choose settings, so that the test set stays untouched."],
    ["ViT-B/16", "Vision Transformer, Base size, 16-pixel patches: the image encoder used."],
    ["VLM", "Vision-language model."],
    ["Zero-shot", "Classifying without any training example of the class (or, in Stage A, without any training at all)."]],
    [2600, W - 2600], { size: 18 }),
);

// Appendix: figure files
add(H1("Appendix. Figure files"),
  P("Every figure is an editable draw.io file in *docs/vlm/diagrams/breakdown/*, with a PNG next to it. They are generated by *docs/vlm/diagrams/make_breakdown_diagrams.py*; edit the script (or the .drawio file in draw.io) to change a figure. The titles inside the images carry no figure number, so they can be renumbered freely in the report's appendix."),
  ...table(["Figure", "File", "Shows"], [
    ["1", "fig01_pipeline", "The whole pipeline"], ["2", "fig02_prompts", "Zero-shot prompt changes"], ["3", "fig03_text_encoder", "Inside the text encoder"],
    ["4", "fig04_composition", "How an unseen pair is composed"], ["5", "fig05_loss", "The training losses"], ["6", "fig06_lora", "LoRA"],
    ["7", "fig07_calibration", "Calibration with gamma"], ["8", "fig08_validation", "Validation and refit"], ["9", "fig09_feature_gen", "Feature generation"],
    ["10", "fig10_ensemble", "The ensemble"], ["11", "fig11_progress", "Progress over versions"], ["12", "fig12_lesion", "Lesion-level attempts"]], [1000, 3000, W - 4000]),
);

// ---------- document ----------
const heading = (size, before, after, color) => ({ run: { font: FONT, size, bold: true, color }, paragraph: { spacing: { before, after }, keepNext: true } });
const doc = new Document({
  creator: "VLM pipeline", title: "VLM Pipeline: Technical Breakdown",

  styles: {
    default: {
      document: { run: { font: FONT, size: 21 } },
      heading1: heading(32, 0, 200, "1B1F24"), heading2: heading(26, 280, 120, "1B1F24"), heading3: heading(22, 200, 100, "1B1F24"),
    },
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 320 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["VLM technical breakdown  |  page ", PageNumber.CURRENT], font: FONT, size: 16, color: "868E96" })] })] }) },
    children: c,
  }],
});
const out = path.join(ROOT, "docs", "vlm", "VLM_technical_breakdown.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length); });
