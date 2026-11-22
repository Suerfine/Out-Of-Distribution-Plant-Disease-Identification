# Multi-pair LeafGAN development study

Run the five generators sequentially with `run_generators.py`. Inputs come
from the existing 5,798-image fitting split, matching healthy and diseased
photos by crop and source collection. Each pair uses the authors' LeafGAN
generator, discriminator, losses and pretrained LFLSeg attention. The pilot
uses 60 epochs, 256 px inputs, a 0.35 mask threshold and rejects nearly empty or
nearly full masks. The eight source previews per pair are held aside from
generator training. The source pool may shrink after mask filtering.

Inspect epoch 60 previews and candidate outputs before using their labels.
Record acceptable source filenames by pair in
`multi_pair/reviewed_sources.json`, with at most 40 per pair.
Run `export_multi.py` to export those images. Review the exported files against
their healthy sources before classifier training, and remove any rejected rows
from `multi_pair/composites.json`.

Run `classifier_multi.py --arm leafgan` and
`classifier_multi.py --arm real_only` for the matched comparison.
Both start from the same saved development checkpoint and train the frozen
backbone's disease head for three epochs, with identical example counts and
pair labels. The real-only arm fills each extra slot with a fitting photo of
the same crop and disease. Compare the final epoch on the real development
validation split, reporting generated pairs, same disease on other crops,
and the subset of those other pairs withheld during fitting. Also report
overall seen and held-out scores. Official test data is outside this study.
