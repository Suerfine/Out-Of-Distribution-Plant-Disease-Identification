"""Refit disease-only SigLIP (untuned or LoRA-adapted) + Crop-Anchored GAN on official training data and evaluate the two experts."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / 'dpd').exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import synthetic_gate as synthesis
from clip_disease_only import OUT, score
from study import (
    HERE, PROJECT as _STUDY_PROJECT, WORKSPACE, COLUMNS,
    load_clip, text_conditions, train_lora, ensure_resized, clip_transforms,
    seed_everything, fit_classifier,
)
PROJECT = _REPO_ROOT if (_REPO_ROOT / 'config').exists() else _STUDY_PROJECT
from dpd.clip_utils import image_norm_stats
from dpd.data import pair_table, read_metadata


def _load_pairs():
    prepared = HERE.parent / 'cnn_detail_base_v1/runs/development_seed42/prepared.json'
    if prepared.exists():
        return pd.DataFrame(json.loads(prepared.read_text())['pairs']).sort_values('pair_id').reset_index(drop=True)
    return pair_table(read_metadata())


def _resolve_data_dir():
    env = os.environ.get('DPD_ROOT')
    if env:
        return Path(env)
    if (PROJECT / 'DPD_Small' / 'DPD_Small_train.csv').exists():
        return PROJECT / 'DPD_Small'
    return WORKSPACE / 'Plant_Disease'


def main(no_lora: bool = False, disease_only_lora: bool = False, gamma: float = 3.0, lora_epochs: int = 8):
    torch.set_num_threads(4)
    tag = 'official_siglip_untuned' if no_lora else ('official_siglip_disease_only_lora' if disease_only_lora else 'official_siglip_lora')
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        print((out / 'results.json').read_text(), flush=True)
        return

    base = json.loads((PROJECT / 'config/vlm/v11.3_lora_features_config.json').read_text())
    lora = json.loads((PROJECT / 'config/vlm/v14.1_clip_siglip_lora_config.json').read_text())
    base['model'] = deepcopy(lora['model'])
    if disease_only_lora:
        lora['training']['loss_weights'].update(pair=0.0, crop=0.0)

    pairs = _load_pairs()
    assert pairs.pair_id.tolist() == list(range(333))
    ci, di = pairs.crop_id.to_numpy(), pairs.disease_id.to_numpy()
    held = pairs.loc[pairs.is_unseen, 'pair_id'].to_numpy()

    data = _resolve_data_dir()
    fit = pd.read_csv(data / 'DPD_Small_train.csv', header=None, names=COLUMNS)
    fit['source'] = fit.file.str.split('_').str[0]
    fit['row'] = np.arange(len(fit))
    seen_ids = np.sort(fit.pair_id.unique())
    assert len(fit) == 8000 and fit.file.is_unique and len(seen_ids) == 323
    assert len(held) == 10 and not np.intersect1d(held, seen_ids).size

    head_epochs = 1
    budget = base['synthesis']['n_per_unseen']
    selection = dict(
        backbone=lora['model']['arch'],
        pretrained=lora['model']['pretrained'],
        tuned=not no_lora,
        gamma=gamma,
        lora_epochs=0 if no_lora else lora_epochs,
        head_epochs=head_epochs,
        training_images=len(fit),
        synthetic_per_pair=budget,
        candidate_target_pairs=pairs.loc[held, 'pair'].tolist(),
        lora_loss_weights=None if no_lora else lora['training']['loss_weights'],
    )
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))

    norm_stats = image_norm_stats(lora['model'])
    tf = clip_transforms(lora['input'], lora['augmentation'], norm_stats=norm_stats)

    def encode(frame, model):
        cache = ensure_resized(frame.file, lora['input']['resize_cache'])
        loader = train_lora.loader(frame, cache, tf[1], lora, shuffle=False)
        return train_lora.encode(
            model, loader, 'cuda', lora['runtime']['amp'],
            tuple(lora['evaluation'].get('tta', ['identity'])),
        )[0]

    seed_everything(lora['seed'])
    if no_lora:
        lora['image_encoder']['lora']['last_n_blocks'] = 0
    model, params = train_lora.build(lora, pairs, 'cuda')

    if not no_lora:
        encoder_path = out / 'visual_encoder.pt'
        if encoder_path.exists():
            model.clip.visual.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
        else:
            cache = ensure_resized(fit.file, lora['input']['resize_cache'])
            print(f"Refitting SigLIP LoRA ({lora['model']['arch']}) on all 8,000 training images", flush=True)
            train_lora.fit(model, params, fit, seen_ids.tolist(), lora_epochs, lora, cache, tf,
                           log=lambda row: print(row, flush=True))
            torch.save(model.clip.visual.state_dict(), encoder_path.with_suffix('.tmp'))
            encoder_path.with_suffix('.tmp').replace(encoder_path)
    else:
        print(f"Using untuned frozen SigLIP ({lora['model']['arch']}) without LoRA", flush=True)

    print('Fitting the one-epoch disease head with crop-anchored GAN vectors', flush=True)
    x = encode(fit, model)
    synthesis.OUT = out
    synthesis.EXTRA_PER_PAIR = budget
    vectors, synthetic_ids = synthesis.synthetic_bank(fit, fit, pairs, held, base, x)
    assert np.array_equal(synthetic_ids, np.repeat(held, budget))

    disease_cfg = deepcopy(base)
    disease_cfg['condition']['parts'] = ['disease']
    net, _, tokenizer = load_clip(base['model'], 'cuda')
    _, conditions = text_conditions(disease_cfg, net, tokenizer, pairs, 'cuda')
    del net

    disease_ids, indices = np.unique(di, return_index=True)
    assert np.array_equal(disease_ids, np.arange(175))
    seed_everything(42)
    head = fit_classifier(
        torch.as_tensor(np.concatenate((x, vectors)), device='cuda'),
        torch.as_tensor(np.concatenate((fit.disease_id.to_numpy(), di[synthetic_ids])), device='cuda'),
        torch.ones(175, device='cuda', dtype=torch.bool),
        torch.as_tensor(conditions[indices], device='cuda'),
        base['classifier'], head_epochs, tag='disease_plus_gan',
    )
    torch.save(head.state_dict(), out / 'disease_plus_gan_head.pt')

    cnn_candidates = [
        PROJECT / 'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz',
        Path('/content/drive/MyDrive/Nat_AML_DPD_group_assignment/runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'),
    ]
    cnn_path = next((p for p in cnn_candidates if p.exists()), None)
    if cnn_path is None:
        raise FileNotFoundError(
            'ConvNeXt factor logits not found. Checked:\n  ' + '\n  '.join(str(p) for p in cnn_candidates)
        )

    all_labels, all_seen = [], []
    prediction = {g: [] for g in (0.0, gamma)}
    disease_accuracy = {}
    with np.load(cnn_path) as cnn:
        for split, count in [('test_seen', 2000), ('test_unseen', 1448)]:
            frame = pd.read_csv(data / f'DPD_Small_{split}.csv', header=None, names=COLUMNS)
            labels = frame.pair_id.to_numpy()
            is_seen = split == 'test_seen'
            assert len(frame) == count and set(fit.file).isdisjoint(frame.file)
            assert (np.isin(labels, held) == (not is_seen)).all()
            print(f'Encoding official {split}', flush=True)
            features = encode(frame, model)
            with torch.inference_mode():
                disease = head(torch.as_tensor(features, device='cuda')).cpu().numpy()
            crop = cnn[f'{split}_crop']
            assert np.isfinite(disease).all() and len(crop) == count
            disease_hit = disease.argmax(1) == frame.disease_id.to_numpy()
            disease_accuracy[split] = dict(
                correct=int(disease_hit.sum()),
                images=count,
                accuracy=float(disease_hit.mean()),
            )
            seen_scores = crop[:, ci[seen_ids]] + cnn[f'{split}_disease'][:, di[seen_ids]]
            seen_pred = seen_ids[seen_scores.argmax(1)]
            routed_seen_scores = crop[:, ci[seen_ids]] + disease[:, di[seen_ids]]
            unseen_scores = crop[:, ci[held]] + disease[:, di[held]]
            unseen_pred = held[unseen_scores.argmax(1)]
            gap = routed_seen_scores.max(1) - unseen_scores.max(1)
            for g in prediction:
                prediction[g].append(np.where(gap >= g, seen_pred, unseen_pred))
            all_labels.append(labels)
            all_seen.append(np.full(count, is_seen))
            np.savez_compressed(
                out / f'{split}.npz',
                filenames=frame.file.to_numpy(dtype=str),
                labels=labels,
                seen_pred=seen_pred,
                unseen_pred=unseen_pred,
                gap=gap,
            )

    labels, seen_mask = np.concatenate(all_labels), np.concatenate(all_seen)
    result = dict(
        selection=selection,
        disease_accuracy=disease_accuracy,
        official={f'{g:g}': score(np.concatenate(pred), labels, seen_mask) for g, pred in prediction.items()},
    )
    temp = out / 'results.tmp'
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(out / 'results.json')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-lora', action='store_true', help='Use untuned frozen SigLIP without LoRA fine-tuning')
    parser.add_argument('--disease-only-lora', action='store_true', help='Train LoRA with disease loss only')
    parser.add_argument('--gamma', type=float, default=3.0, help='Routing offset for seen vs unseen expert')
    parser.add_argument('--lora-epochs', type=int, default=8, help='Number of LoRA epochs when tuning')
    args = parser.parse_args()
    main(args.no_lora, args.disease_only_lora, args.gamma, args.lora_epochs)
