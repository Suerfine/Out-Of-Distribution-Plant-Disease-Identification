"""Refit disease-only CLIP on official training data and evaluate the two experts."""
from copy import deepcopy
import argparse

import synthetic_gate as synthesis
from clip_disease_only import OUT, score
from study import (
    HERE, PROJECT, WORKSPACE, COLUMNS, np, pd, torch, json,
    load_clip, text_conditions, train_lora, ensure_resized, clip_transforms,
    seed_everything, fit_classifier,
)


def main(disease_only_lora=False):
    torch.set_num_threads(4)
    out = OUT / ('official_disease_only_lora' if disease_only_lora else 'official')
    out.mkdir(exist_ok=True)
    if (out / 'results.json').exists():
        print((out / 'results.json').read_text(), flush=True)
        return
    gamma = json.loads((OUT / 'development_results.json').read_text())['disease_plus_gan']['gamma']
    base = json.loads((PROJECT / 'config/vlm/v11.3_lora_features_config.json').read_text())
    lora = json.loads((PROJECT / 'config/vlm/v10.3_text_anchor_three_splits_config.json').read_text())
    if disease_only_lora:
        lora['training']['loss_weights'].update(pair=0., crop=0.)
    for cfg in (base, lora):
        cfg['model']['pretrained'] = str(HERE / 'ViT-B-16-state.pt')
    prepared = HERE.parent / 'cnn_detail_base_v1/runs/development_seed42/prepared.json'
    pairs = pd.DataFrame(json.loads(prepared.read_text())['pairs']).sort_values('pair_id').reset_index(drop=True)
    assert pairs.pair_id.tolist() == list(range(333))
    ci, di = pairs.crop_id.to_numpy(), pairs.disease_id.to_numpy()
    held = pairs.loc[pairs.is_unseen, 'pair_id'].to_numpy()
    data = WORKSPACE / 'Plant_Disease'
    fit = pd.read_csv(data / 'DPD_Small_train.csv', header=None, names=COLUMNS)
    fit['source'] = fit.file.str.split('_').str[0]
    fit['row'] = np.arange(len(fit))
    seen_ids = np.sort(fit.pair_id.unique())
    assert len(fit) == 8000 and fit.file.is_unique and len(seen_ids) == 323
    assert len(held) == 10 and not np.intersect1d(held, seen_ids).size
    lora_epochs, head_epochs = 8, 1
    budget = base['synthesis']['n_per_unseen']
    selection = dict(gamma=gamma, lora_epochs=lora_epochs, head_epochs=head_epochs,
                     training_images=len(fit), synthetic_per_pair=budget,
                     candidate_target_pairs=pairs.loc[held, 'pair'].tolist(),
                     lora_loss_weights=lora['training']['loss_weights'])
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    tf = clip_transforms(lora['input'], lora['augmentation'])

    def encode(frame, model):
        cache = ensure_resized(frame.file, lora['input']['resize_cache'])
        loader = train_lora.loader(frame, cache, tf[1], lora, shuffle=False)
        return train_lora.encode(
            model, loader, 'cuda', lora['runtime']['amp'],
            tuple(lora['evaluation'].get('tta', ['identity'])),
        )[0]

    seed_everything(lora['seed'])
    model, params = train_lora.build(lora, pairs, 'cuda')
    encoder_path = out / 'visual_encoder.pt'
    if encoder_path.exists():
        model.clip.visual.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
    else:
        cache = ensure_resized(fit.file, lora['input']['resize_cache'])
        print('Refitting CLIP LoRA on all 8,000 training images', flush=True)
        train_lora.fit(model, params, fit, seen_ids.tolist(), lora_epochs, lora, cache, tf,
                       log=lambda row: print(row, flush=True))
        torch.save(model.clip.visual.state_dict(), encoder_path.with_suffix('.tmp'))
        encoder_path.with_suffix('.tmp').replace(encoder_path)

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

    cnn_path = PROJECT / 'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'
    all_labels, all_seen = [], []
    prediction = {g: [] for g in (0., gamma)}
    disease_accuracy = {}
    with np.load(cnn_path) as cnn:
        for split, count in [('test_seen', 2000), ('test_unseen', 1448)]:
            frame = pd.read_csv(data / f'DPD_Small_{split}.csv', header=None, names=COLUMNS)
            labels = frame.pair_id.to_numpy()
            is_seen = split == 'test_seen'
            assert len(frame) == count and set(fit.file).isdisjoint(frame.file)
            assert (np.isin(labels, held) == (not is_seen)).all()
            with np.load(HERE / 'official' / f'anchored_{split}.npz') as reference:
                assert np.array_equal(reference['filenames'], frame.file.to_numpy())
                assert np.array_equal(reference['labels'], labels)
            print(f'Encoding official {split}', flush=True)
            features = encode(frame, model)
            with torch.inference_mode():
                disease = head(torch.as_tensor(features, device='cuda')).cpu().numpy()
            crop = cnn[f'{split}_crop']
            assert np.isfinite(disease).all() and len(crop) == count
            disease_hit = disease.argmax(1) == frame.disease_id.to_numpy()
            disease_accuracy[split] = dict(correct=int(disease_hit.sum()), images=count,
                                           accuracy=float(disease_hit.mean()))
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
            np.savez_compressed(out / f'{split}.npz', filenames=frame.file.to_numpy(dtype=str),
                                labels=labels, seen_pred=seen_pred, unseen_pred=unseen_pred, gap=gap)
    labels, seen_mask = np.concatenate(all_labels), np.concatenate(all_seen)
    result = dict(selection=selection, disease_accuracy=disease_accuracy, official={
        f'{g:g}': score(np.concatenate(pred), labels, seen_mask) for g, pred in prediction.items()
    })
    temp = out / 'results.tmp'
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(out / 'results.json')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disease-only-lora', action='store_true')
    main(parser.parse_args().disease_only_lora)
