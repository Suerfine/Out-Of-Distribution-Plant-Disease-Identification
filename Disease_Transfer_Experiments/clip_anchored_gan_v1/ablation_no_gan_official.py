"""Official-test ablation: the gated system (M5) and the blends (M6) with and without GAN vectors in the disease head.

Both arms use the same saved official CLIP encoder (clip_disease_only/official/visual_encoder.pt) and the same saved
ConvNeXt factor logits. The 'gan' arm reloads the saved disease_plus_gan_head.pt and must reproduce the published
counts (asserted). The 'real_only' arm fits the same one-epoch head on the 8,000 real training embeddings only.
Offsets and blend weights come from the development selection of each arm (ablation_no_gan_dev.py), never from test.
Nothing existing is modified; outputs go to ablation_no_gan/.
"""
import json
from copy import deepcopy

from study import (
    HERE, PROJECT, WORKSPACE, COLUMNS, np, pd, torch,
    load_clip, text_conditions, train_lora, ensure_resized, clip_transforms,
    seed_everything, fit_classifier,
)
from dpd.feature_gen import CosineClassifier


def log_probs(scores):  # copied from soft_gate.py: importing it would put a second train.py on sys.path
    return scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)


def soft_weight(gap, center, temperature, strength, bias):  # copied from soft_gate.py
    transition = 1 / (1 + np.exp(np.clip((center - gap) / temperature, -50, 50)))
    return np.clip(.5 + bias + strength * (transition - .5), 0., 1.)[:, None]

OUT = HERE / 'ablation_no_gan'
ENCODER = HERE / 'clip_disease_only/official/visual_encoder.pt'
GAN_HEAD = HERE / 'clip_disease_only/official/disease_plus_gan_head.pt'
EXPECTED = {  # published official counts [seen, unseen] for the GAN arm
    ('hard', 'zero'): [1458, 700], ('hard', 'tuned'): [1336, 1199],
    ('fixed', 'zero'): [1394, 633], ('fixed', 'tuned'): [1311, 1318],
    ('soft', 'zero'): [1445, 689], ('soft', 'tuned'): [1288, 1323],
}


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    base = json.loads((PROJECT / 'config/vlm/v11.3_lora_features_config.json').read_text())
    lora = json.loads((PROJECT / 'config/vlm/v10.3_text_anchor_three_splits_config.json').read_text())
    for cfg in (base, lora):
        cfg['model']['pretrained'] = str(HERE / 'ViT-B-16-state.pt')
    prepared = HERE.parent / 'cnn_detail_base_v1/runs/development_seed42/prepared.json'
    pairs = pd.DataFrame(json.loads(prepared.read_text())['pairs']).sort_values('pair_id').reset_index(drop=True)
    ci, di = pairs.crop_id.to_numpy(), pairs.disease_id.to_numpy()
    held = pairs.loc[pairs.is_unseen, 'pair_id'].to_numpy()
    data = WORKSPACE / 'Plant_Disease'
    fit = pd.read_csv(data / 'DPD_Small_train.csv', header=None, names=COLUMNS)
    seen_ids = np.sort(fit.pair_id.unique())
    assert len(fit) == 8000 and len(seen_ids) == 323 and len(held) == 10
    ids = np.sort(np.r_[seen_ids, held])
    seen_cols = np.isin(ids, seen_ids)

    tf = clip_transforms(lora['input'], lora['augmentation'])
    seed_everything(lora['seed'])
    model, _ = train_lora.build(lora, pairs, 'cuda')
    model.clip.visual.load_state_dict(torch.load(ENCODER, map_location='cpu', weights_only=True))

    def encode(frame):
        cache = ensure_resized(frame.file, lora['input']['resize_cache'])
        loader = train_lora.loader(frame, cache, tf[1], lora, shuffle=False)
        return train_lora.encode(model, loader, 'cuda', lora['runtime']['amp'],
                                 tuple(lora['evaluation'].get('tta', ['identity'])))[0]

    print('Encoding the 8,000 training images', flush=True)
    x = encode(fit)
    frames = {s: pd.read_csv(data / f'DPD_Small_{s}.csv', header=None, names=COLUMNS) for s in ('test_seen', 'test_unseen')}
    features = {}
    for s, frame in frames.items():
        print(f'Encoding {s}', flush=True)
        features[s] = encode(frame)

    disease_cfg = deepcopy(base)
    disease_cfg['condition']['parts'] = ['disease']
    net, _, tokenizer = load_clip(base['model'], 'cuda')
    _, conditions = text_conditions(disease_cfg, net, tokenizer, pairs, 'cuda')
    del net
    _, first = np.unique(di, return_index=True)
    seed_everything(42)
    head = fit_classifier(
        torch.as_tensor(x, device='cuda'), torch.as_tensor(fit.disease_id.to_numpy(), device='cuda'),
        torch.ones(175, device='cuda', dtype=torch.bool), torch.as_tensor(conditions[first], device='cuda'),
        base['classifier'], 1, tag='disease_real_only_official')
    torch.save(head.state_dict(), OUT / 'disease_real_only_official_head.pt')
    scale = base['classifier']['logit_scale']
    gan_weight = torch.load(GAN_HEAD, map_location='cpu', weights_only=True)['weight']

    labels, is_seen, crop, cnn_disease, logits = [], [], [], [], {'gan': [], 'real_only': []}
    with np.load(PROJECT / 'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz') as cnn:
        for s, frame in frames.items():
            labels.append(frame.pair_id.to_numpy())
            is_seen.append(np.full(len(frame), s == 'test_seen'))
            crop.append(cnn[f'{s}_crop'])
            cnn_disease.append(cnn[f'{s}_disease'])
            f = torch.as_tensor(features[s], device='cuda')
            with torch.inference_mode():
                logits['real_only'].append(head(f).cpu().numpy())
                logits['gan'].append(CosineClassifier(gan_weight, scale).cuda()(f).cpu().numpy())
    labels, is_seen = np.concatenate(labels), np.concatenate(is_seen)
    crop, cnn_disease = np.concatenate(crop)[:, ci[ids]], np.concatenate(cnn_disease)
    logits = {k: np.concatenate(v) for k, v in logits.items()}
    cnn_scores = crop + cnn_disease[:, di[ids]]

    def predict(clip_disease, rule, params, offset):
        clip_scores = crop + clip_disease[:, di[ids]]
        gap_clip = clip_scores[:, seen_cols].max(1) - clip_scores[:, ~seen_cols].max(1)
        if rule == 'hard':
            seen_pred = ids[seen_cols][cnn_scores[:, seen_cols].argmax(1)]
            unseen_pred = ids[~seen_cols][clip_scores[:, ~seen_cols].argmax(1)]
            return np.where(gap_clip >= offset, seen_pred, unseen_pred)
        w = params['weight'] if rule == 'fixed' else soft_weight(gap_clip, **params)
        clip_norm, cnn_norm = (log_probs(s.astype(np.float64)) for s in (clip_scores, cnn_scores))
        mix = clip_norm + w * (cnn_norm - clip_norm)
        gap = mix[:, seen_cols].max(1) - mix[:, ~seen_cols].max(1)
        seen_pred = ids[seen_cols][mix[:, seen_cols].argmax(1)]
        unseen_pred = ids[~seen_cols][mix[:, ~seen_cols].argmax(1)]
        return np.where(gap >= offset, seen_pred, unseen_pred)

    def score(pred):
        hit = pred == labels
        n, m = int((hit & is_seen).sum()), int((hit & ~is_seen).sum())
        a, b = n / is_seen.sum(), m / (~is_seen).sum()
        return dict(correct=[n, m], seen=float(a), unseen=float(b), h=float(2 * a * b / (a + b)) if a + b else 0.)

    result = {}
    for arm in ('gan', 'real_only'):
        dev = json.loads((OUT / f'dev_{arm}.json').read_text())
        sel, gamma = dev['selection'], dev['hard_gamma']
        result[arm] = {
            'hard': {'zero': score(predict(logits[arm], 'hard', None, 0.)),
                     'tuned': score(predict(logits[arm], 'hard', None, gamma))},
        }
        for rule in ('fixed', 'soft'):
            result[arm][rule] = {mode: score(predict(logits[arm], rule, sel[rule][mode]['params'], sel[rule][mode]['offset']))
                                 for mode in ('zero', 'tuned')}
    for (rule, mode), counts in EXPECTED.items():
        assert result['gan'][rule][mode]['correct'] == counts, (rule, mode, result['gan'][rule][mode]['correct'], counts)
    result['disease_head_accuracy'] = {
        arm: float((logits[arm].argmax(1) == pd.concat(frames.values()).disease_id.to_numpy()).mean())
        for arm in logits}
    (OUT / 'official_results.json').write_text(json.dumps(result, indent=2))
    for arm in ('gan', 'real_only'):
        print(f'\n== {arm}')
        for rule in ('hard', 'fixed', 'soft'):
            for mode, v in result[arm][rule].items():
                print(f"  {rule:5s} {mode:5s} seen {v['seen']:.4f} unseen {v['unseen']:.4f} H {v['h']:.4f}  {v['correct']}")
    print('disease head accuracy (all test images):', result['disease_head_accuracy'])


if __name__ == '__main__':
    main()
