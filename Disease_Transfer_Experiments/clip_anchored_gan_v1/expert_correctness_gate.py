"""Compare routing labels and expert-correctness labels on held-out development pairs."""
import argparse
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BASE = next(parent for parent in HERE.parents if (parent/'dpd').is_dir())
DETAIL = HERE.parent/'cnn_detail_base_v1'
DATA = BASE.parent/'Plant_Disease'
sys.path.insert(0, str(BASE))
os.environ.setdefault('DPD_ROOT', str(DATA))

from dpd.feature_gen import CosineClassifier


def fit_gate(x, target, fit):
    center = x[fit].mean(axis=0)
    scale = np.maximum(x[fit].std(axis=0), 1e-6)
    features = torch.as_tensor((x - center) / scale, dtype=torch.float64)
    y = torch.as_tensor(target[fit], dtype=torch.float64)
    head = torch.nn.Linear(x.shape[1], 1, dtype=torch.float64)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.LBFGS(head.parameters(), max_iter=60, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(features[fit]).flatten(), y)
        loss = loss + .05 * head.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return dict(center=center.tolist(), scale=scale.tolist(),
                weight=head.weight.detach().flatten().tolist(), bias=head.bias.item())


def gate_score(x, model):
    return ((x - model['center']) / model['scale']) @ model['weight'] + model['bias']


def fit_scores(x, target, fit):
    return gate_score(x, fit_gate(x, target, fit))


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def borderline_override(features, gamma, rule, *, to_unseen=False):
    gap, seen_margin, unseen_margin = features.T
    if rule is None:
        return np.zeros(len(gap), dtype=bool)
    width, min_preferred, max_other = rule
    if to_unseen:
        return ((gap >= gamma) & (gap < gamma+width) &
                (unseen_margin >= min_preferred) & (seen_margin <= max_other))
    return ((gap >= gamma-width) & (gap < gamma) &
            (seen_margin >= min_preferred) & (unseen_margin <= max_other))


def test_borderline(features, gamma, seen_ok, unseen_ok, y, seen, rows, *, to_unseen=False):
    recovered, protected = (unseen_ok, seen_ok) if to_unseen else (seen_ok, unseen_ok)
    gain_name, loss_name = ('unseen_recovered', 'seen_lost') if to_unseen else ('seen_recovered', 'unseen_lost')
    grid = list(product((.1, .25, .5, .75, 1., 1.5, 2.),
                        (.5, 1., 2., 3., 4., 6., 8., 10.),
                        (.25, .5, 1., 2., 3., 4., 6.)))
    candidates = [(rule, borderline_override(features, gamma, rule, to_unseen=to_unseen)) for rule in grid]

    def select(fit):
        best_rule, best_gain, best_flips = None, 0, 0
        for rule, flips in candidates:
            changed = fit & flips
            if np.any(changed & protected):
                continue
            gain = int((changed & recovered).sum())
            size = int(changed.sum())
            if (gain, -size) > (best_gain, -best_flips):
                best_rule, best_gain, best_flips = rule, gain, size
        return best_rule

    def summary(mask, changes):
        flipped = mask & changes
        return {'overrides':int(flipped.sum()), gain_name:int((flipped & recovered).sum()),
                loss_name:int((flipped & protected).sum())}

    source = np.array([r['filename'].split('_', 1)[0] for r in rows])
    rng = np.random.default_rng(42)
    groups = {pair:i%5 for group in (y[seen], y[~seen])
              for i,pair in enumerate(rng.permutation(np.unique(group)))}
    pair_folds = np.array([groups[pair] for pair in y])
    all_images = np.ones(len(y), dtype=bool)
    checks = {}
    for kind, set_ids in [('pair_folds', pair_folds), ('leave_one_source_out', source)]:
        changed = np.zeros(len(y), dtype=bool)
        records = []
        for key in np.unique(set_ids):
            test = set_ids == key
            rule = select(~test)
            flips = borderline_override(features, gamma, rule, to_unseen=to_unseen)
            changed[test] = flips[test]
            records.append(dict(fold=str(key), rule=rule, **summary(test, flips)))
        checks[kind] = dict(**summary(all_images, changed), folds=records,
                            by_source={str(s):summary(source==s, changed) for s in np.unique(source)
                                       if np.any(changed & (source==s))})
    rule = select(all_images)
    selected = summary(all_images, borderline_override(features, gamma, rule, to_unseen=to_unseen))
    result = dict(gamma=gamma, grid_size=len(grid), development_rule=rule, development=selected, checks=checks)
    filename = 'borderline_gate_zero_development.json' if gamma == 0 else 'borderline_gate_development.json'
    if to_unseen:
        filename = 'borderline_to_unseen_zero_development.json'
    out = HERE/'convnext_crop_clip_disease'/filename
    write_json(out, result)
    print(json.dumps(dict(rule=rule, development=selected,
                          checks={k:{key:v[key] for key in selected} for k,v in checks.items()})), flush=True)


def two_way_selection(features, gamma, seen_ok, unseen_ok, labels, seen):
    grid = [None, *product((.1, .25, .5, .75, 1., 1.5, 2.),
                           (.5, 1., 2., 3., 4., 6., 8., 10.),
                           (.25, .5, 1., 2., 3., 4., 6.))]
    to_scope = {direction: np.stack([borderline_override(features, gamma, rule, to_unseen=direction)
                                     for rule in grid]) for direction in (False, True)}
    base = features[:, 0] >= gamma
    constrained = 0 if gamma == 0 else 1
    benefit = 1 - constrained
    modes = {'current':([0], [0]), 'to_seen':(range(len(grid)), [0]),
             'to_unseen':([0], range(len(grid))), 'two_way':(range(len(grid)), range(len(grid)))}

    def select(mask, options):
        left, right = [np.array(i) for i in options]
        indices = np.flatnonzero(mask)
        a, b = to_scope[False][left][:, indices], to_scope[True][right][:, indices]
        ds = a @ seen_ok[indices].astype(int)
        ls = b @ seen_ok[indices].astype(int)
        du = b @ unseen_ok[indices].astype(int)
        lu = a @ unseen_ok[indices].astype(int)
        deltas = (ds[:,None] - ls, du - lu[:,None])
        cost = a.sum(1)[:,None] + b.sum(1)
        order = np.lexsort((cost.ravel(), -deltas[constrained].ravel(), -deltas[benefit].ravel()))
        best = order[np.flatnonzero(deltas[constrained].ravel()[order] >= 0)[0]]
        i, j = np.unravel_index(best, cost.shape)
        return left[i], right[j]

    def evaluate(mask, chosen):
        i, j = chosen
        route = (base | to_scope[False][i]) & ~to_scope[True][j]
        return [int((mask & route & seen_ok).sum()), int((mask & ~route & unseen_ok).sum())]

    rng = np.random.default_rng(42)
    groups = {pair:i%5 for g in (labels[seen], labels[~seen])
              for i,pair in enumerate(rng.permutation(np.unique(g)))}
    folds = np.array([groups[pair] for pair in labels])
    full = np.ones(len(labels), dtype=bool)
    result = {}
    for name, options in modes.items():
        locked = select(full, options)
        checks = []
        for fold in range(5):
            test = folds == fold
            chosen = select(~test, options)
            checks.append(dict(fold=fold, counts=evaluate(test, chosen),
                               to_seen=grid[chosen[0]], to_unseen=grid[chosen[1]]))
        result[name] = dict(to_seen=grid[locked[0]], to_unseen=grid[locked[1]],
                            development=evaluate(full, locked),
                            held_out=np.sum([row['counts'] for row in checks],axis=0).tolist(),
                            folds=checks)
    return result


def joint_selection(features, gamma, seen_ok, unseen_ok, labels, seen):
    grid = [None, *product((.1, .25, .5, .75, 1., 1.5, 2.),
                           (.5, 1., 2., 3., 4., 6., 8., 10.),
                           (.25, .5, 1., 2., 3., 4., 6.))]
    base_route = features[:,0] >= gamma
    rng = np.random.default_rng(42)
    groups = {pair:i%5 for g in (labels[seen], labels[~seen])
              for i,pair in enumerate(rng.permutation(np.unique(g)))}
    fold_ids = np.array([groups[pair] for pair in labels])
    train_masks = [np.ones(len(labels),dtype=bool), *(fold_ids != i for i in range(5))]
    test_masks = [train_masks[0], *(~mask for mask in train_masks[1:])]
    floors = [int((seen_ok & base_route & mask).sum()) for mask in train_masks]
    best = [{name:None for name in ('plain','joint')} for _ in train_masks]
    for offset in np.arange(0., 6.01, .25):
        base = features[:,0] >= offset
        a, b = (np.stack([borderline_override(features, offset, rule, to_unseen=d) for rule in grid])
                for d in (False,True))
        for k,(mask,floor) in enumerate(zip(train_masks,floors)):
            s, u = (mask & v for v in (seen_ok,unseen_ok))
            ns, nu = int((base & s).sum()), int((~base & u).sum())
            moved_s = ns + (a @ s.astype(int))[:,None] - (b @ s.astype(int))
            moved_u = nu - (a @ u.astype(int))[:,None] + (b @ u.astype(int))
            n_changed = (a @ mask.astype(int))[:,None] + (b @ mask.astype(int))
            for name in ('plain','joint'):
                if name == 'plain':
                    if ns < floor:
                        continue
                    i,j = 0,0
                else:
                    valid = moved_s >= floor
                    if not valid.any():
                        continue
                    # Maximize unseen hits, then seen hits, then minimize interventions.
                    score = moved_u*10_000_000 + moved_s*10_000 - n_changed
                    i,j = np.unravel_index(np.where(valid,score,-np.inf).argmax(),score.shape)
                value = (int(moved_u[i,j]),int(moved_s[i,j]),-int(n_changed[i,j]),-abs(float(offset)-gamma))
                if best[k][name] is None or value > best[k][name][0]:
                    best[k][name] = (value,float(offset),grid[i],grid[j])
    result = {}
    for name in ('plain','joint'):
        folds = []
        for choices,mask in zip(best,test_masks):
            _,offset,to_seen,to_unseen = choices[name]
            route = ((features[:,0]>=offset) | borderline_override(features,offset,to_seen))
            route &= ~borderline_override(features,offset,to_unseen,to_unseen=True)
            folds.append(dict(offset=offset,to_seen=to_seen,to_unseen=to_unseen,
                              counts=[int((mask & route & seen_ok).sum()),int((mask & ~route & unseen_ok).sum())]))
        result[name] = dict(selection=folds[0], folds=folds[1:],
                            held_out=np.sum([r['counts'] for r in folds[1:]],axis=0).tolist())
    return result


def main(official=False, borderline=False, zero_offset=False, to_unseen=False, reverse_official=False,
         two_way=False, joint=False):
    torch.set_num_threads(4)
    prepared = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    fit, vs, vu = (prepared['roles'][p] for p in ('fit', 'validation_seen', 'validation_unseen'))
    rows = vs + vu
    y = np.array([r['composition_id'] for r in rows])
    seen = np.arange(len(rows)) < len(vs)
    si = np.unique([r['composition_id'] for r in fit])
    ui = np.unique(y[~seen])
    pairs = sorted(prepared['pairs'], key=lambda r:r['pair_id'])
    ci, di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    prior = HERE/'clip_disease_only'
    gamma = json.loads((prior/'development_results.json').read_text())['disease_plus_gan']['gamma']
    scale = json.loads((BASE/'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    weight = torch.load(prior/'disease_plus_gan_head.pt', weights_only=True, map_location='cpu')['weight']
    with np.load(HERE/'runs/seed_42/features.npz') as b:
        index = {n:i for i,n in enumerate(b['filenames'])}
        x = torch.from_numpy(b['embeddings'][[index[r['filename']] for r in rows]])
    with torch.inference_mode():
        clip = CosineClassifier(weight,scale)(x).numpy()
    with np.load(prepared['baseline_validation_logits']) as b:
        crop, disease = b['crop'], b['disease']
    clip_seen = crop[:,ci[si]] + clip[:,di[si]]
    clip_unseen = crop[:,ci[ui]] + clip[:,di[ui]]
    cnn_seen = crop[:,ci[si]] + disease[:,di[si]]
    seen_ok = si[cnn_seen.argmax(1)] == y
    unseen_ok = ui[clip_unseen.argmax(1)] == y
    assert not np.any(seen_ok & unseen_ok)
    gap = clip_seen.max(1) - clip_unseen.max(1)

    def margin(scores):
        two = np.partition(scores, -2, axis=1)[:, -2:]
        return two[:,1]-two[:,0]

    features = np.column_stack((gap, margin(cnn_seen), margin(clip_unseen)))
    assert np.isfinite(features).all()

    def correct(scores, mask, cutoff):
        route = scores >= cutoff
        return int((mask & route & seen_ok).sum()), int((mask & ~route & unseen_ok).sum())

    def threshold(scores, mask, target):
        if not target:
            return float('-inf')
        scores = np.sort(scores[mask & unseen_ok].astype(np.float64))
        assert target <= len(scores)
        return float(np.nextafter(scores[target-1], np.inf))

    assert correct(gap, np.ones(len(y), dtype=bool), gamma) == (593,315)
    if joint:
        rules = joint_selection(features,gamma,seen_ok,unseen_ok,y,seen)
        write_json(HERE/'convnext_crop_clip_disease/joint_override_development.json', rules)
        cutoffs = {'gamma_3_current':gamma}
        overrides = {'gamma_3_current':dict(to_seen=None,to_unseen=None)}
        for name, result in rules.items():
            chosen = result['selection']
            cutoffs[name] = chosen['offset']
            overrides[name] = dict(to_seen=chosen['to_seen'],to_unseen=chosen['to_unseen'])
        print(json.dumps({name:{k:v for k,v in r.items() if k!='folds'} for name,r in rules.items()}),flush=True)
        evaluate_official(pairs,ci,di,scale,margin,dict(cutoffs=cutoffs,overrides=overrides,joint=True))
        return
    if two_way:
        report = {f'{g:g}':two_way_selection(features, g, seen_ok, unseen_ok, y, seen) for g in (0., gamma)}
        filename = HERE/'convnext_crop_clip_disease/two_way_override_development.json'
        write_json(filename, report)
        cutoffs, overrides = {}, {}
        for g, modes in report.items():
            for name, rule in modes.items():
                key = f'gamma_{g}_{name}'
                cutoffs[key] = float(g)
                overrides[key] = dict(to_seen=rule['to_seen'], to_unseen=rule['to_unseen'])
        previous = json.loads((HERE/'convnext_crop_clip_disease/borderline_to_unseen_zero_development.json').read_text())
        cutoffs['gamma_0_previous_reverse'] = 0.
        overrides['gamma_0_previous_reverse'] = dict(to_seen=None, to_unseen=previous['development_rule'])
        print(json.dumps({g:{name:{k:v for k,v in ans.items() if k!='folds'} for name,ans in m.items()}
                          for g,m in report.items()}), flush=True)
        evaluate_official(pairs, ci, di, scale, margin, dict(cutoffs=cutoffs, overrides=overrides))
        return
    if reverse_official:
        source = HERE/'convnext_crop_clip_disease/borderline_to_unseen_zero_development.json'
        locked = json.loads(source.read_text())
        assert locked['gamma'] == 0 and locked['development_rule']
        selection = dict(cutoffs=dict(current=0., reverse_override=0.),
                         override_rule=locked['development_rule'], development=locked['development'])
        evaluate_official(pairs, ci, di, scale, margin, selection)
        return
    if borderline:
        test_borderline(features, 0. if zero_offset else gamma, seen_ok, unseen_ok, y, seen, rows,
                        to_unseen=to_unseen)
        return
    if official:
        all_images = np.ones(len(y), dtype=bool)
        target = correct(gap, all_images, gamma)[1]
        model = fit_gate(features, seen_ok, seen_ok | unseen_ok)
        scores = gate_score(features, model)
        cutoff = threshold(scores, all_images, target)
        gap64 = gap.astype(np.float64)
        retuned = threshold(gap64, all_images, target)
        choices = dict(current=gamma, retuned_gap=retuned, expert_correctness=cutoff)
        development = {name: correct(scores if name == 'expert_correctness' else gap64,
                                     all_images, value) for name, value in choices.items()}
        selection = dict(split=42, calibration_target=target, fit_answerable=int((seen_ok | unseen_ok).sum()),
                         features=['clip_seen_unseen_gap','convnext_seen_margin','clip_unseen_margin'],
                         model=model, cutoffs=choices, development_correct=development)
        write_json(HERE/'convnext_crop_clip_disease/expert_correctness_gate_selection.json', selection)
        print('Locked development settings', json.dumps(selection), flush=True)
        evaluate_official(pairs, ci, di, scale, margin, selection)
        return
    rng = np.random.default_rng(42)
    groups = {pair:i%5 for group in (y[seen], y[~seen])
              for i,pair in enumerate(rng.permutation(np.unique(group)))}
    folds = np.array([groups[pair] for pair in y])
    records = []
    for fold in range(5):
        test, cal = folds==fold, folds==(fold+1)%5
        fit_rows = ~(test | cal)
        assert set(y[test]).isdisjoint(y[~test])
        candidates = {
            'fixed_current': gap.astype(np.float64),
            'retuned_gap': gap.astype(np.float64),
            'group_label': fit_scores(features, seen, fit_rows),
            'expert_correctness': fit_scores(features, seen_ok, fit_rows & (seen_ok | unseen_ok)),
        }
        target = correct(gap,cal,gamma)[1]
        test_target = correct(gap,test,gamma)[1]
        record = {}
        for name, scores in candidates.items():
            cutoff = gamma if name=='fixed_current' else threshold(scores,cal,target)
            diagnostic = threshold(scores,test,test_target)
            record[name] = dict(locked=correct(scores,test,cutoff),
                                matched_unseen=correct(scores,test,diagnostic))
        records.append(record)
    totals = {mode:{arm:np.sum([r[arm][mode] for r in records],axis=0).tolist() for arm in records[0]}
              for mode in ('locked','matched_unseen')}
    out = HERE/'convnext_crop_clip_disease/expert_correctness_gate_development.json'
    write_json(out, dict(split=42, images=[len(vs),len(vu)], totals=totals, folds=records))
    print(json.dumps(totals), flush=True)


def evaluate_official(pairs, ci, di, scale, margin, selection):
    from dpd.clip_utils import clip_transforms
    from dpd.data import COLUMNS
    from dpd.resize_cache import ensure_resized
    from train import train_lora

    prior = HERE/'clip_disease_only/official'
    cfg = json.loads((BASE/'config/vlm/v10.3_text_anchor_three_splits_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    cfg['runtime']['num_workers'] = 0
    table = pd.DataFrame(pairs)
    si, ui = np.flatnonzero(~table.is_unseen.to_numpy()), np.flatnonzero(table.is_unseen.to_numpy())
    model, _ = train_lora.build(cfg, table, 'cuda')
    model.clip.visual.load_state_dict(torch.load(prior/'visual_encoder.pt', map_location='cpu', weights_only=True))
    state = torch.load(prior/'disease_plus_gan_head.pt', map_location='cpu', weights_only=True)
    head = CosineClassifier(state['weight'], scale).cuda().eval()
    _, tf = clip_transforms(cfg['input'], cfg['augmentation'])
    cnn_path = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'
    outputs = {}
    reverse = 'override_rule' in selection
    two_way = selection.get('overrides')
    with np.load(cnn_path) as cnn:
        for split in ('test_seen', 'test_unseen'):
            frame = pd.read_csv(DATA/f'DPD_Small_{split}.csv', header=None, names=COLUMNS)
            print(f'Encoding official {split}', flush=True)
            cache = ensure_resized(frame.file, cfg['input']['resize_cache'])
            loader = train_lora.loader(frame, cache, tf, cfg, shuffle=False)
            embeddings, _ = train_lora.encode(model, loader, 'cuda', cfg['runtime']['amp'],
                                              tuple(cfg['evaluation'].get('tta', ['identity'])))
            with torch.inference_mode():
                clip = head(torch.as_tensor(embeddings, device='cuda')).cpu().numpy()
            crop = cnn[f'{split}_crop']
            clip_s = crop[:,ci[si]] + clip[:,di[si]]
            clip_u = crop[:,ci[ui]] + clip[:,di[ui]]
            cnn_s = crop[:,ci[si]] + cnn[f'{split}_disease'][:,di[si]]
            gap = (clip_s.max(1) - clip_u.max(1)).astype(np.float64)
            features = np.column_stack((gap, margin(cnn_s), margin(clip_u)))
            with np.load(prior/f'{split}.npz') as bank:
                assert np.array_equal(bank['filenames'], frame.file.to_numpy())
                assert np.array_equal(bank['labels'], frame.pair_id.to_numpy())
                assert np.allclose(bank['gap'], gap, atol=1e-4)
                assert np.array_equal(bank['seen_pred'], si[cnn_s.argmax(1)])
                assert np.array_equal(bank['unseen_pred'], ui[clip_u.argmax(1)])
                rows = {}
                for name, cutoff in selection['cutoffs'].items():
                    scores = gate_score(features, selection['model']) if name == 'expert_correctness' else bank['gap']
                    route_seen = scores.astype(np.float64) >= cutoff
                    if name == 'reverse_override':
                        route_seen &= ~borderline_override(features, cutoff, selection['override_rule'],
                                                           to_unseen=True)
                    if two_way:
                        overrides = two_way[name]
                        route_seen |= borderline_override(features, cutoff, overrides['to_seen'])
                        route_seen &= ~borderline_override(features, cutoff, overrides['to_unseen'], to_unseen=True)
                    prediction = np.where(route_seen, bank['seen_pred'], bank['unseen_pred'])
                    rows[name] = dict(correct=int((prediction == bank['labels']).sum()),
                                      correct_route=int((route_seen == (split == 'test_seen')).sum()),
                                      images=len(frame))
            outputs[split] = rows
    summary = {}
    for name in selection['cutoffs']:
        a, b = (outputs[split][name]['correct'] / outputs[split][name]['images']
                for split in ('test_seen', 'test_unseen'))
        summary[name] = dict(seen=a, unseen=b, h=2*a*b/(a+b))
    if two_way:
        validation = [('gamma_3_current',1336,1199)]
        if not selection.get('joint'):
            validation += [('gamma_0_current',1458,700), ('gamma_0_previous_reverse',1452,777)]
        for key, a, b in validation:
            assert outputs['test_seen'][key]['correct']==a and outputs['test_unseen'][key]['correct']==b
    else:
        assert outputs['test_seen']['current']['correct'] == (1458 if reverse else 1336)
        assert outputs['test_unseen']['current']['correct'] == (700 if reverse else 1199)
    report = dict(selection=selection, official=summary, counts=outputs)
    name = ('two_way_override_official.json' if two_way else 'borderline_to_unseen_zero_official.json'
            if reverse else 'expert_correctness_gate_official.json')
    if selection.get('joint'):
        name = 'joint_override_official.json'
    write_json(HERE/'convnext_crop_clip_disease'/name, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--official', action='store_true')
    modes.add_argument('--borderline', action='store_true')
    modes.add_argument('--reverse-official', action='store_true', help='Evaluate the saved zero-offset reverse override.')
    modes.add_argument('--two-way', action='store_true', help='Fit and evaluate both override directions.')
    modes.add_argument('--joint', action='store_true', help='Tune both override directions and the base offset.')
    parser.add_argument('--zero-offset', action='store_true', help='Use a zero base cutoff for the borderline experiment.')
    parser.add_argument('--to-unseen', action='store_true', help='Try moving borderline seen routes to unseen.')
    args = parser.parse_args()
    if args.zero_offset and not args.borderline:
        parser.error('--zero-offset requires --borderline')
    if args.to_unseen and not args.zero_offset:
        parser.error('--to-unseen requires --zero-offset')
    main(args.official, args.borderline, args.zero_offset, args.to_unseen, args.reverse_official,
         args.two_way, args.joint)
