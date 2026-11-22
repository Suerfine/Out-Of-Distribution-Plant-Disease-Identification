"""Compare frozen CLIP and SigLIP with matched development GAN and gate fitting."""
from copy import deepcopy
import argparse
import json

import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import sentencepiece as sp
import torch
from torch.nn import functional as F

from expert_correctness_gate import DATA, DETAIL, HERE, borderline_override, two_way_selection, write_json
from open_clip.tokenizer import basic_clean, canonicalize_text
from study import (
    COLUMNS, PROJECT, build_anchor_pool, ensure_resized, fit_classifier,
    seed_everything, text_conditions, train_generator,
)


class CachedImages(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert('RGB'))


def tokenizer_for_siglip():
    processor = sp.SentencePieceProcessor(model_file=str(HERE/'SigLIP-c4-en.model'))
    assert processor.vocab_size() == 32000 and processor.eos_id() == 1

    def tokenize(texts):
        if isinstance(texts, str):
            texts = [texts]
        ids = [processor.encode(canonicalize_text(basic_clean(text)))[:63] for text in texts]
        return torch.tensor([row + [1]*(64-len(row)) for row in ids], dtype=torch.long)

    return tokenize


def fit_head(name, spec, rows, fit, pairs, config, out, unseen_ids):
    architecture, weights = spec
    opts = {}
    if name == 'siglip':
        p = open_clip.get_pretrained_cfg(architecture,'webli')
        opts = {f'image_{k}':p[k] for k in ('mean','std','interpolation','resize_mode')}
    seed_everything(42)
    model, _, preprocess = open_clip.create_model_and_transforms(architecture,pretrained=str(weights),**opts)
    model.cuda().eval()
    tokenizer = tokenizer_for_siglip() if name == 'siglip' else open_clip.get_tokenizer(architecture)
    prompt, conditions = [torch.as_tensor(z,device='cuda') for z in
                          text_conditions(config,model,tokenizer,pairs,'cuda')]
    disease_cfg = deepcopy(config)
    disease_cfg['condition']['parts'] = ['disease']
    _, by_pair = text_conditions(disease_cfg,model,tokenizer,pairs,'cuda')
    cache = ensure_resized(rows.file,256)
    paths = [cache/(file.rsplit('.',1)[0]+'.jpg') for file in rows.file]
    loader = torch.utils.data.DataLoader(CachedImages(paths,preprocess),batch_size=32)
    print(f'{name}: encoding {len(rows)} training and validation images',flush=True)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        x_all = torch.cat([F.normalize(model.encode_image(images.cuda()).float(),dim=-1) for images in loader])
    del model
    torch.cuda.empty_cache()

    di = pairs.disease_id.to_numpy()
    disease_ids, first = np.unique(di,return_index=True)
    assert np.array_equal(disease_ids,np.arange(175))
    x = x_all[:len(fit)]
    y = torch.as_tensor(fit.pair_id.to_numpy(copy=True),device='cuda')
    trained = torch.zeros(len(pairs),device='cuda',dtype=torch.bool)
    trained[y.unique()] = True
    assert not trained[unseen_ids].any()
    aux_cfg = config['aux_classifier']
    seed_everything(42)
    aux = fit_classifier(x,y,trained,prompt,aux_cfg,aux_cfg['epochs'],tag=f'{name}_aux')
    anchors, _ = build_anchor_pool(x,fit,pairs)
    gan = train_generator(x,y,conditions,aux,config['gan'],anchors)
    synthetic_pairs = torch.as_tensor(unseen_ids,device='cuda').repeat_interleave(config['synthesis']['n_per_unseen'])
    with torch.no_grad():
        synthetic = gan(conditions[synthetic_pairs],synthetic_pairs)
    assert torch.isfinite(synthetic).all()
    disease = torch.as_tensor(di,device='cuda')
    seed_everything(42)
    head = fit_classifier(torch.cat((x,synthetic)),torch.cat((disease[y],disease[synthetic_pairs])),
                          torch.ones(175,device='cuda',dtype=torch.bool),
                          torch.as_tensor(by_pair[first],device='cuda'),config['classifier'],1,
                          tag=f'{name}_disease')
    torch.save(head.state_dict(),out/f'{name}_head.pt')
    with torch.inference_mode():
        return head(x_all[len(fit):]).cpu().numpy()


def margin(scores):
    top = np.partition(scores,-2,axis=1)[:,-2:]
    return top[:,1]-top[:,0]


def gate_inputs(crop, cnn, logits, ci, di, si, ui, labels):
    cnn_seen = crop[:,ci[si]]+cnn[:,di[si]]
    seen_scores = crop[:,ci[si]]+logits[:,di[si]]
    unseen_scores = crop[:,ci[ui]]+logits[:,di[ui]]
    gap = seen_scores.max(1)-unseen_scores.max(1)
    features = np.column_stack((gap,margin(cnn_seen),margin(unseen_scores)))
    return features, si[cnn_seen.argmax(1)]==labels, ui[unseen_scores.argmax(1)]==labels


def hm(counts, n_seen, n_unseen):
    a,b = counts[0]/n_seen, counts[1]/n_unseen
    return 2*a*b/(a+b) if a+b else 0.


def evaluate_official(out, pairs, config, train, encoder):
    dev = json.loads((out/'development_results.json').read_text())['results'][encoder]
    dev_settings = {'zero':(0.,dev['zero_offset']), 'tuned':(dev['tuned_gamma'],dev['tuned_offset'])}
    result_dir = out/('official' if encoder == 'siglip' else 'official_frozen_clip')
    result_dir.mkdir(exist_ok=True)
    test = [pd.read_csv(DATA/f'DPD_Small_{split}.csv',header=None,names=COLUMNS)
            for split in ('test_seen','test_unseen')]
    si, ui = np.sort(train.pair_id.unique()), np.flatnonzero(pairs.is_unseen.to_numpy())
    assert len(train)==8000 and len(si)==323 and len(ui)==10
    rows = pd.concat((train,*test),ignore_index=True)
    assert rows.file.is_unique
    score_file = result_dir/f'{encoder}_scores.npy'
    if score_file.exists():
        logits = np.load(score_file)
    else:
        spec = (('ViT-B-16-SigLIP',HERE/'ViT-B-16-SigLIP.npz') if encoder == 'siglip'
                else ('ViT-B-16-quickgelu',HERE/'ViT-B-16-state.pt'))
        logits = fit_head(encoder,spec,rows,train,pairs,config,result_dir,ui)
        np.save(score_file,logits)
    cnn_path = PROJECT/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'
    with np.load(cnn_path) as bank:
        crop, cnn = [np.concatenate([bank[f'{split}_{factor}'] for split in ('test_seen','test_unseen')])
                     for factor in ('crop','disease')]
    for split, frame in zip(('test_seen','test_unseen'),test):
        with np.load(HERE/'clip_disease_only/official'/f'{split}.npz') as bank:
            assert np.array_equal(bank['filenames'],frame.file.to_numpy())
            assert np.array_equal(bank['labels'],frame.pair_id.to_numpy())
    ci,di = pairs.crop_id.to_numpy(),pairs.disease_id.to_numpy()
    labels = rows.pair_id.to_numpy()[len(train):]
    seen = np.arange(len(labels))<len(test[0])
    features, seen_ok, unseen_ok = gate_inputs(crop,cnn,logits,ci,di,si,ui,labels)
    result = {}
    for name,(gamma,rules) in dev_settings.items():
        result[name] = {}
        for arm in ('current','to_unseen','two_way'):
            rule = rules[arm]
            route = ((features[:,0]>=gamma) | borderline_override(features,gamma,rule['to_seen']))
            route &= ~borderline_override(features,gamma,rule['to_unseen'],to_unseen=True)
            counts = [int((route&seen_ok).sum()),int((~route&unseen_ok).sum())]
            result[name][arm] = dict(gamma=gamma, to_seen=rule['to_seen'], to_unseen=rule['to_unseen'],
                                     counts=counts, h=hm(counts,*map(len,test)))
    hit = logits.argmax(1)==di[labels]
    report = dict(images=list(map(len,test)), disease_correct=[int(hit[seen].sum()),int(hit[~seen].sum())],
                  results=result)
    write_json(result_dir/'results.json',report)
    print(json.dumps(report),flush=True)


def main(official=False, encoder='siglip'):
    torch.set_num_threads(4)
    out = HERE/'siglip_gate_pilot'
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = pd.DataFrame(p['pairs']).sort_values('pair_id').reset_index(drop=True)
    train = pd.read_csv(DATA/'DPD_Small_train.csv',header=None,names=COLUMNS).set_index('file',drop=False)
    config = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    if official:
        evaluate_official(out,pairs,config,train,encoder)
        return
    fit, vs, vu = [train.loc[[r['filename'] for r in p['roles'][part]]]
                   for part in ('fit','validation_seen','validation_unseen')]
    rows = pd.concat((fit,vs,vu),ignore_index=True)
    assert rows.file.is_unique and len(fit)==5798 and len(vu)==533
    ci, di = pairs.crop_id.to_numpy(),pairs.disease_id.to_numpy()
    si, ui = np.sort(fit.pair_id.unique()),np.sort(vu.pair_id.unique())
    labels = rows.pair_id.to_numpy()[len(fit):]
    seen = np.arange(len(labels)) < len(vs)
    with np.load(p['baseline_validation_logits']) as bank:
        crop, cnn = bank['crop'],bank['disease']
    results = {}
    for name, architecture, filename in (
        ('siglip','ViT-B-16-SigLIP','ViT-B-16-SigLIP.npz'),
        ('clip','ViT-B-16-quickgelu','ViT-B-16-state.pt'),
    ):
        score_file = out/f'{name}_development_scores.npy'
        if score_file.exists():
            logits = np.load(score_file)
        else:
            logits = fit_head(name,(architecture,HERE/filename),rows,fit,pairs,config,out,ui)
            np.save(score_file,logits)
        assert logits.shape==(len(labels),175) and np.isfinite(logits).all()
        features, seen_ok, unseen_ok = gate_inputs(crop,cnn,logits,ci,di,si,ui,labels)
        gap = features[:,0]
        candidates = np.arange(0.,10.01,.1)
        gamma = max(candidates,key=lambda g: hm(((seen_ok&(gap>=g)).sum(),(unseen_ok&(gap<g)).sum()),
                                                len(vs),len(vu)))
        disease_hit = logits.argmax(1)==di[labels]
        results[name] = dict(standalone_disease=[int(disease_hit[seen].sum()),int(disease_hit[~seen].sum())],
                             unseen_expert_ceiling=int(unseen_ok.sum()),
                             zero_offset=two_way_selection(features,0.,seen_ok,unseen_ok,labels,seen),
                             tuned_gamma=float(gamma),
                             tuned_offset=two_way_selection(features,gamma,seen_ok,unseen_ok,labels,seen))
        write_json(out/'development_results.json',dict(images=[len(vs),len(vu)],results=results))
        print(name, json.dumps({k:v['held_out'] for k,v in results[name]['zero_offset'].items()}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official', action='store_true')
    parser.add_argument('--encoder', choices=('siglip','clip'), default='siglip')
    args = parser.parse_args()
    main(args.official,args.encoder)
