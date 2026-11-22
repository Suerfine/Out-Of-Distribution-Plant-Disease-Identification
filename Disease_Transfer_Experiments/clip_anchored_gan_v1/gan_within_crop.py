"""Test same-crop disease supervision in the crop-anchored CLIP GAN."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch

from crop_context_gate import build
from expert_correctness_gate import DATA, DETAIL, HERE, gate_score, write_json
from study import (
    COLUMNS, PROJECT, build_anchor_pool, fit_classifier, load_clip, seed_everything,
    text_conditions, train_generator,
)
from dpd.feature_gen import CosineClassifier


def main():
    torch.set_num_threads(4)
    out = HERE/'gan_within_crop'
    out.mkdir(exist_ok=True)
    prepared = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = pd.DataFrame(prepared['pairs']).sort_values('pair_id').reset_index(drop=True)
    ci,di = pairs.crop_id.to_numpy(),pairs.disease_id.to_numpy()
    cfg = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    train = pd.read_csv(DATA/'DPD_Small_train.csv',header=None,names=COLUMNS)
    train['row'] = np.arange(len(train))
    index = train.set_index('file',drop=False)
    fit,vs,vu = [index.loc[[r['filename'] for r in prepared['roles'][part]]]
                 for part in ('fit','validation_seen','validation_unseen')]
    validation = pd.concat((vs,vu),ignore_index=True)
    assert fit.index.is_unique and not set(fit.file).intersection(validation.file)
    si,ui = np.unique(fit.pair_id),np.unique(vu.pair_id)
    ids = np.union1d(si,ui)
    assert len(ui)==20 and not np.intersect1d(si,ui).size
    with np.load(HERE/'runs/seed_42/features.npz') as b:
        assert np.array_equal(b['filenames'],train.file.to_numpy())
        features = b['embeddings']
    prior = HERE/'clip_disease_only'
    with np.load(prior/'synthetic.npz') as b:
        old_ids = b['pair_ids']
    synthetic_ids = np.repeat(ui,cfg['synthesis']['n_per_unseen'])
    assert np.array_equal(old_ids,synthetic_ids)
    gan_weight = cfg['gan']['cls_weight']
    model,_,tokenizer = load_clip(cfg['model'],'cuda')
    prompt,condition = [torch.as_tensor(z,device='cuda') for z in text_conditions(cfg,model,tokenizer,pairs,'cuda')]
    disease_cfg = deepcopy(cfg)
    disease_cfg['condition']['parts'] = ['disease']
    _,by_pair = text_conditions(disease_cfg,model,tokenizer,pairs,'cuda')
    del model
    _,classes = np.unique(di,return_index=True)
    x = torch.as_tensor(features[fit.row.to_numpy()],device='cuda')
    y = torch.as_tensor(fit.pair_id.to_numpy(copy=True),device='cuda')
    trained = torch.zeros(len(pairs),device='cuda',dtype=torch.bool)
    trained[si] = True
    same_crop = torch.as_tensor(ci[:,None]==ci[None,:],device='cuda') & trained[None,:]
    assert same_crop[y,y].all()
    anchors,_ = build_anchor_pool(x,fit,pairs)
    seed_everything(42)
    aux_cfg = cfg['aux_classifier']
    aux = fit_classifier(x,y,trained,prompt,aux_cfg,aux_cfg['epochs'],tag='aux')
    opts = dict(cfg['gan'],within_crop_weight=gan_weight)
    generate = train_generator(x,y,condition,aux,opts,anchors,within_crop_mask=same_crop)
    sy = torch.as_tensor(synthetic_ids,device='cuda')
    with torch.no_grad():
        synthetic = generate(condition[sy],sy)
    assert torch.isfinite(synthetic).all()
    seed_everything(42)
    disease = torch.as_tensor(di,device='cuda')
    head = fit_classifier(torch.cat((x,synthetic)),torch.cat((disease[y], disease[sy])),
                          torch.ones(175,dtype=torch.bool,device='cuda'),
                          torch.as_tensor(by_pair[classes],device='cuda'),
                          cfg['classifier'],1,tag='within_crop')
    torch.save(head.state_dict(),out/'disease_head.pt')
    xv = torch.as_tensor(features[validation.row.to_numpy()],device='cuda')
    original = torch.load(prior/'disease_plus_gan_head.pt',map_location='cpu',weights_only=True)['weight']
    baseline = CosineClassifier(original,cfg['classifier']['logit_scale']).cuda()
    with torch.inference_mode():
        scores = dict(current=baseline(xv).cpu().numpy(),within_crop=head(xv).cpu().numpy())
    with np.load(prepared['baseline_validation_logits']) as b:
        crop,cnn = b['crop'],b['disease']
    y = validation.pair_id.to_numpy()
    seen = np.arange(len(y))<len(vs)
    gate = json.loads((prior/'official/crop_context_gate.json').read_text())['selection']['models']['crop_context']
    cutoff = json.loads((prior/'official/crop_context_calibration.json').read_text())['development_selected']['cutoff']
    report = {}
    routing = {}
    for name,logits in scores.items():
        cp = ci[y]
        permitted = ci[ids][None,:]==cp[:,None]
        local = np.where(permitted,logits[:,di[ids]],-np.inf)
        hit = ids[local.argmax(1)]==y
        multi = permitted.sum(1)>1
        all_hit = logits.argmax(1)==di[y]
        f,s,u = build(crop,cnn,logits,ci,di,si,ui,y)
        score = gate_score(f['crop_context'],gate)
        routing[name] = (score,s,u)
        route = score>=cutoff
        report[name] = dict(true_crop_correct=[int((m&hit).sum()) for m in (seen,~seen)],
                            ambiguous_true_crop_correct=[int((m&multi&hit).sum()) for m in (seen,~seen)],
                            ambiguous_images=[int((m&multi).sum()) for m in (seen,~seen)],
                            disease_only_correct=[int((m&all_hit).sum()) for m in (seen,~seen)],
                            learned_gate_correct=[int((route&s).sum()),int((~route&u).sum())])
    floor = report['current']['learned_gate_correct'][0]
    assert report['current']['learned_gate_correct']==[682,315]
    for name,(z,s,u) in routing.items():
        eligible = np.sort(z[s])
        threshold = float(eligible[-floor])
        route = z>=threshold
        report[name]['matched_seen_diagnostic'] = [int((route&s).sum()),int((~route&u).sum())]
    result = dict(images=[len(vs),len(vu)],within_crop_weight=gan_weight,
                  vectors_per_pair=cfg['synthesis']['n_per_unseen'],results=report)
    write_json(out/'development_results.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
