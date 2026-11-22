"""Evaluate the validation-selected checkpoint on the two official splits."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_full import HERE, DETAIL, Images, IndependentNet, transforms, sha256, write_json
from dpd.cnn import compose_pair_scores


def main():
    run = HERE/'full_data_refit'
    output = run/'official_test_once'
    output.mkdir(exist_ok=True)
    if (output/'results.json').exists():
        return
    while True:
        trainer = json.loads((run/'supervisor_status.json').read_text())
        if trainer['state'] == 'complete':
            break
        if trainer['state'] != 'running':
            raise RuntimeError(f"Training ended with state {trainer['state']}")
        time.sleep(30)
    best = {'epoch':json.loads((run/'recipe.json').read_text())['epochs'], 'harmonic_mean':None}
    checkpoint = run/'final.pt'
    digest = sha256(checkpoint)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['epoch'] == best['epoch']
    write_json(output/'selection.json',{'epoch':best['epoch'],'sha256':digest,'validation_h':best['harmonic_mean']})
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    model = IndependentNet(p['baseline_config'],'disease')
    model.load_state_dict(saved['model'])
    model.cuda().eval()
    _, transform = transforms(p['baseline_config'])
    root = HERE.parents[1]
    baseline = root/'Out-Of-Distribution-Plant-Disease-Identification/runs/baseline/independent_cnn_baseline_convnext_base_seed42'
    cache = baseline.parents[1]/'cache/resized_256'
    pairs = pd.DataFrame(p['pairs'])
    assert pairs.pair_id.tolist() == list(range(333))
    result = {'epoch':best['epoch'],'checkpoint_sha256':digest,'gamma':0}
    with np.load(baseline/'factor_logits.npz') as factors, np.load(baseline/'scores.npz') as reference:
        for split,count in [('test_seen',2000),('test_unseen',1448)]:
            with (root/f'Plant_Disease/DPD_Small_{split}.csv').open(encoding='utf-8-sig',newline='') as file:
                rows = list(csv.reader(file))
            labels = np.array([int(r[3]) for r in rows])
            assert len(rows)==count and np.array_equal(labels, reference[f'{split}_labels'])
            inputs = [{'path':str(cache/(Path(r[0]).stem+'.jpg')),'disease_id':int(r[2])} for r in rows]
            assert all(Path(r['path']).is_file() for r in inputs)
            crop = torch.from_numpy(factors[f'{split}_crop'])
            old_disease = torch.from_numpy(factors[f'{split}_disease'])
            def rank(disease):
                return compose_pair_scores(crop.log_softmax(1).numpy(),disease.log_softmax(1).numpy(),pairs).argmax(1)
            old = rank(old_disease)
            expected = {'test_seen':1476,'test_unseen':203}
            assert int((old==labels).sum()) == expected[split]
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                disease = torch.cat([model(x.cuda()).float().cpu() for x,_ in
                                     DataLoader(Images(inputs,transform),batch_size=8,num_workers=2)])
            assert torch.isfinite(disease).all()
            pred = rank(disease)
            np.savez_compressed(output/f'{split}.npz',filenames=[r[0] for r in rows],labels=labels,
                                disease=disease.numpy(),predictions=pred,baseline_predictions=old)
            result[split] = {'images':count,'correct':int((pred==labels).sum()),'baseline_correct':expected[split]}
            print(split,result[split],flush=True)
    write_json(output/'results.json',result)


if __name__ == '__main__':
    main()

