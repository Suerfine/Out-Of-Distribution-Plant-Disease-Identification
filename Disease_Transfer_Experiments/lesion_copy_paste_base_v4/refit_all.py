"""Full-data copy-paste continuation of the saved full-data baseline."""
import csv
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from train_full import HERE, DETAIL, Images, IndependentNet, optimizer, transforms, seed_all, status, write_json, sha256
from storage import CheckpointStore, checked_save


def main():
    prior = HERE/'full_run'
    run = HERE/'full_data_refit'
    run.mkdir(exist_ok=True)
    dev_test = prior/'official_test_once/results.json'
    status(run,'waiting_for_development_evaluation')
    deadline = time.monotonic()+600
    while not dev_test.is_file():
        if time.monotonic()>deadline:
            raise RuntimeError('Development evaluation did not produce results')
        time.sleep(10)
    history = json.loads((prior/'history.json').read_text())
    assert len(history)==15
    epochs = max(history,key=lambda r:r['harmonic_mean'])['epoch']
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    project = HERE.parents[1]/'Out-Of-Distribution-Plant-Disease-Identification'
    baseline = project/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/disease_net.pt'
    manifest = next(Path(k) for k in p['reference_hashes'] if Path(k).name=='all_train.csv')
    assert sha256(manifest)==p['reference_hashes'][str(manifest)]
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        real = [{'path':str(project/'runs/cache/resized_256'/(Path(r['file']).stem+'.jpg')),
                 'disease_id':int(r['disease_id'])} for r in csv.DictReader(f)]
    composites = HERE/'composites.json'
    synthetic = json.loads(composites.read_text())
    assert len(real)==8000 and len(synthetic)==1028
    assert len({r['path'] for r in real})==8000
    rows = real+synthetic
    assert all(Path(r['path']).is_file() for r in rows)
    digest = sha256(baseline)
    recipe = {'epochs':epochs,'originals':len(real),'composites':len(synthetic),
              'baseline_sha256':digest,'manifest_sha256':sha256(manifest),'composites_sha256':sha256(composites)}
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()+json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    write_json(run/'recipe.json',recipe)
    torch.set_num_threads(4)
    seed_all(42)
    base = p['baseline_config']
    model = IndependentNet(base,'disease')
    model.load_state_dict(torch.load(baseline,map_location='cpu',weights_only=True))
    model.cuda()
    model.backbone.set_grad_checkpointing(True)
    opt = optimizer(model,base)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    fit = DataLoader(Images(rows,transforms(base)[0]),batch_size=4,shuffle=True,num_workers=2)
    store = CheckpointStore(run,fingerprint,6*1024**3)
    saved = store.load()
    done = 0
    if saved is not None:
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        done = saved['epoch']
        del saved
    for epoch in range(done+1,epochs+1):
        seed_all(42+epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        status(run,'training',epoch=epoch,epochs=epochs,step=0,total_steps=len(fit))
        for step,(x,y) in enumerate(fit,1):
            samples = min(32,len(rows)-((step-1)//8)*32)
            with torch.autocast('cuda',dtype=torch.float16):
                loss = criterion(model(x.cuda()),y.cuda())*len(y)/samples
            scaler.scale(loss).backward()
            if step%8==0 or step==len(fit):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step%100==0:
                status(run,'training',epoch=epoch,epochs=epochs,step=step,total_steps=len(fit))
        store.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch,'cursor':0,'stage':'finetune','fingerprint':fingerprint})
        if (run/'STOP.request').exists():
            status(run,'paused',epoch=epoch)
            return
    checked_save(run/'final.pt',{'model':model.state_dict(),'epoch':epochs},6*1024**3)
    status(run,'complete',epochs=epochs)


if __name__=='__main__':
    main()

