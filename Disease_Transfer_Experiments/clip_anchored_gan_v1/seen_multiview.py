"""Select disease-only test-time views on seen development photos."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from seen_expert import HERE, DETAIL, IndependentNet, transforms, sha256, write_json

ROOT = HERE.parents[1]
BASE = ROOT/'Out-Of-Distribution-Plant-Disease-Identification'
REF = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42'
GROUPS = {'flip':('flip',),'wide':('wide',),'local':('top_left','top_right','bottom_left','bottom_right')}
RECIPES = ((), ('flip',), ('wide',), ('local',), ('flip','wide'), ('wide','local'), ('flip','wide','local'))


class Views(Dataset):
    def __init__(self, paths, cfg, views):
        self.paths, self.views = paths, views
        self.size = cfg['input']['crop_size']
        self.standard = transforms(cfg)[1]
        self.tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32,scale=True),
                                 T.Normalize(cfg['input']['mean'],cfg['input']['std'])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        with Image.open(self.paths[index]) as source:
            im = source.convert('RGB')
        out = []
        for view in self.views:
            if view == 'flip':
                x = self.standard(ImageOps.mirror(im))
            elif view == 'wide':
                x = self.tensor(ImageOps.pad(im,(self.size,self.size),Image.Resampling.BICUBIC,color=(124,116,104)))
            else:
                canvas = ImageOps.fit(im,(256,256),Image.Resampling.BICUBIC)
                side = 208
                x0 = 256-side if view.endswith('right') else 0
                y0 = 256-side if view.startswith('bottom') else 0
                crop = canvas.crop((x0,y0,x0+side,y0+side)).resize((self.size,self.size),Image.Resampling.BICUBIC)
                x = self.tensor(crop)
            out.append(x)
        return torch.stack(out)


def inference(paths,cfg,source,groups):
    keys = [key for key in GROUPS if key in groups]
    views = [view for key in keys for view in GROUPS[key]]
    model = IndependentNet(cfg,'disease')
    state = torch.load(source,map_location='cpu',weights_only=True)
    model.load_state_dict(state.get('model_state',state))
    del state
    model.cuda().eval()
    loader = DataLoader(Views(paths,cfg,views),batch_size=8,num_workers=0)
    chunks = []
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for step,batch in enumerate(loader,1):
            logits = model(batch.flatten(0,1).cuda()).float().reshape(len(batch),len(views),-1)
            chunks.append(logits.cpu().numpy())
            if step%40==0:
                print(f'Scored {min(step*8,len(paths))}/{len(paths)} photos',flush=True)
    result = np.concatenate(chunks)
    assert np.isfinite(result).all()
    offset = 0
    grouped = {}
    for key in keys:
        stop = offset+len(GROUPS[key])
        grouped[key] = result[:,offset:stop].mean(1)
        offset = stop
    return grouped


def mix(center, estimates, recipe):
    return np.mean([center]+[estimates[g] for g in recipe],axis=0)


def main():
    torch.set_num_threads(4)
    out = HERE/'seen_multiview'
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    cfg = p['baseline_config']
    fit, val = (p['roles'][k] for k in ('fit','validation_seen'))
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[key] for r in pairs]) for key in ('crop_id','disease_id')]
    dev_pairs = np.array(sorted({r['composition_id'] for r in fit}))
    y = np.array([r['composition_id'] for r in val])
    assert len(val)==862 and len(dev_pairs)==303
    source = next(Path(k) for k in p['reference_hashes'] if Path(k).parent.name=='disease' and Path(k).name=='best_epoch_15.pt')
    for path in (source,Path(p['baseline_validation_logits'])):
        assert sha256(path)==p['reference_hashes'][str(path)]
    with np.load(p['baseline_validation_logits']) as b:
        crop,disease = b['crop'][:len(val)],b['disease'][:len(val)]
    estimates = inference([r['legacy_cache_path'] for r in val],cfg,source,GROUPS)
    grid = []
    for recipe in RECIPES:
        scores = crop[:,ci[dev_pairs]] + mix(disease,estimates,recipe)[:,di[dev_pairs]]
        grid.append({'groups':recipe,'correct':int((dev_pairs[scores.argmax(1)]==y).sum())})
    assert grid[0]['correct']==773
    chosen = max(grid,key=lambda row:row['correct'])
    selection = {'grid':grid,'selected':chosen,'images':len(val)}
    write_json(out/'selection.json',selection)
    print('Development',selection,flush=True)

    candidates = np.array([r['pair_id'] for r in pairs if not r['is_unseen']])
    with (ROOT/'Plant_Disease/DPD_Small_test_seen.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.reader(f))
    with np.load(HERE/'seen_expert_official/test_seen.npz') as b:
        names,y,old,routed = (b[key] for key in ('filenames','labels','old','routed_seen'))
    assert np.array_equal(names,[r[0] for r in rows]) and np.array_equal(y,[int(r[3]) for r in rows])
    with np.load(REF/'factor_logits.npz') as b:
        crop,disease = b['test_seen_crop'][routed],b['test_seen_disease'][routed]
    assert np.array_equal(candidates[(crop[:,ci[candidates]]+disease[:,di[candidates]]).argmax(1)],old[routed])
    recipe = chosen['groups']
    if recipe:
        cache = BASE/'runs/cache/resized_256'
        paths = [cache/(Path(name).stem+'.jpg') for name in names[routed]]
        estimates = inference(paths,cfg,REF/'disease_net.pt',recipe)
    else:
        estimates = {}
    new = old.copy()
    score = crop[:,ci[candidates]] + mix(disease,estimates,recipe)[:,di[candidates]]
    new[routed] = candidates[score.argmax(1)]
    with np.load(HERE/'seen_expert_official/test_unseen.npz') as b:
        unseen = int((b['labels']==b['old']).sum())
        nu = len(b['labels'])
    assert np.array_equal(old[~routed],new[~routed]) and int((old==y).sum())==1338 and unseen==1134
    official = {'old_seen_correct':int((old==y).sum()),'new_seen_correct':int((new==y).sum()),
                'seen_images':len(y),'unseen_correct':unseen,'unseen_images':nu,
                'fixes':int(((new==y)&(old!=y)).sum()),'regressions':int(((new!=y)&(old==y)).sum())}
    np.savez_compressed(out/'test_seen.npz',filenames=names,labels=y,old=old,new=new,routed_seen=routed)
    write_json(out/'results.json',{'selection':selection,'official':official})
    print('Official',official,flush=True)


if __name__=='__main__':
    main()
