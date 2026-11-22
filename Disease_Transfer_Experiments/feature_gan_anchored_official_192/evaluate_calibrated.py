"""Score saved heads at the baseline's fixed gamma without fitting on test."""
from train import HERE, DETAIL, IndependentNet, transforms, torch, np, json, write_json

from classifier_multi import Images

def main():
    baseline = HERE.parents[1]/'Out-Of-Distribution-Plant-Disease-Identification/runs/baseline/independent_cnn_baseline_convnext_base_seed42'
    cache = baseline.parents[1]/'cache/resized_256'
    cfg = json.loads((baseline/'config.json').read_text())
    pairs = sorted(json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())['pairs'],key=lambda r:r['pair_id'])
    ci,di = [[r[k] for r in pairs] for k in ('crop_id','disease_id')]
    offset = 6.2*np.array([not r['is_unseen'] for r in pairs])
    torch.set_num_threads(4)
    net = IndependentNet(cfg,'disease')
    net.load_state_dict(torch.load(baseline/'disease_net.pt',weights_only=True,map_location='cpu'))
    net.cuda().eval()
    _,transform = transforms(cfg)
    out = HERE/'calibrated'
    out.mkdir(exist_ok=True)
    result = {}
    with np.load(baseline/'factor_logits.npz') as factors:
        for split in ('test_seen','test_unseen'):
            paths = [HERE/'runs'/f'seed_{seed}'/f'{split}.npz' for seed in (42,43,44)]
            with np.load(paths[0]) as bank:
                names,labels = bank['filenames'],bank['labels']
            rows = [{'path':str(cache/(HERE.joinpath(name).stem+'.jpg')),'disease_id':int(pairs[label]['disease_id'])} for name,label in zip(names,labels)]
            loader = torch.utils.data.DataLoader(Images(rows,transform),batch_size=16,num_workers=0)
            with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
                x = torch.cat([net.backbone(images.cuda()).float().cpu() for images,_ in loader])
            crop = factors[f'{split}_crop'][:,ci]
            def scores(logits):
                return crop+logits[:,di]
            def correct(value):
                return int(((value-offset).argmax(1)==labels).sum())
            reference = scores(factors[f'{split}_disease'])
            split_result = {'images':len(labels),'baseline':correct(reference),'gan':[],'real_only':[]}
            for path in paths:
                with np.load(path) as bank:
                    assert np.array_equal(names,bank['filenames']) and np.array_equal(labels,bank['labels'])
                    assert np.array_equal(reference.argmax(1),bank['baseline'])
                    for arm in ('gan','real_only'):
                        net.head.load_state_dict(torch.load(path.parent/f'{arm}_head.pt',weights_only=True,map_location='cpu'))
                        with torch.inference_mode():
                            value = scores(net.head(x.cuda()).cpu().numpy())
                        assert np.array_equal(value.argmax(1),bank[arm])
                        split_result[arm].append(correct(value))
            result[split] = split_result
            print(split,split_result,flush=True)
    write_json(out/'results.json',{'gamma':6.2,'seeds':[42,43,44],**result})

if __name__=='__main__':
    main()

