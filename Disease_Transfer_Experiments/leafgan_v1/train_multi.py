"""Tomato LeafGAN pilot using the authors' networks, objective, and LFLSeg weights."""
import argparse
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'cnn_detail_base_v1'))
from protocol import status, seed_all, sha256
from storage import CheckpointStore
sys.path.insert(0, str(HERE/'upstream'))
from models.leaf_gan_model import LeafGANModel
from models.grad_cam import GradCAM


def load_images(rows, size):
    transform = transforms.Compose([transforms.Resize((size,size),Image.Resampling.BICUBIC),
                                    transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    images = []
    for row in rows:
        with Image.open(row['original_path']) as im:
            images.append(transform(im.convert('RGB')))
    return images


def main(index, smoke=False):
    config = json.loads((HERE/"study.json").read_text(encoding="utf-8-sig"))
    spec = config["pairs"][index]
    slug = spec["crop"] + "_" + spec["disease"]
    seed_all(42)
    torch.set_num_threads(4)
    run = HERE/'multi_pair'/slug/('smoke' if smoke else 'generator')
    run.mkdir(exist_ok=True,parents=True)
    prepared = HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json'
    rows = json.loads(prepared.read_text())['roles']['fit']
    groups = [[r for r in rows if r['crop_name']==spec['crop'] and r['disease_name']==d
               and r['filename'].startswith(spec['source']+'_')] for d in ('healthy',spec['disease'])]
    rng = random.Random(42)
    for group in groups:
        rng.shuffle(group)
    n_preview = config["preview_per_pair"]
    preview = groups[0][:n_preview]
    groups[0] = groups[0][n_preview:]
    (run/"sources.json").write_text(json.dumps({"healthy":groups[0],"diseased":groups[1],"preview":preview},indent=2))
    size = 256
    weights = HERE/'LFLSeg_resnet101.pth'
    fingerprint = sha256(Path(__file__)) + sha256(weights) + sha256(prepared) + sha256(HERE/"study.json")
    store = CheckpointStore(run,fingerprint,2*1024**3)
    status(run,'preparing_masks',healthy=len(groups[0]),diseased=len(groups[1]))
    seg = models.resnet101()
    seg.fc = torch.nn.Linear(seg.fc.in_features,3)
    seg.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    seg.cuda().eval()
    cam = GradCAM(seg)
    data = []
    for group in groups:
        samples = []
        for x in load_images(group[:4] if smoke else group, size):
            seg.zero_grad(set_to_none=True)
            attention_input = torch.nn.functional.interpolate(x[None].cuda(),size=(224,224),mode="bicubic",align_corners=False,antialias=True)
            cam.forward(attention_input)
            cam.backward(idx=0)
            heat = torch.tensor(cam.generate(target_layer='layer4.2'))[None,None]
            mask = torch.nn.functional.interpolate(heat,size=(size,size),mode='bilinear',align_corners=False)[0]
            mask = (mask >= .35).float().expand_as(x)
            if .01 < mask.mean() < .99:
                samples.append((x,mask))
        data.append(samples)
    status(run,"masks_ready",healthy=len(data[0]),diseased=len(data[1]))
    del cam,seg
    torch.cuda.empty_cache()
    if min(map(len,data)) < (1 if smoke else 18):
        status(run,'skipped_insufficient_masks',healthy=len(data[0]),diseased=len(data[1]))
        return
    A,B = data
    opt = SimpleNamespace(gpu_ids=[0],isTrain=True,checkpoints_dir=str(run),name='leafgan',
        preprocess='scale_width',dataset_mode='unaligned_masked',input_nc=3,output_nc=3,
        ngf=64,ndf=64,netG='resnet_9blocks',netD='basic',norm='instance',no_dropout=True,
        init_type='normal',init_gain=.02,n_layers_D=3,lambda_identity=.5,lambda_A=10.,
        lambda_B=10.,pool_size=50,gan_mode='lsgan',lr=.0002,beta1=.5,direction='AtoB')
    gan = LeafGANModel(opt)
    saved = store.load()
    completed = 0
    if saved:
        for name in gan.model_names:
            getattr(gan,'net'+name).load_state_dict(saved['models'][name])
        for optimizer, state in zip(gan.optimizers,saved['optimizers']):
            optimizer.load_state_dict(state)
        completed = saved['epoch']
    epochs = 1 if smoke else config["generator_epochs"]
    image_batch = torch.stack(load_images(preview,size)).cuda()
    status(run,'training',healthy=len(A),diseased=len(B),epochs=epochs)
    for epoch in range(completed+1,epochs+1):
        seed_all(42+epoch)
        lr = .0002*min(1.,(epochs+1-epoch)/30)
        for optimizer in gan.optimizers:
            for group in optimizer.param_groups:
                group['lr'] = lr
        order = list(range(max(len(A),len(B))))
        rng = random.Random(42+epoch)
        rng.shuffle(order)
        for step,index in enumerate(order,1):
            a,ma = A[index%len(A)]
            b,mb = rng.choice(B)
            gan.set_input({'A':a[None],'B':b[None],'mask_A':ma[None],'mask_B':mb[None],'A_paths':['tomato']})
            gan.optimize_parameters()
            if step == 1 or step%25 == 0:
                losses = gan.get_current_losses()
                if not all(torch.isfinite(torch.tensor(v)) for v in losses.values()):
                    raise RuntimeError('Nonfinite LeafGAN loss')
                status(run,'training',epoch=epoch,step=step,total_steps=len(order),losses=losses)
            if smoke:
                break
        with torch.inference_mode():
            generated = torch.cat([gan.netG_A(x[None]) for x in image_batch])
            save_image(torch.cat((image_batch,generated)).add(1).div(2).clamp(0,1),
                       run/f'preview_{epoch:03d}.jpg',nrow=8)
        store.save({'models':{name:getattr(gan,'net'+name).state_dict() for name in gan.model_names},
                    'optimizers':[o.state_dict() for o in gan.optimizers],
                    'fingerprint':fingerprint,'epoch':epoch,'cursor':0,'stage':'leafgan'})
        if (run/'STOP.request').exists():
            status(run,'paused',epoch=epoch)
            return
    status(run,'complete',epochs=epochs)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--pair',type=int,required=True)
    args = parser.parse_args()
    main(args.pair,args.smoke)




