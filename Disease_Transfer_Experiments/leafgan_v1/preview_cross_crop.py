import json
from PIL import Image
import torch
from torchvision.utils import save_image
from train import HERE, load_images, sha256
from models import networks

def main():
    p=json.loads((HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json').read_text())
    rows=[r for r in p['roles']['fit'] if r['crop_name']=='papaya' and r['disease_name']=='healthy']
    assert len(rows)==4
    run=HERE/'multi_pair/tomato_bacterial_spot/generator'
    pointer=json.loads((run/'checkpoint_pointer.json').read_text())['active']
    path=run/pointer['file']
    assert pointer['epoch']==60 and sha256(path)==pointer['sha256']
    model=networks.define_G(3,3,64,'resnet_9blocks','instance',False,'normal',.02,[0]).eval()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['models']['G_A'])
    x=torch.stack(load_images(rows,256)).cuda()
    with torch.inference_mode():
        y=torch.cat([model(im[None]) for im in x])
    assert torch.isfinite(y).all()
    out=HERE/'cross_crop_papaya'
    out.mkdir(exist_ok=True)
    save_image(torch.cat((x,y)).add(1).div(2).clamp(0,1).cpu(),out/'preview.jpg',nrow=4)
    for i,im in enumerate(y):
        save_image(im.add(1).div(2).clamp(0,1).cpu(),out/f'candidate_{i+1}.png')
    (out/'sources.json').write_text(json.dumps({'generator':pointer,'recipients':[r['filename'] for r in rows]},indent=2))

if __name__=='__main__':
    main()
