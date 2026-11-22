import json, random, sys
from pathlib import Path
import torch
from torchvision.utils import save_image
from train import HERE, load_images, sha256
from models import networks

run=HERE/'tomato_bacterial_spot'
p=json.loads((HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json').read_text())
fit=p['roles']['fit']
rows=[r for r in fit if r['crop_name']=='tomato' and r['disease_name']=='healthy']
random.Random(42).shuffle(rows)
rows=rows[:8]
pointer=json.loads((run/'checkpoint_pointer.json').read_text())['active']
path=run/pointer['file']
assert pointer['epoch']==60 and sha256(path)==pointer['sha256']
net=networks.define_G(3,3,64,'resnet_9blocks','instance',False,'normal',.02,[0])
net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['models']['G_A'])
net.eval()
out=HERE/'synthetic'
out.mkdir(exist_ok=True)
label=next(r['disease_id'] for r in fit if r['crop_name']=='tomato' and r['disease_name']=='bacterial_spot')
manifest=[]
with torch.inference_mode():
 for row,x in zip(rows,load_images(rows,256)):
  image=net(x[None].cuda()).cpu().add(1).div(2).clamp(0,1)
  dest=out/(Path(row['filename']).stem+'.png')
  save_image(image,dest)
  manifest.append({'path':str(dest),'disease_id':label,'recipient':row['filename']})
(HERE/'composites.json').write_text(json.dumps(manifest,indent=2))
