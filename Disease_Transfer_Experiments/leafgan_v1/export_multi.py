"""Export generated samples selected during review of each generator."""
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from train import HERE, load_images, sha256
from models import networks


def main():
    config = json.loads((HERE/'study.json').read_text(encoding='utf-8-sig'))
    output = HERE/'multi_pair'
    approval = json.loads((output/'reviewed_sources.json').read_text(encoding='utf-8-sig'))
    manifest = []
    for spec in config['pairs']:
        slug = spec['crop']+'_'+spec['disease']
        filenames = approval.get(slug, [])
        if not filenames:
            continue
        run = output/slug/'generator'
        pointer = json.loads((run/'checkpoint_pointer.json').read_text())['active']
        checkpoint = run/pointer['file']
        if pointer['epoch'] != config['generator_epochs'] or sha256(checkpoint) != pointer['sha256']:
            raise RuntimeError(f'Final generator checkpoint failed verification for {slug}')
        sources = json.loads((run/'sources.json').read_text())
        eligible = {r['filename']: r for r in sources['healthy']+sources['preview']}
        if len(set(filenames)) != len(filenames) or len(filenames) > config['max_generated_per_pair']:
            raise ValueError(f'Invalid reviewed source list for {slug}')
        rows = [eligible[name] for name in filenames]
        dest = output/slug/'synthetic'
        dest.mkdir(exist_ok=True)
        model = networks.define_G(3,3,64,'resnet_9blocks','instance',False,'normal',.02,[0])
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['models']['G_A'])
        model.eval()
        with torch.inference_mode():
            for row, image in zip(rows,load_images(rows,256)):
                path = dest/(Path(row['filename']).stem+'.png')
                save_image(model(image[None].cuda()).cpu().add(1).div(2).clamp(0,1),path)
                manifest.append({'path':str(path),'recipient':row['filename'],'crop_id':row['crop_id'],
                                 'disease_id':sources['diseased'][0]['disease_id'],'pair':slug})
        del model
    if not manifest:
        raise RuntimeError('The reviewed source list is empty')
    (output/'composites.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
