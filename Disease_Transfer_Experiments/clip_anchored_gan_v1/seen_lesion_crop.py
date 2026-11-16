"""Test reviewed lesion close-ups as training views of existing images."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from seen_hinge_head import HERE, main


def reviewed_views(fit, features, net, transform):
    fit_index = {r['filename']: i for i, r in enumerate(fit)}
    boxes = {}
    for version, size in ((1, 200), (2, 160), (3, 160)):
        path = HERE.parent / f'lesion_copy_paste_base_v{version}/composites.json'
        for row in json.loads(path.read_text()):
            # Later manifests include prior rows. Keep the first coordinate system.
            if row['donor'] in boxes:
                continue
            if 'polygon' in row:
                x, y = zip(*row['polygon'])
                box = (min(x), min(y), max(x)+1, max(y)+1)
            else:
                box = row['box']
            boxes[row['donor']] = (size, box)
    assert len(boxes) == 29 and set(boxes) <= fit_index.keys()
    lesion_views, corner_views, donor_indices = [], [], []
    for name, (size, box) in boxes.items():
        index = fit_index[name]
        with Image.open(fit[index]['path']) as file:
            image = file.convert('RGB')
            side = min(image.size)
            tile = ImageOps.fit(image, (side, side))
        left, top, right, bottom = [round(coordinate * side / size) for coordinate in box]
        for padding in (.25, .5, .75, 1.):
            width = min(side, round((right-left) * (1 + 2*padding)))
            height = min(side, round((bottom-top) * (1 + 2*padding)))
            x = min(max(0, (left+right-width)//2), side-width)
            y = min(max(0, (top+bottom-height)//2), side-height)
            # Select the same-size corner farthest from the reviewed box.
            cx, cy = max(((a,b) for a in (0,side-width) for b in (0,side-height)),
                         key=lambda point: (point[0]-x)**2 + (point[1]-y)**2)
            lesion_views.append(transform(tile.crop((x,y,x+width,y+height))))
            corner_views.append(transform(tile.crop((cx,cy,cx+width,cy+height))))
            donor_indices.append(index)

    def encode(images):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            output = torch.cat([net.backbone(batch.cuda()).float().cpu()
                                for batch in torch.stack(images).split(16)])
        return output.cuda()

    views = {'repeat_full': features[donor_indices], 'corner': encode(corner_views),
             'reviewed_lesion': encode(lesion_views)}
    y = torch.tensor([fit[i]['disease_id'] for i in donor_indices], device='cuda')
    donor_pairs = sorted({fit[i]['composition_id'] for i in donor_indices})
    return views, y, donor_pairs


def compare(initial, x, y, val, views, extra_y, donor_pairs, labels, rank, report):
    targeted = np.isin(labels, donor_pairs)
    results = {}
    for name, extra in {'full_only': None, **views}.items():
        generator = torch.Generator(device='cuda').manual_seed(42)
        head = deepcopy(initial)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        # Every arm sees the same number of examples and uses the same batches.
        if extra is None:
            sampled = torch.randperm(len(y), device='cuda', generator=generator)[:len(extra_y)]
            train_x, train_y = torch.cat((x,x[sampled])), torch.cat((y,y[sampled]))
        else:
            train_x, train_y = torch.cat((x,extra)), torch.cat((y,extra_y))
        generator.manual_seed(43)
        for _ in range(3):
            for idx in torch.randperm(len(train_y), device='cuda', generator=generator).split(64):
                loss = F.cross_entropy(head(train_x[idx]), train_y[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        head.eval()
        with torch.inference_mode():
            logits = head(val).cpu().numpy()
        assert np.isfinite(logits).all()
        top = rank(logits, 1.)
        results[name] = dict(**report(logits, top), targeted_top1=int((targeted & (top[:,0]==labels)).sum()))
    return dict(donors=len(extra_y)//4, extra_views=len(extra_y), donor_pair_ids=donor_pairs,
                targeted_images=int(targeted.sum()), epochs=3, arms=results)


if __name__ == '__main__':
    main(lesion_pilot=True)
