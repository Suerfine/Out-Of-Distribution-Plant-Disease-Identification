import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
previous = PARENT/'lesion_copy_paste_base_v3'


def tile(row):
    with Image.open(row['path']) as im:
        return ImageOps.fit(im.convert('RGB'),(160,160))


def main():
    root = PARENT.parent
    p = json.loads((PARENT/'cnn_detail_base_v1/runs/development_seed42/prepared.json').read_text())
    fit = {r['filename']:r for r in p['roles']['fit']}
    records = json.loads((previous/'composites.json').read_text())
    sources = json.loads((previous/'expanded_sources.json').read_text())
    # Reviewed arrays from the previous preparation.
    import sys
    sys.path.insert(0,str(previous))
    from prepare import GROUPS
    donors = [(sources[i], box) for group,_ in GROUPS for i,box in group]
    recipients = [(sources[i],x,y) for _,group in GROUPS for i,x,y in group]
    known = {(r['crop_id'],r['disease_id']) for r in p['pairs']}
    candidates = {}
    # Pair by leaf symptom type using the existing prepared healthy leaves.
    # Generate the candidate list before checking either official test manifest.
    for donor,box in donors:
        for recipient,x,y in recipients:
            key = recipient['crop_id'],donor['disease_id']
            if key not in known:
                candidates.setdefault(key,[]).append((donor,box,recipient,x,y))
    official = set()
    for split in ('seen','unseen'):
        with (root/f'Plant_Disease/DPD_Small_test_{split}.csv').open(encoding='utf-8-sig',newline='') as f:
            official.update((int(r[1]),int(r[2])) for r in csv.reader(f))
    rng = random.Random(421)
    accepted = sorted(candidates.keys()-official)
    assert accepted and not (set(accepted)&known)
    dest = HERE/'novel'
    dest.mkdir(exist_ok=True)
    review = Image.new('RGB',(640,180*((len(accepted)+3)//4)))
    labels = ImageDraw.Draw(review)
    counts = Counter()
    for group,key in enumerate(accepted):
        options = candidates[key]
        rng.shuffle(options)
        for i,(donor,box,recipient,x,y) in enumerate(options[:12]):
            assert fit[donor['filename']] == donor and fit[recipient['filename']] == recipient
            assert recipient['disease_name'] == 'healthy'
            lesion = tile(donor).crop(box)
            mask = Image.new('L',lesion.size)
            ImageDraw.Draw(mask).ellipse((1,1,lesion.width-2,lesion.height-2), fill=255)
            lesion.putalpha(mask.filter(ImageFilter.GaussianBlur(1.25)))
            for variant,scale in enumerate((.62,.85)):
                stamp = lesion.resize(tuple(round(n*scale) for n in lesion.size),Image.Resampling.LANCZOS)
                stamp = stamp.rotate(rng.uniform(-20,20),Image.Resampling.BICUBIC,expand=True)
                xy = (x-stamp.width//2,y-stamp.height//2)
                im = tile(recipient)
                im.paste(stamp,xy,stamp)
                path = dest/f'{group}_{i}_{variant}.png'
                im.save(path)
                records.append({'path':str(path),'donor':donor['filename'],'recipient':recipient['filename'],
                                'crop_id':key[0],'disease_id':key[1],'box':box,'placement':xy})
                counts[f"{recipient['crop_name']} / {donor['disease_name']}"]+=1
                if i == variant == 0:
                    a,b=group%4*160,group//4*180
                    review.paste(im,(a,b))
                    labels.text((a,b+161),f"{recipient['crop_name']} {donor['disease_name']}",fill='white')
    review.save(HERE/'novel_review.jpg')
    (HERE/'composites.json').write_text(json.dumps(records,indent=2))
    (HERE/'audit.json').write_text(json.dumps({'novel_pairs':counts,'total_composites':len(records),
        'proposals_excluded_by_test_overlap':len(candidates.keys() & official)},indent=2))
    print(len(accepted),sum(counts.values()),len(records),dict(counts))


if __name__ == '__main__':
    main()
