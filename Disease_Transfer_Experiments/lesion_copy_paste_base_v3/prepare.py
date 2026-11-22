"""Extend the prior manifest with same-pair composites from reviewed leaves."""
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

HERE = Path(__file__).resolve().parent
# Donor boxes and recipient centers refer to the 160 px source review tiles.
GROUPS = [
    ([(0,(51,69,84,87)),(1,(75,63,101,90)),(2,(74,67,101,91))],
     [(3,70,92),(4,72,83),(5,82,74)]),
    ([(6,(45,74,70,97)),(7,(35,61,63,88)),(8,(54,86,80,109))],
     [(9,74,75),(10,90,85),(11,84,83)]),
    ([(12,(96,76,118,98)),(13,(58,107,76,124)),(14,(36,30,56,55))],
     [(15,86,80),(16,72,86),(17,84,84)]),
    ([(24,(101,105,121,128)),(26,(70,83,94,103))],
     [(27,105,54),(28,59,63),(29,93,75)]),
    ([(30,(68,76,86,119)),(31,(72,103,96,118)),(32,(81,50,96,76))],
     [(33,34,79),(34,65,86),(35,92,83)]),
]


def tile(row):
    with Image.open(row['path']) as image:
        return ImageOps.fit(image.convert('RGB'), (160, 160))


def main():
    previous = HERE.parent/'lesion_copy_paste_base_v2'
    records = json.loads((previous/'composites.json').read_text())
    rows = json.loads((HERE/'expanded_sources.json').read_text())
    prepared = HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json'
    roles = json.loads(prepared.read_text())['roles']
    fit = {row['filename']: row for row in roles['fit']}
    fit_pairs = {(r['crop_id'], r['disease_id']) for r in fit.values()}
    rng = random.Random(372)
    folder = HERE/'composites'
    folder.mkdir(exist_ok=True)
    review = Image.new('RGB',(480,178*len(GROUPS)))
    draw = ImageDraw.Draw(review)
    for group,(donors,recipients) in enumerate(GROUPS):
        for i,(index,box) in enumerate(donors):
            donor = rows[index]
            patch = tile(donor).crop(box)
            mask = Image.new('L',patch.size)
            ImageDraw.Draw(mask).ellipse((1,1,patch.width-2,patch.height-2),fill=255)
            patch.putalpha(mask.filter(ImageFilter.GaussianBlur(1.25)))
            for j,(index,cx,cy) in enumerate(recipients):
                recipient = rows[index]
                assert fit[donor['filename']] == donor and fit[recipient['filename']] == recipient
                assert recipient['disease_name'] == 'healthy' and recipient['crop_id'] == donor['crop_id']
                for variant,scale in enumerate((.65,.8,.95,1.1)):
                    stamp = patch.resize(tuple(round(n*scale) for n in patch.size),Image.Resampling.LANCZOS)
                    stamp = stamp.rotate(rng.uniform(-25,25),Image.Resampling.BICUBIC,expand=True)
                    xy = (cx+rng.randint(-3,3)-stamp.width//2,cy+rng.randint(-3,3)-stamp.height//2)
                    im = tile(recipient)
                    im.paste(stamp,xy,stamp)
                    path = folder/f'{group}_{i}_{j}_{variant}.png'
                    im.save(path)
                    records.append({'path':str(path),'donor':donor['filename'],'recipient':recipient['filename'],
                                    'crop_id':recipient['crop_id'],'disease_id':donor['disease_id'],
                                    'box':box,'placement':xy})
                    if variant == 2 and j == 0:
                        review.paste(im,(i*160,group*178))
        draw.text((0,group*178+161),f"{donor['crop_name']} / {donor['disease_name']}",fill='white')
    review.save(HERE/'expanded_review.jpg')
    counts = Counter()
    for row in records:
        donor, recipient = fit[row['donor']], fit[row['recipient']]
        assert row['disease_id'] == donor['disease_id'] and row['crop_id'] == recipient['crop_id']
        assert recipient['disease_name'] == 'healthy'
        with Image.open(row['path']) as image:
            image.verify()
        counts[row['crop_id'],row['disease_id']] += 1
    assert len(counts) == 12 and len(counts.keys() - fit_pairs) == 3
    (HERE/'composites.json').write_text(json.dumps(records,indent=2))
    print(len(records),len(counts),len({r['donor'] for r in records}),len({r['recipient'] for r in records}))


if __name__ == '__main__':
    main()
