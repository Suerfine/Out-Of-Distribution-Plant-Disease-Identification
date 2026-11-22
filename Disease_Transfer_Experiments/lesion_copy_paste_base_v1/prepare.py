"""Create an auditable, manually localized lesion copy-paste pilot."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

HERE = Path(__file__).resolve().parent
# Coordinates are on the 200 px tiles in source_review.jpg.
LESIONS = [
    (4, [(12, 29), (22, 23), (39, 33), (46, 48), (40, 58), (25, 54), (16, 45)]),
    (4, [(80, 80), (89, 73), (105, 77), (111, 89), (103, 102), (88, 100)]),
    (7, [(70, 110), (82, 107), (92, 118), (86, 130), (74, 129)]),
    (1, [(67, 111), (86, 105), (103, 109), (107, 119), (92, 135), (74, 134)]),
]
CENTERS = [(96, 132), (92, 84), (144, 101), (95, 107), (104, 105), (98, 93), (102, 109), (111, 108)]


def tile(row):
    with Image.open(row['path']) as im:
        return ImageOps.fit(im.convert('RGB'), (200, 200))


def main():
    sources = json.loads((HERE/'sources.json').read_text())
    prepared = HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json'
    fit = {r['filename']: r for r in json.loads(prepared.read_text())['roles']['fit']}
    assert all(fit[r['filename']] == r for r in sources)
    out = HERE/'composites'
    out.mkdir(exist_ok=True)
    rng = random.Random(42)
    records = []
    review = Image.new('RGB', (800, 800))
    for li, (di, polygon) in enumerate(LESIONS):
        donor = sources[di]
        assert donor['disease_name'] == 'powdery_mildew'
        mask = Image.new('L', (200, 200))
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(1.25))
        box = mask.getbbox()
        lesion = tile(donor).crop(box)
        lesion.putalpha(mask.crop(box))
        for ri, center in enumerate(CENTERS):
            recipient = sources[12+ri]
            assert (recipient['crop_name'], recipient['disease_name']) == ('mango', 'healthy')
            for variant in range(4):
                leaf = tile(recipient)
                stamp = lesion.resize((round(lesion.width*.62), round(lesion.height*.62)), Image.Resampling.LANCZOS)
                stamp = stamp.rotate(rng.uniform(-22, 22), Image.Resampling.BICUBIC, expand=True)
                offsets = [(0,0),(-8,2),(7,-4),(4,5)]
                dx, dy = offsets[variant]
                xy = (center[0]+dx-stamp.width//2, center[1]+dy-stamp.height//2)
                leaf.paste(stamp, xy, stamp)
                path = out/f'{li}_{ri}_{variant}.png'
                leaf.save(path)
                records.append({'path': str(path), 'donor':donor['filename'], 'recipient':recipient['filename'],
                                'polygon':polygon,'placement':xy,'crop_id':recipient['crop_id'],
                                'disease_id':donor['disease_id']})
                if li == 0 and ri < 4:
                    review.paste(leaf, (ri*200,variant*200))
    (HERE/'composites.json').write_text(json.dumps(records,indent=2))
    review.save(HERE/'composite_review.jpg')


if __name__ == '__main__':
    main()

