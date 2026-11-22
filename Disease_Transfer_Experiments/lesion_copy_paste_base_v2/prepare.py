"""Build a multi-pair pilot from reviewed points on the source contact sheet."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

HERE = Path(__file__).resolve().parent
# Each entry identifies a donor tile and a symptom ellipse on the 160 px tile.
DONORS = {
    'powdery_mildew': [(1, (75, 72, 91, 90)), (4, (95, 94, 115, 111)), (0, (68, 69, 84, 90))],
    'bacterial_spot': [(6, (33, 65, 61, 84)), (7, (89, 102, 117, 122)), (8, (94, 51, 112, 70))],
    'anthracnose': [(12, (84, 69, 103, 90)), (14, (79, 57, 96, 77)), (15, (72, 72, 91, 94))],
    'brown_spot': [(18, (49, 37, 70, 57)), (19, (55, 81, 73, 103)), (20, (78, 81, 99, 102)), (21, (82, 65, 101, 89))],
}
CENTERS = {
    24:(64,79),25:(79,73),26:(95,68),27:(89,81),28:(70,73),29:(73,82),
    30:(73,91),31:(87,74),32:(78,77),33:(83,83),34:(78,74),35:(103,94),
    36:(82,82),37:(85,73),38:(77,69),39:(90,85),40:(79,72),41:(81,90),
    43:(62,100),44:(75,76),45:(71,88),
    48:(47,73),49:(111,104),50:(70,88),51:(79,81),
    52:(85,76),53:(77,81),54:(74,81),55:(81,83),56:(65,97),
}
PAIRS = [
    ('apple','powdery_mildew',range(24,30)),
    ('tomato','powdery_mildew',range(30,36)),
    ('tomato','bacterial_spot',range(30,36)),
    ('blueberry','anthracnose',range(36,42)),
    ('apple','brown_spot',range(24,30)),
    ('cassava','brown_spot',range(43,46)),
    ('papaya','bacterial_spot',range(48,52)),
    ('blackgram','anthracnose',range(52,57)),
]


def tile(row):
    with Image.open(row['path']) as im:
        return ImageOps.fit(im.convert('RGB'), (160,160))


def main():
    rows = json.loads((HERE/'sources.json').read_text())
    p = json.loads((HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json').read_text())
    fit = {r['filename']:r for r in p['roles']['fit']}
    assert all(fit[r['filename']] == r for r in rows)
    review = Image.new('RGB',(640,len(PAIRS)*180))
    draw = ImageDraw.Draw(review)
    output = HERE/'composites'
    output.mkdir(exist_ok=True)
    rng = random.Random(124)
    records = []
    for group,(crop,disease,recipients) in enumerate(PAIRS):
        for i,(donor_index,box) in enumerate(DONORS[disease]):
            donor = rows[donor_index]
            assert donor['disease_name'] == disease
            patch = tile(donor).crop(box)
            alpha = Image.new('L',patch.size)
            ImageDraw.Draw(alpha).ellipse((1,1,patch.width-2,patch.height-2),fill=255)
            patch.putalpha(alpha.filter(ImageFilter.GaussianBlur(1)))
            for j,recipient_index in enumerate(recipients):
                recipient = rows[recipient_index]
                assert (recipient['crop_name'],recipient['disease_name']) == (crop,'healthy')
                cx,cy = CENTERS[recipient_index]
                for variant in range(4):
                    size = (.65,.8,.95,1.1)[variant]
                    stamp = patch.resize(tuple(round(n*size) for n in patch.size),Image.Resampling.LANCZOS)
                    stamp = stamp.rotate(rng.uniform(-35,35),Image.Resampling.BICUBIC,expand=True)
                    xy = (cx+rng.randint(-4,4)-stamp.width//2,cy+rng.randint(-4,4)-stamp.height//2)
                    im = tile(recipient)
                    im.paste(stamp,xy,stamp)
                    path = output/f'{group}_{i}_{j}_{variant}.png'
                    im.save(path)
                    records.append({'path':str(path),'donor':donor['filename'],'recipient':recipient['filename'],
                                    'crop_id':recipient['crop_id'],'disease_id':donor['disease_id'],
                                    'crop':crop,'disease':disease,'box':box,'placement':xy})
                    if variant == 2 and j == 0:
                        review.paste(im,(i*160,group*180))
        draw.text((0,group*180+161),f'{crop} / {disease}',fill='white')
    review.save(HERE/'composite_review.jpg')
    previous = HERE.parent/'lesion_copy_paste_base_v1/composites.json'
    mango = json.loads(previous.read_text())
    assert all(r['donor'] in fit and r['recipient'] in fit and Path(r['path']).is_file() for r in mango)
    records.extend(mango)
    (HERE/'composites.json').write_text(json.dumps(records,indent=2))
    print(f'{len(records)} composites across {len(PAIRS)+1} pairs')


if __name__ == '__main__':
    main()
