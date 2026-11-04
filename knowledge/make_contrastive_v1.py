"""Builds disease_descriptions_contrastive_v1.json: v1 with rewrites for diseases whose v1 prompts are near-duplicates
in frozen CLIP text space (cosine > ~0.92 to a genuinely different disease). Each rewrite leads with the cue that
separates it from its neighbours (shape, size, colour, location, pattern) and drops shared generic words such as
"brown spots" or "yellow halo" unless they are the distinguishing cue. True synonyms (e.g. mildew / powdery_mildew,
sigatoka / segatoka) are left alone: they sit on different crops, so the crop vector separates them.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
REWRITES = {
    # target / ringed spots (early_blight absorbs many seen disease errors)
    "early_blight": "dark brown bullseye spots with tight concentric rings, first on the oldest lower leaves, which turn yellow",
    "target_spot": "light tan spots with faint wide rings and pale centres that crack and fall out, on leaves of all ages",
    "late_blight": "large greasy grey-green to black water-soaked patches without rings, with white fuzzy mould at the edge",
    "septoria_leaf_spot": "dozens of tiny round spots with white-grey centres, thin dark rims and black pepper-like dots",
    "bacterial_spot": "tiny dark greasy specks that merge so the leaf becomes ragged and torn with holes",
    "peacock_spot": "sooty dark circular spots ringed by a yellow halo, like the eye of a peacock feather",
    # leaf-spot family
    "late_leaf_spot": "near-black round spots without a halo, crowded on the leaf underside",
    "early_leaf_spot": "reddish-brown circular spots each surrounded by a distinct yellow ring, on the upper leaf surface",
    "brown_leaf_spot": "small angular brown spots with dark borders limited by the veins, on older leaves",
    "red_leaf_spot": "tiny red to crimson dots and specks scattered over the leaf",
    "angular_leaf_spot": "angular straw-coloured patches confined between the veins, drying and tearing out",
    "cordana_leaf_spot": "large pale brown eye-shaped oval spots with dark rims and a bright yellow halo along the leaf edge",
    "gray_leaf_spot": "long narrow rectangular grey matchstick lesions running parallel between the veins",
    "northern_leaf_blight": "very long grey-green cigar-shaped lesions several centimetres long with wavy edges",
    "brown_spot": "oval sesame-seed-like brown spots with a grey centre and a dark border",
    "grey_spot": "pale ash-grey spots with thin dark borders",
    "spot": "small scattered dark specks without rings or halos",
    "white_spot": "small chalky white round spots",
    "black_spot": "jet-black round spots with fringed feathery edges on a yellowing leaf",
    "yellow_spots": "small bright yellow flecks on a green leaf",
    "anthracnose": "dark sunken dead patches spreading inward from the leaf tip and margins, with tiny orange or black spore dots",
    "leaf_blight": "large tan dead areas covering much of the leaf, drying from the tip downward",
    "brown_blight": "brown blotches with grey zoned centres starting at the leaf margin, dotted with black specks",
    # stem borers
    "yellow_stem_borer": "dried central shoot and empty white upright grain heads",
    "white_stem_borer": "brown dead central shoots that pull out easily, with bore holes in the stem",
    "black_stem_borer": "black bore holes and frass in the stem, with wilting leaves",
    # yellowing / viral
    "yellowing": "uniformly pale yellow leaves with no pattern, spots or distortion",
    "greening": "asymmetric blotchy yellow mottle that differs on each side of the midrib, small upright leaves",
    "nutrition_deficiency": "yellowing between green veins or of whole older leaves, with no spots, mould or distortion",
    "mosaic": "patchwork of light green and yellow patches with narrow, twisted leaflets",
    "yellow_leaf_curl_virus": "small cupped leaves curled upward with bright yellow margins on bushy stunted plants",
    "curl": "leaves curled, puckered, thickened and twisted while staying green",
    "crinckle": "wrinkled leaf surface with raised bumps and light mottling",
}

v1 = json.loads((HERE / "disease_descriptions_v1.json").read_text(encoding="utf-8"))
assert set(REWRITES) <= set(v1), set(REWRITES) - set(v1)
out = {"_note": "v1 with contrastive rewrites for diseases whose v1 prompts were near-duplicates of a different disease "
                "in frozen CLIP text space; built by knowledge/make_contrastive_v1.py, which lists the rewrites. Written by "
                "Claude from general plant-pathology knowledge, unverified."}
out.update({k: REWRITES.get(k, v) for k, v in v1.items() if not k.startswith("_")})
(HERE / "disease_descriptions_contrastive_v1.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{len(REWRITES)} rewritten of {len(out) - 1}")
