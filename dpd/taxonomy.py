"""Side knowledge for the knowledge-based approaches: botanical family of each crop and pathogen group of each disease.

Only used by approaches that declare it (loss.taxonomy); nothing else reads this file. The crop families are standard
botany. The disease groups are a best-effort grouping from the disease names (the dataset gives no pathogen labels):
"other" collects names that do not say what causes them (generic spots, abiotic or physiological problems, algae), so
treat the groups as approximate and a declared ingredient of the approach, not as ground truth.
"""
import numpy as np

CROP_FAMILY = {
    "apple": "Rosaceae", "apricot": "Rosaceae", "banana": "Musaceae", "basil": "Lamiaceae", "bean": "Fabaceae",
    "blackgram": "Fabaceae", "blueberry": "Ericaceae", "brassica": "Brassicaceae", "broccoli": "Brassicaceae",
    "cabbage": "Brassicaceae", "carrot": "Apiaceae", "cashew": "Anacardiaceae", "cassava": "Euphorbiaceae",
    "cauliflower": "Brassicaceae", "celery": "Apiaceae", "cherry": "Rosaceae", "chilli": "Solanaceae",
    "citrus": "Rutaceae", "coffee": "Rubiaceae", "coriander": "Apiaceae", "corn": "Poaceae", "cotton": "Malvaceae",
    "cucumber": "Cucurbitaceae", "eggplant": "Solanaceae", "garlic": "Amaryllidaceae", "ginger": "Zingiberaceae",
    "grape": "Vitaceae", "guava": "Myrtaceae", "lettuce": "Asteraceae", "mango": "Anacardiaceae", "maple": "Sapindaceae",
    "olive": "Oleaceae", "paddy": "Poaceae", "palm": "Arecaceae", "papaya": "Caricaceae", "peach": "Rosaceae",
    "peanut": "Fabaceae", "pear": "Rosaceae", "pepper_bell": "Solanaceae", "plum": "Rosaceae", "potato": "Solanaceae",
    "pumpkin": "Cucurbitaceae", "raspberry": "Rosaceae", "rose": "Rosaceae", "soybean": "Fabaceae",
    "squash": "Cucurbitaceae", "strawberry": "Rosaceae", "sugarcane": "Poaceae", "sunflower": "Asteraceae",
    "tea": "Theaceae", "tobacco": "Solanaceae", "tomato": "Solanaceae", "walnut": "Juglandaceae", "wheat": "Poaceae",
    "zucchini": "Cucurbitaceae",
}

_G = {
    "fungal": """alternaria_blotch alternaria_leaf_blight alternaria_leaf_spot anthracnose bean_rust berry_blotch
        bird_eye_spot black_leaf_streak black_rot black_spot blast blight_leaf blue_mold botrytis_blight brown_blight
        brown_eye_spot brown_leaf_spot brown_rot brown_rust brown_spot cavity_spot cedar_apple_rust cercospora
        cercospora_leaf_blight cigar_end_rot common_rust cordana_leaf_spot coryneum_blight downy_mildew early_blight
        early_leaf_spot esca frog_eye_leaf_spot frogeye_leaf_spot fusarium_wilt gnomonia_leptostyla gray_leaf_spot
        gray_light gray_mold grey_spot gummy_stem_blight head_scab late_blight late_leaf_spot leaf_blight leaf_mold leafblast
        loose_smut marssonina_blotch melanose mildew monillia_laxa mummy_berry northern_leaf_blight panama_disease
        peacock_spot pestalotiopsis phoma phomopsis_fruit_rot phytophthora pokkah_boeng powdery_mildew red_rot root_rot
        rust scab segatoka septoria septoria_blotch septoria_leaf_spot sett_rot sheath_blight sigatoka smut sooty_mould
        stripe_rust tar_spot target_spot venturia_inaequalis verticulium_wilt yellow_rust""",
    "bacterial": """angular_leaf_spot bacterial_blight bacterial_canker bacterial_leaf_spot bacterial_leaf_streak
        bacterial_panicle_blight bacterial_soft_rot bacterial_spot bacterial_wilt erwinia_amylovora fire_blight greening
        halo_blight xamthomonas""",
    "viral": """brown_streak_disease bunchy_top crinckle curl curl_virus grassy_shoot green_mottle leaf_yellow_virus
        leafroll_disease lethal_necrosis mosaic mosaic_virus pox_virus streak streak_virus tungro viral
        yellow_leaf_curl_virus yellow_mosaic yellow_mosaic_virus""",
    "pest": """aculus_olearius aphis_spp apoderus_javanicus aulacaspis_tubercularis black_stem_borer caterpillar
        ceroplastes_rubens cisaberoptus_kenyae cutting_weevil dappula_tertia diabrotica_speciosa
        dialeuropora_decempuncta eriosoma_lanigerum erosomyia_sp gall_midge green_mite hispa icerya_seychellarum
        insects_damages ischnaspis_longirostris leaf_roller mictis_longicornis miner neomelicharia_sparsa
        orthaga_euadrusalis parthenolecanium_corni procontarinia_matteiana procontarinia_rubus red_mite slug
        spider_mites valanga_nigricornis white_scale white_stem_borer yellow_stem_borer""",
    "healthy": "healthy",
    "other": """algal_leaf banded_chlorosis blossom_end_rot canker chlorotic_leaf_spot die_back dot_leaf leaf_scars
        mummification_leaf nutrition_deficiency pocket_disease purple_discoloration red_leaf_spot red_rust red_scab
        ring_spot scorch spot stripe stylar_end_rot violet_decoloration white_spot yellow_spots yellowing""",
}
DISEASE_GROUP = {d: g for g, names in _G.items() for d in names.split()}


def taxonomy_ids(pairs) -> dict:
    """Group ids for the crop and disease heads: {"crop": (family id per crop id, family names),
    "disease": (group id per disease id, group names)}. `pairs` is dpd.data.pair_table()."""
    out = {}
    for head, name_col, id_col, table in (("crop", "crop", "crop_id", CROP_FAMILY), ("disease", "disease", "disease_id", DISEASE_GROUP)):
        names = pairs.drop_duplicates(id_col).set_index(id_col)[name_col].sort_index()
        missing = [n for n in names if n not in table]
        assert not missing, f"no {head} group for: {missing}"
        groups = sorted(set(table[n] for n in names))
        out[head] = (np.array([groups.index(table[n]) for n in names], dtype=np.int64), groups)
    return out
