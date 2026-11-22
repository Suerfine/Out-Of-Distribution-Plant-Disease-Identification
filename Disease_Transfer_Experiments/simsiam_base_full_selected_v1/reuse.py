"""Verify and reuse the completed full-data SSL stage without redoing updates."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from storage import IntegrityError, sha

SSL_KEYS = ('model', 'seed', 'ssl_epochs', 'ssl_batch', 'micro_batch', 'workers',
            'image_size', 'projector_dim', 'predictor_dim', 'ssl_backbone_lr',
            'ssl_projector_lr', 'ssl_predictor_lr', 'ssl_weight_decay',
            'ssl_warmup_epochs', 'amp', 'amp_initial_scale')


def validate_ssl_settings(a, b):
    for key in SSL_KEYS:
        if a[key] != b[key]: raise IntegrityError('SSL recipe mismatch: ' + key)
    if a['model'] != 'convnext_base.fb_in22k_ft_in1k' or a['ssl_epochs'] != 20:
        raise IntegrityError('Expected the fixed 20-epoch ConvNeXt-Base recipe')


def audit_reuse(cfg, p):
    from protocol import HERE, tensor_hash
    source = Path(cfg['ssl_source_run'])
    old = json.loads((source / 'prepared.json').read_text())
    validate_ssl_settings(cfg, old['config'])
    if old['baseline_config'] != p['baseline_config']:
        raise IntegrityError('Original Base model/transform/optimizer configuration changed')
    if old['roles']['fit'] != p['roles']['fit']:
        raise IntegrityError('Full-data SSL image membership, order, or labels changed')
    for row in p['roles']['fit']:
        if old['cache_hashes'][row['path']] != p['cache_hashes'][row['path']]:
            raise IntegrityError('An SSL training image changed')
    if old['pretrained_sha256'] != p['pretrained_sha256']:
        raise IntegrityError('Original SSL pretrained weights changed')
    # Pin the operations that generated SSL views, encoder/head loss and SGD.
    for name in ('data.py', 'model.py'):
        if sha(HERE / name) != old['source_hashes'][str(source.parent.parent / name)]:
            raise IntegrityError('SSL implementation changed: ' + name)
    for path, digest in old['source_hashes'].items():
        if sha(path) != digest: raise IntegrityError('Saved SSL source changed: ' + path)
    check = json.loads((source / 'independent_verification.json').read_text())
    if not check['passed'] or not check['no_test_image_ssl']:
        raise IntegrityError('Source SSL run did not pass its no-test-exposure verification')
    init = json.loads((source / 'finetune_initialization.json').read_text())
    ssl_init = json.loads((source / 'ssl_initialization.json').read_text())
    encoder_path = source / 'ssl_encoder.pt'
    encoder_sha = sha(encoder_path)
    if encoder_sha != init['source_ssl_encoder_sha256']:
        raise IntegrityError('Full-data SSL encoder checksum changed')
    encoder = torch.load(encoder_path, map_location='cpu', weights_only=True)
    if encoder['epochs'] != 20 or encoder['fingerprint'] != old['fingerprint']:
        raise IntegrityError('SSL payload provenance differs')
    if tensor_hash(encoder['encoder']) != init['encoder_hash']:
        raise IntegrityError('Full-data SSL encoder tensor identity differs')
    if sum(v.numel() for v in encoder['encoder'].values()) < 80_000_000:
        raise IntegrityError('SSL encoder is not ConvNeXt-Base')
    del encoder
    history = json.loads((source / 'ssl/history.json').read_text())
    if len(history) != 20 or [h['epoch'] for h in history] != list(range(1, 21)):
        raise IntegrityError('Incomplete source SSL history')
    paths = [source.parent.parent / 'config.json'] + [source / name for name in ('prepared.json', 'ssl_encoder.pt',
        'finetune_initialization.json', 'ssl_initialization.json', 'independent_verification.json',
        'preflight.json', 'ssl/history.json')]
    for h in history:
        path = source / 'ssl' / f"exposures_epoch{h['epoch']:02}.npz"
        if sha(path) != h['exposure_sha256']: raise IntegrityError('Source SSL exposure hash differs')
        with np.load(path) as bank:
            if not np.array_equal(bank['image_counts'], np.ones(8000, dtype=int)):
                raise IntegrityError('Source SSL did not expose every training image once')
        if h['training']['samples'] != 8000: raise IntegrityError('Incomplete source SSL exposure')
        paths.append(path)
    dev = Path(cfg['development_run'])
    selected = json.loads((dev / 'selection_frozen.json').read_text())
    verification = json.loads((dev / 'verification.json').read_text())
    devp = json.loads((dev / 'prepared.json').read_text())
    validate_ssl_settings(cfg, devp['config'])
    ft = json.loads((dev / 'finetune/history.json').read_text())
    expected_epoch = 1 + int(np.argmax([h['validation']['objective']['harmonic_mean_top1'] for h in ft]))
    if not verification['passed'] or selected['official_scores_used']:
        raise IntegrityError('Invalid development selection record')
    if cfg['finetune_epochs'] != selected['best']['epoch'] or selected['best']['epoch'] != expected_epoch:
        raise IntegrityError('Full-data supervised budget differs from the earliest development maximum')
    if any(h['validation']['objective']['gamma'] != 0 for h in ft):
        raise IntegrityError('Development budget was not selected at gamma zero')
    paths += [dev / name for name in ('prepared.json', 'selection_frozen.json', 'verification.json', 'finetune/history.json')]
    source_hashes = {**old['source_hashes'], **{str(path): sha(path) for path in paths}}
    return {'source_run': str(source), 'fingerprint': old['fingerprint'],
        'encoder_path': str(encoder_path), 'encoder_sha256': encoder_sha,
        'encoder_tensor_hash': init['encoder_hash'], 'expected_head_hash': init['head_hash'],
        'initial_encoder_hash': ssl_init['encoder_hash'], 'epochs': 20,
        'source_verification_checks': check['checks'], 'recipe_and_full_data_match': True,
        'selected_supervised_epochs': expected_epoch, 'development_source': str(dev)}, source_hashes


def install_reused_ssl(p, cfg, run):
    from protocol import write_json
    source = Path(p['ssl_reuse']['source_run'])
    (run / 'ssl').mkdir(exist_ok=True)
    shutil.copyfile(source / 'ssl_encoder.pt', run / 'ssl_encoder.pt')
    shutil.copyfile(source / 'ssl_initialization.json', run / 'ssl_initialization.json')
    history = json.loads((source / 'ssl/history.json').read_text())
    for h in history:
        name = f"exposures_epoch{h['epoch']:02}.npz"
        shutil.copyfile(source / 'ssl' / name, run / 'ssl' / name)
    write_json(run / 'ssl/history.json', history)
    write_json(run / 'ssl/summary.json', {'completed_epochs': 20, 'last': history[-1], 'reused_from': str(source)})
    if sha(run / 'ssl_encoder.pt') != p['ssl_reuse']['encoder_sha256']:
        raise IntegrityError('Copied SSL encoder differs from the verified source')
    write_json(run / 'ssl_reuse.json', {**p['ssl_reuse'], 'ssl_updates_repeated': 0,
        'new_supervised_epochs': cfg['finetune_epochs'], 'source_ssl_history_copied_unchanged': True})
    return history
