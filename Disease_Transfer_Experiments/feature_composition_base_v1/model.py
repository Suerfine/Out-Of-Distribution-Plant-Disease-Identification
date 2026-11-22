"""Factor encoders, learned recomposition and stochastic encoder gradient replay."""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from protocol import seed_all, tensor_hash
from storage import IntegrityError
from independent_cnn_baseline.models import IndependentNet


class FeatureComposition(nn.Module):
    def __init__(self, base, cfg, pretrained=False):
        super().__init__()
        net = IndependentNet(base, 'disease', pretrained=pretrained)
        self.backbone, self.head = net.backbone, net.head
        width = self.backbone.num_features
        if width != 1024 or sum(p.numel() for p in self.backbone.parameters()) < 80000000:
            raise IntegrityError('Expected ConvNeXt-Base')
        self.add_factor_modules(width, cfg['factor_dim'], cfg['composer_hidden'], 55, 175, cfg['seed'] + 1703)

    def add_factor_modules(self, width, dim, hidden, crops, diseases, seed):
        # Preserve the original backbone/head seed stream when adding new modules.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            def factor(): return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim), nn.GELU(), nn.Linear(dim, dim))
            self.crop_factor, self.disease_factor = factor(), factor()
            self.crop_classifier, self.factor_disease_classifier = nn.Linear(dim, crops), nn.Linear(dim, diseases)
            self.composer = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, images):
        # The same independent disease output is used on real and synthesized features.
        return self.head(self.backbone(images))


def initialize(p, cfg):
    seed_all(cfg['seed']); model = FeatureComposition(p['baseline_config'], cfg, pretrained=True)
    saved = torch.load(p['ssl_encoder'], map_location='cpu', weights_only=True)
    model.backbone.load_state_dict(saved['encoder'], strict=True)
    if tensor_hash(model.backbone.state_dict()) != p['ssl_encoder_tensor_hash']:
        raise IntegrityError('SSL encoder transfer differs')
    if tensor_hash(model.head.state_dict()) != p['ssl_head_initial_hash']:
        raise IntegrityError('Fresh disease head differs from seed42 reference')
    return model.to(cfg['device'])


def optimizer(model, base):
    ocfg = base['optimizer']; groups = {}
    for name, parameter in model.named_parameters():
        head, decay = not name.startswith('backbone.'), parameter.ndim > 1 and not name.endswith('.bias')
        groups.setdefault((head, decay), []).append(parameter)
    return torch.optim.AdamW([{'params': params, 'lr': ocfg['lr_head' if head else 'lr_backbone'],
        'initial_lr': ocfg['lr_head' if head else 'lr_backbone'], 'weight_decay': ocfg['weight_decay'] if decay else 0.}
        for (head, decay), params in groups.items()], betas=tuple(ocfg['betas']), eps=ocfg['eps'])


def composition_loss(model, features, meta, cfg, epoch):
    n = meta['primary_count']; c, d = model.crop_factor(features), model.disease_factor(features)
    real_logits = model.head(features[:n])
    real = F.cross_entropy(real_logits.float(), meta['primary_disease'], label_smoothing=cfg['label_smoothing'])
    auxiliary = F.cross_entropy(model.crop_classifier(c[:n]).float(), meta['primary_crop'])
    auxiliary = auxiliary + F.cross_entropy(model.factor_disease_classifier(d[:n]).float(), meta['primary_disease'])
    reconstructed = model.composer(torch.cat([c[:n], d[:n]], 1))
    reconstruction = F.mse_loss(reconstructed.float(), features[:n].detach().float())
    crop_code, disease_code = c[meta['crop_positions']], d[meta['disease_positions']]
    synthetic = model.composer(torch.cat([crop_code, disease_code], 1))
    synthetic_crop, synthetic_disease = model.crop_factor(synthetic), model.disease_factor(synthetic)
    synthetic_ce = F.cross_entropy(model.head(synthetic).float(), meta['target_disease'], label_smoothing=cfg['label_smoothing'])
    synthetic_ce = synthetic_ce + .5 * F.cross_entropy(model.crop_classifier(synthetic_crop).float(), meta['target_crop'])
    synthetic_ce = synthetic_ce + .5 * F.cross_entropy(model.factor_disease_classifier(synthetic_disease).float(), meta['target_disease'])
    cycle = F.mse_loss(F.normalize(synthetic_crop.float(), dim=1), F.normalize(crop_code.detach().float(), dim=1))
    cycle = cycle + F.mse_loss(F.normalize(synthetic_disease.float(), dim=1), F.normalize(disease_code.detach().float(), dim=1))
    ramp = min(1., epoch / cfg['synthesis_ramp_epochs'])
    loss = real + cfg['auxiliary_weight'] * auxiliary + cfg['reconstruction_weight'] * reconstruction
    loss = loss + ramp * (cfg['synthetic_weight'] * synthetic_ce + cfg['cycle_weight'] * cycle)
    detail = {name: float(value.detach()) for name, value in [('real_ce', real), ('factor_ce', auxiliary),
        ('reconstruction', reconstruction), ('synthetic_ce', synthetic_ce), ('cycle', cycle)]}
    detail['real_correct'] = int((real_logits.detach().argmax(1) == meta['primary_disease']).sum())
    return loss, detail


def cached_backward(model, images, meta, cfg, epoch, micro, scaler=None, check_replay=False):
    """Retain logical-batch feature gradients; replay each stochastic encoder chunk.

    RNG states reproduce DropPath exactly. Forked replay preserves the RNG state
    after the first encoder pass and the actual stochastic classification heads.
    Encoder BatchNorm is forbidden because replay would update its buffers twice.
    """
    device = torch.device(cfg['device']); gpu = device.type == 'cuda'
    gpu_index = (device.index if device.index is not None else torch.cuda.current_device()) if gpu else None
    def cast(): return torch.autocast('cuda', dtype=torch.float16) if cfg['amp'] else nullcontext()
    states, chunks = [], []
    with torch.no_grad():
        for start in range(0, len(images), micro):
            states.append((torch.get_rng_state(), torch.cuda.get_rng_state(gpu_index) if gpu else None))
            with cast(): chunks.append(model.backbone(images[start:start + micro].to(device, non_blocking=True)).float())
    features = torch.cat(chunks).detach().requires_grad_(True)
    with cast(): loss, detail = composition_loss(model, features, meta, cfg, epoch)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite composition loss')
    (scaler.scale(loss) if scaler else loss).backward()
    max_error = 0.
    with torch.random.fork_rng(devices=[gpu_index] if gpu else []):
        for start, state in zip(range(0, len(images), micro), states):
            torch.set_rng_state(state[0])
            if gpu: torch.cuda.set_rng_state(state[1], gpu_index)
            with cast(): replay = model.backbone(images[start:start + micro].to(device, non_blocking=True)).float()
            if check_replay:
                max_error = max(max_error, float((replay.detach() - features[start:start + micro].detach()).abs().max()))
            replay.backward(features.grad[start:start + micro])
    if check_replay and max_error != 0: raise IntegrityError('Stochastic encoder replay differs: ' + str(max_error))
    detail['replay_max_error'] = max_error
    return float(loss.detach()), detail
