"""Shared disease classifier with crop-conditioned feature modulation."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class CropConditioner(nn.Module):
    def __init__(self, crops, features, embedding_dim=64, scale=.1):
        super().__init__()
        self.embedding = nn.Embedding(crops, embedding_dim)
        self.affine = nn.Linear(embedding_dim, 2*features)
        self.scale = scale
        nn.init.normal_(self.embedding.weight, std=.02)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features, classifier, crop=None):
        # Explicit residual computation preserves the baseline logits exactly
        # when affine is zero, including under autocast.
        base = classifier(features)
        indices = torch.arange(self.embedding.num_embeddings, device=features.device) if crop is None else crop
        gamma, beta = (self.scale*self.affine(self.embedding(indices)).tanh()).chunk(2, dim=-1)
        if crop is None:
            residual = features[:, None, :]*gamma[None, :, :] + beta[None, :, :]
            return base[:, None, :] + F.linear(residual, classifier.weight, bias=None)
        residual = features*gamma + beta
        return base + F.linear(residual, classifier.weight, bias=None)


class ConditionalDiseaseCNN(nn.Module):
    def __init__(self, cfg, ontology, warm_start=True):
        super().__init__()
        import timm
        self.net = timm.create_model(cfg['model'], pretrained=False, num_classes=len(ontology['disease']))
        if warm_start:
            source = torch.load(Path(cfg['baseline_run'])/'disease/best.pt', map_location='cpu', weights_only=True)
            if source['label_ids'] != ontology['disease'] or source['model_name'] != cfg['model']:
                raise ValueError('Saved disease checkpoint label/model mismatch')
            self.net.load_state_dict(source['model_state'], strict=True)
        self.conditioner = CropConditioner(len(ontology['crop']), self.net.num_features,
                                          cfg['conditioning_dim'], cfg['conditioning_scale'])

    def forward(self, images, crop=None):
        features = self.net.forward_head(self.net.forward_features(images), pre_logits=True)
        return self.conditioner(features, self.net.get_classifier(), crop)


def conditional_scores(crop_logits, conditional_logits, ontology):
    """No labels: P(c|x) P(d|x,c), plus the disease marginal over all crops.

    Crop-dependent normalizers matter: adding the raw logits would give a
    different, incorrect ranking. Float64 permits independent score verification.
    """
    crops, diseases = len(ontology['crop']), len(ontology['disease'])
    if conditional_logits.shape != (len(crop_logits), crops, diseases):
        raise ValueError('Conditional score shape mismatch')
    log_c = F.log_softmax(crop_logits.double(), dim=1)
    log_d_c = F.log_softmax(conditional_logits.double(), dim=2)
    cm = {v:i for i,v in enumerate(ontology['crop'])}
    dm = {v:i for i,v in enumerate(ontology['disease'])}
    ci = torch.tensor([cm[c['crop_id']] for c in ontology['candidates']], device=crop_logits.device)
    di = torch.tensor([dm[c['disease_id']] for c in ontology['candidates']], device=crop_logits.device)
    pair = log_c[:, ci] + log_d_c[:, ci, di]
    disease = torch.logsumexp(log_c[:, :, None] + log_d_c, dim=1)
    return pair, disease
