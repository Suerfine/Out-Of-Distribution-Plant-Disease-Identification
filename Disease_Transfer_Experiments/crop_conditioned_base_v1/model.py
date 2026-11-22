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
    def __init__(self,cfg,ontology,warm_start=True):
        super().__init__()
        import json,sys
        project=Path(__file__).resolve().parents[2]/'Out-Of-Distribution-Plant-Disease-Identification'
        sys.path.insert(0,str(project))
        from independent_cnn_baseline.models import IndependentNet
        config=json.loads((Path(cfg['baseline_run'])/'config.json').read_text())
        if config['model']['backbone']!='convnext_base.fb_in22k_ft_in1k': raise ValueError('Expected the saved Base model')
        self.net=IndependentNet(config,'disease',pretrained=False)
        if self.net.backbone.num_features!=1024: raise ValueError('Expected ConvNeXt-Base width 1024')
        if warm_start: self.net.load_state_dict(torch.load(Path(cfg['baseline_run'])/'disease_net.pt',map_location='cpu',weights_only=True),strict=True)
        self.conditioner=CropConditioner(len(ontology['crop']),1024,cfg['conditioning_dim'],cfg['conditioning_scale'])

    def forward(self,images,crop=None):
        features=self.net.head[0](self.net.backbone(images))
        return self.conditioner(features,self.net.head[1],crop)


def conditional_scores(crop_logits, conditional_logits, ontology):
    """No labels: P(c|x) P(d|x,c), plus the disease marginal over all crops.

    Crop-dependent normalizers matter: adding the raw logits would give a
    different, incorrect ranking. FP32 log probabilities and pair addition match the saved Base scorer;
    export in float64 so frozen calibration does not lose precision.
    """
    crops, diseases = len(ontology['crop']), len(ontology['disease'])
    if conditional_logits.shape != (len(crop_logits), crops, diseases):
        raise ValueError('Conditional score shape mismatch')
    log_c = F.log_softmax(crop_logits.float(), dim=1)
    log_d_c = F.log_softmax(conditional_logits.float(), dim=2)
    cm = {v:i for i,v in enumerate(ontology['crop'])}
    dm = {v:i for i,v in enumerate(ontology['disease'])}
    ci = torch.tensor([cm[c['crop_id']] for c in ontology['candidates']], device=crop_logits.device)
    di = torch.tensor([dm[c['disease_id']] for c in ontology['candidates']], device=crop_logits.device)
    pair = (log_c[:, ci] + log_d_c[:, ci, di]).double()
    disease = torch.logsumexp(log_c.double()[:, :, None] + log_d_c.double(), dim=1)
    return pair, disease
