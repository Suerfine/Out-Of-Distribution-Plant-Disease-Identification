"""Cross-crop adaptation of supervised contrastive learning."""
import torch
from torch import nn
from torch.nn import functional as F


class FeatureMemory:
    def __init__(self, capacity):
        self.capacity=capacity
        self.bank=None
    def add(self,z,d,c):
        values=(F.normalize(z.detach().float(),dim=1),d.detach().clone(),c.detach().clone())
        if self.bank is not None: values=tuple(torch.cat((a,b)) for a,b in zip(self.bank,values))
        self.bank=tuple(v[-self.capacity:].detach().clone() for v in values)
    def values(self): return self.bank


def cross_crop_supcon(features,disease,crop,temperature=0.1,excluded_label=None,memory=None):
    if temperature<=0: raise ValueError('Temperature must be positive')
    z=F.normalize(features.float(),dim=1)
    refs,rd,rc=z,disease,crop
    if memory is not None:
        mz,md,mc=memory
        refs=torch.cat((z,mz.detach()))
        rd=torch.cat((disease,md)); rc=torch.cat((crop,mc))
    same=disease[:,None].eq(rd[None,:])
    positive=same & crop[:,None].ne(rc[None,:])
    if excluded_label is not None: positive &= disease[:,None].ne(excluded_label)
    valid=positive.any(1)
    if not valid.any(): return z.sum()*0.,{'eligible_anchors':0,'positive_pairs':0}
    comparisons=(~same)|positive
    scores=(z[valid]@refs.T)/temperature
    mask=positive[valid]
    log_prob=scores-scores.masked_fill(~comparisons[valid],-torch.inf).logsumexp(1)[:,None]
    loss=-(log_prob.masked_fill(~mask,0).sum(1)/mask.sum(1)).mean()
    return loss,{'eligible_anchors':int(valid.sum()),'positive_pairs':int(positive.sum())}

class FactorCNN(nn.Module):
    def __init__(self, cfg, classes, pretrained=True):
        super().__init__()
        import timm
        self.net = timm.create_model(cfg["model"], pretrained=False, num_classes=classes)
        if pretrained:
            from safetensors.torch import load_file
            state = load_file(cfg["pretrained_weights"])
            state.pop("head.fc.weight")
            state.pop("head.fc.bias")
            missing, extra = self.net.load_state_dict(state, strict=False)
            if set(missing) != {"head.fc.weight", "head.fc.bias"} or extra:
                raise ValueError(f"Unexpected pretrained conversion: {missing}, {extra}")
        self.projector = nn.Sequential(nn.Linear(self.net.num_features, cfg["projection_hidden"]),
                                       nn.ReLU(), nn.Linear(cfg["projection_hidden"], cfg["projection_dim"]))

    def forward(self, images):
        features = self.net.forward_head(self.net.forward_features(images), pre_logits=True)
        return self.net.get_classifier()(features), self.projector(features)
