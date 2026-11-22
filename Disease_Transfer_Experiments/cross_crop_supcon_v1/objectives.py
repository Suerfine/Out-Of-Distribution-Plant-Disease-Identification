"""Cross-crop adaptation of supervised contrastive learning."""
import torch
from torch import nn
from torch.nn import functional as F


def cross_crop_supcon(features, disease, crop, temperature=0.1, excluded_label=None):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    z = F.normalize(features.float(), dim=1)
    same = disease[:, None].eq(disease[None, :])
    different_crop = crop[:, None].ne(crop[None, :])
    positive = same & different_crop
    if excluded_label is not None:
        positive &= disease[:, None].ne(excluded_label)
    valid_anchor = positive.any(1)
    if not valid_anchor.any():
        return z.sum() * 0.0, {"eligible_anchors": 0, "positive_pairs": 0}
    # Neutral same-label/same-crop examples are not false negatives.
    comparison = (~same) | positive
    similarities = (z @ z.T) / temperature
    log_denominator = similarities.masked_fill(~comparison, -torch.inf).logsumexp(1)
    log_prob = similarities - log_denominator[:, None]
    selected = log_prob[valid_anchor].masked_fill(~positive[valid_anchor], 0.0)
    loss = -(selected.sum(1) / positive[valid_anchor].sum(1)).mean()
    return loss, {"eligible_anchors": int(valid_anchor.sum()),
                  "positive_pairs": int(positive.sum())}


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
