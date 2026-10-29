"""Group-app adapter for Independent CNN Baseline runs, including retrieval."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import CachedImages, IndependentNet, transforms


class IndependentCnnRunModel:
    logit_scale = 1.0
    score_help = "log P(crop) + log P(disease), after calibration"

    def __init__(self, run, pairs):
        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"]["gamma"])
        self.config = run["cfg"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp = self.device == "cuda" and self.config["runtime"]["amp"]
        self.transform = transforms(self.config)[1]
        self.nets = {}
        for target in ("crop", "disease"):
            net = IndependentNet(self.config, target, pretrained=False)
            net.load_state_dict(torch.load(Path(run["dir"]) / f"{target}_net.pt", map_location="cpu", weights_only=True))
            self.nets[target] = net.eval()

    @torch.no_grad()
    def predict(self, src):
        from dpd.cnn import compose_pair_scores
        from dpd.resize_cache import standardise

        x = self.transform(standardise(src, self.config["input"]["resize_cache"])).unsqueeze(0).to(self.device)
        features, logp = [], {}
        for target, net in self.nets.items():
            net.to(self.device)
            with torch.autocast(self.device, enabled=self.amp):
                f = net.backbone(x)
                logits = net.head(f)
            features.append(F.normalize(f.float(), dim=-1).cpu())
            # Match training/evaluation: retain raw FP32 logits, then log_softmax on CPU.
            logp[target] = F.log_softmax(logits.float().cpu(), dim=-1).numpy()
            net.cpu()
        embedding = F.normalize(torch.cat(features, dim=-1), dim=-1)[0].numpy()
        return embedding, compose_pair_scores(logp["crop"], logp["disease"], self.pairs)[0]

    @torch.no_grad()
    def train_bank(self, train):
        from dpd.resize_cache import ensure_resized

        output = Path(self.run["dir"]) / "train_features.npy"
        files = Path(self.run["dir"]) / "train_feature_files.json"
        import json
        expected = train.file.tolist()
        if output.exists() and files.exists() and json.loads(files.read_text()) == expected:
            bank = np.load(output)
            if len(bank) == len(train):
                return bank
        cache = ensure_resized(train.file, self.config["input"]["resize_cache"])
        data = DataLoader(CachedImages(train, cache, self.transform),
                          batch_size=self.config["runtime"]["inference_batch_size"], num_workers=0)
        features = []
        for net in self.nets.values():
            net.to(self.device)
            batches = []
            for x, _, _ in data:
                with torch.autocast(self.device, enabled=self.amp):
                    f = net.backbone(x.to(self.device))
                batches.append(F.normalize(f.float(), dim=-1).cpu())
            features.append(torch.cat(batches))
            net.cpu()
        bank = F.normalize(torch.cat(features, dim=-1), dim=-1).numpy()
        np.save(output, bank)
        files.write_text(json.dumps(expected), encoding="utf-8")
        return bank
