"""Compare regularized linear and approximate RBF multiclass SVMs."""
import numpy as np
import torch
from torch.nn import functional as F

from seen_hinge_head import main


def fit_svm(features, labels, strength):
    """Fit class-balanced one-vs-rest squared hinge with L2 regularization."""
    head = torch.nn.Linear(features.shape[1], 175, device=features.device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=80, line_search_fn='strong_wolfe')
    positive = F.one_hot(labels, num_classes=175).bool()
    n_positive = positive.sum(dim=0).clamp_min(1)
    n_negative = len(labels) - n_positive
    targets = torch.where(positive, 1., -1.)
    sample_weights = torch.where(positive, .5/n_positive, .5/n_negative)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        scores = head(features)
        hinge = (F.relu(1 - targets * scores).square() * sample_weights).sum(dim=0)
        loss = (.5 * head.weight.square().sum(dim=1) + strength * hinge).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return head


def compare(_initial, x, y, val, pairs, rank, report):
    x, val = (F.normalize(z, dim=1) for z in (x, val))
    generator = torch.Generator(device=x.device).manual_seed(42)
    landmarks = x[torch.randperm(len(x), device=x.device, generator=generator)[:768]]
    fold_rng = np.random.default_rng(42)
    fold = {pair:i%5 for i,pair in enumerate(fold_rng.permutation(np.unique(pairs)))}
    fold_ids = np.array([fold[pair] for pair in pairs])
    strengths = (10., 100., 1000.)
    score_weights = (1., 2., 4., 8., 12., 16., 24., 32.)
    kernel_rates = (.5, 2., 8.)
    arms = {'linear': [], 'rbf_nystrom': []}
    for gamma in (None, *kernel_rates):
        if gamma is None:
            name, features, validation = 'linear', x, val
        else:
            name = 'rbf_nystrom'

            def kernel(a, b):
                return torch.exp(-gamma * (2 - 2 * (a @ b.T)).clamp_min(0))

            gram = kernel(landmarks, landmarks)
            chol = torch.linalg.cholesky(gram + 1e-4 * torch.eye(len(gram), device=x.device))
            features, validation = (torch.linalg.solve_triangular(chol, kernel(z, landmarks).T,
                                                                    upper=False).T for z in (x, val))
        for strength in strengths:
            model = fit_svm(features, y, strength)
            with torch.inference_mode():
                logits = model(validation).cpu().numpy()
            assert np.isfinite(logits).all()
            assert np.std(logits, axis=0).mean() > 1e-3, 'SVM returned constant scores'
            print(f'{name} kernel_gamma={gamma} strength={strength}', flush=True)
            for weight in score_weights:
                params = dict(kernel_gamma=gamma, strength=strength, disease_weight=weight)
                arms[name].append((params, logits, rank(logits, weight)))

    results = {}
    for name, candidates in arms.items():
        held_logits = np.empty_like(candidates[0][1])
        held_top = np.empty_like(candidates[0][2])
        params = []
        for fold_id in range(5):
            test = fold_ids == fold_id
            train = ~test
            setting, logits, top = max(candidates, key=lambda item: (item[2][train, 0] == pairs[train]).sum())
            held_logits[test] = logits[test]
            held_top[test] = top[test]
            params.append(setting)
        results[name] = dict(**report(held_logits, held_top), fold_settings=params)
    return dict(landmarks=len(landmarks), **results)


if __name__ == '__main__':
    main(feature_experiment=compare)
