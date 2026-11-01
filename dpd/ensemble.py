"""Common scale for mixing runs: log-probabilities over the 333 pairs. Shared by train/ensemble.py and the app."""

import numpy as np

# Multiplier that turns a member's raw scores into logits: CLIP cosines use CLIP's temperature, CNN scores are
# already sums of log-probabilities.
LOGIT_SCALE = {"clip_prompt_lora": 100.0, "clip_prompt_learning": 100.0, "cnn_independent": 1.0, "cnn_multitask": 1.0, "cnn_crop_norm": 1.0}


def to_logprob(scores: np.ndarray, method: str) -> np.ndarray:
    """log_softmax over pairs of LOGIT_SCALE[method] * scores; accepts one image (1-D) or a batch (2-D)."""
    s = LOGIT_SCALE[method] * np.atleast_2d(scores)
    s = s - s.max(axis=1, keepdims=True)
    out = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    return out[0] if np.ndim(scores) == 1 else out
