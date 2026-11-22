"""Scientific failure modes that would invalidate the matched experiment."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from objectives import cross_crop_supcon
from protocol import EpochBatches, joint_scores, load_config, validate_roles, HERE


class ContrastiveTests(unittest.TestCase):
    def test_numpy_reference_and_gradients(self):
        torch.manual_seed(9)
        z = torch.randn(8, 7, requires_grad=True)
        d = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        c = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1])
        loss, detail = cross_crop_supcon(z, d, c, 0.2, excluded_label=3)
        x = z.detach().numpy().astype(np.float64)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        sims = x @ x.T / 0.2
        expected = []
        for i in range(8):
            positive = [j for j in range(8) if d[j] == d[i] and c[j] != c[i] and d[i] != 3]
            if positive:
                comparisons = [j for j in range(8) if d[j] != d[i] or j in positive]
                logdenom = np.logaddexp.reduce(sims[i, comparisons])
                expected.append(np.mean(logdenom-sims[i, positive]))
        self.assertAlmostEqual(float(loss.detach()), np.mean(expected), places=5)
        self.assertEqual(detail["eligible_anchors"], 4)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.norm()), 0)

    def test_no_cross_crop_positives_contributes_zero(self):
        z = torch.randn(4, 5, requires_grad=True)
        loss, detail = cross_crop_supcon(z, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]))
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(detail["eligible_anchors"], 0)
        loss.backward()
        self.assertEqual(float(z.grad.norm()), 0)

    def test_permutation_invariance(self):
        torch.manual_seed(4)
        z, d, c = torch.randn(8, 5), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), torch.tensor([0, 1]*4)
        p = torch.randperm(8)
        a = cross_crop_supcon(z, d, c)[0]
        b = cross_crop_supcon(z[p], d[p], c[p])[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_neutral_same_disease_same_crop_is_not_a_negative(self):
        z = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        # All one disease. Each anchor's denominator contains positives only.
        loss, detail = cross_crop_supcon(z, torch.zeros(3, dtype=torch.long), torch.tensor([0, 0, 1]))
        self.assertAlmostEqual(float(loss), np.log(2)/3, places=6)
        self.assertEqual(detail["eligible_anchors"], 3)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(HERE / "config.json")
        self.rows = [{"disease_id": d, "crop_id": c} for d in range(6) for c in range(3) for _ in range(5)]

    def test_pairs_distinct_diseases_and_matched_order(self):
        a, b = EpochBatches(self.rows, self.cfg, True), EpochBatches(self.rows, self.cfg, True)
        self.assertEqual(a.indices(), b.indices())
        self.assertEqual(len(a.indices()), len(self.rows))
        for batch in list(a)[:-1]:
            diseases = []
            for i in range(0, len(batch), 2):
                x, y = self.rows[batch[i]], self.rows[batch[i+1]]
                self.assertEqual(x["disease_id"], y["disease_id"])
                self.assertNotEqual(x["crop_id"], y["crop_id"])
                diseases.append(x["disease_id"])
            self.assertEqual(len(set(diseases)), 4)
        a.epoch = 1
        self.assertNotEqual(a.indices(), b.indices())

    def test_random_control_covers_every_image_once(self):
        sampler = EpochBatches(self.rows, self.cfg, False)
        self.assertEqual(sorted(sampler.indices()), list(range(len(self.rows))))

    def test_raw_logit_sum_ranks_like_log_probability_sum(self):
        torch.manual_seed(2)
        c, d = torch.randn(7, 3), torch.randn(7, 4)
        ontology = {"crop": [0, 1, 2], "disease": [0, 1, 2, 3],
                    "candidates": [{"crop_id": a, "disease_id": b} for a in range(3) for b in range(4)]}
        raw = joint_scores(c, d, ontology)
        reference = torch.stack([c.double().log_softmax(1)[:, x["crop_id"]]+d.double().log_softmax(1)[:, x["disease_id"]]
                                 for x in ontology["candidates"]], 1)
        self.assertTrue(torch.equal(raw.argsort(1), reference.argsort(1)))


class DataRoleTests(unittest.TestCase):
    def setUp(self):
        self.roles = {name: [SimpleNamespace(path=name, composition_id=comp, crop_id=0, disease_id=0)]
                      for name, comp in (("fit", 0), ("seen_validation", 0), ("development", 1), ("audit", 2))}
        self.hashes = {name: name for name in self.roles}

    def test_duplicate_content_is_rejected(self):
        validate_roles(self.roles, self.hashes)
        self.hashes["audit"] = self.hashes["fit"]
        with self.assertRaisesRegex(ValueError, "Duplicate content"):
            validate_roles(self.roles, self.hashes)

    def test_previously_learned_composition_is_rejected(self):
        self.roles["development"][0].composition_id = 0
        with self.assertRaisesRegex(ValueError, "Composition holdout leakage"):
            validate_roles(self.roles, self.hashes)


if __name__ == "__main__":
    unittest.main()
