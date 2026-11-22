import unittest
import torch
from evaluation_recovery import argsort_compatibility


class RankingCompatibilityTests(unittest.TestCase):
    def test_positional_dimension_preserves_exact_stable_ranking(self):
        original=torch.Tensor.argsort
        for dtype in (torch.float32,torch.float64):
            scores=torch.tensor([[2.,2.,1.],[.5,.2,.5]],dtype=dtype)
            expected=original(scores,dim=1,descending=True,stable=True)
            with argsort_compatibility():
                self.assertTrue(torch.equal(scores.argsort(1,descending=True,stable=True),expected))
                self.assertTrue(torch.equal(scores.argsort(dim=1,descending=True,stable=True),expected))
                self.assertEqual(scores.argsort(1,descending=True,stable=True).tolist(),[[0,1,2],[0,2,1]])
        self.assertIs(torch.Tensor.argsort,original)

    def test_adapter_restores_original_method_after_an_error(self):
        original=torch.Tensor.argsort
        with self.assertRaises(RuntimeError):
            with argsort_compatibility(): raise RuntimeError('test')
        self.assertIs(torch.Tensor.argsort,original)


if __name__=='__main__': unittest.main()
