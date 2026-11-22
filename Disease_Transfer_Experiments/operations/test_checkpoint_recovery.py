"""Ensure a failed write cannot destroy the previously completed checkpoint."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from checkpoint_recovery import checked_save


class CheckpointRecoveryTests(unittest.TestCase):
    def test_retries_io_failure_and_round_trips(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'last.pt'; real_save=torch.save; calls=[]
            def flaky(payload,stream):
                calls.append(1)
                if len(calls)==1:
                    stream.write(b'partial'); raise RuntimeError('ios_base::badbit set: iostream stream error')
                return real_save(payload,stream)
            payload={'epoch':8,'weight':torch.arange(12)}
            with patch('torch.save',side_effect=flaky): checked_save(path,payload,retry_seconds=0)
            self.assertEqual(len(calls),2)
            loaded=torch.load(path,weights_only=True)
            self.assertEqual(loaded['epoch'],8); self.assertTrue(torch.equal(loaded['weight'],payload['weight']))

    def test_exhausted_failure_preserves_old_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'last.pt'; checked_save(path,{'epoch':8}); before=path.read_bytes()
            with patch('torch.save',side_effect=OSError('simulated storage error')):
                with self.assertRaises(OSError): checked_save(path,{'epoch':9},attempts=2,retry_seconds=0)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(torch.load(path,weights_only=True)['epoch'],8)


if __name__=='__main__': unittest.main()
