"""Administrative budget safety; no training or dataset access."""
import copy
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from independent_cnn_baseline import extend_gamma0_window as supervisor

from independent_cnn_baseline.extend_gamma0_window import (EXTENSION_ID, apply_extension,
    digest, os_lock, process_exited, read, validate, write)


class BudgetExtensionTests(unittest.TestCase):
    def fixture(self, root):
        state = {"state": "window_complete", "phase": "checkpointed", "pid": 123,
            "window_started_at_utc": "2026-11-01T02:31:50.263638+00:00", "window_hours": 10,
            "window_deadline_at_utc": "2026-11-01T12:31:50.263638+00:00", "max_trials": 8,
            "audit_identity": "unchanged", "objective": {"gamma": 0, "adaptive_decisions": "gamma_zero_only"},
            "historical_observations_imported": 0, "official_test_inference": False, "automatic_final_refit": False}
        write(root / "status.json", state)
        for name in ("identity.json", "study_config.json"):
            write(root / name, {"immutable": True})
        (root / "runner.pid").write_text("123")
        (root / "training.log").write_text("checkpointed\n")
        import sqlite3
        with contextlib.closing(sqlite3.connect(root / "study.sqlite3")) as db:
            db.execute("CREATE TABLE trials(number INTEGER,state TEXT)")
            db.execute("INSERT INTO trials VALUES(0,'RUNNING')")
            db.commit()
        folder = root / "trials/trial_000"
        write(folder / "pair_state.json", {"history": [{"epoch": 1}]})
        for target, epoch in (("crop", 2), ("disease", 1)):
            write(folder / target / "history.json", [{"epoch": i} for i in range(1, epoch + 1)])
            (folder / target / "last.pt").write_bytes(b"opaque full recovery state")
        auth = {"id": EXTENSION_ID, "user_instruction": "extend for 10 more hours",
            "window_started_at_utc": state["window_started_at_utc"],
            "previous_deadline_at_utc": state["window_deadline_at_utc"],
            "new_deadline_at_utc": "2026-11-01T22:31:50.263638+00:00", "audit_identity": "unchanged",
            "identity_file_sha256": digest(root / "identity.json"), "study_config_sha256": digest(root / "study_config.json")}
        return state, auth

    def test_extension_keeps_partial_pair_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state, auth = self.fixture(root)
            original = {str(p.relative_to(root)): digest(p) for p in (root / "trials").rglob("*") if p.is_file()}
            result = apply_extension(root, auth)
            self.assertEqual(result["window_hours"], 20)
            self.assertEqual(result["window_started_at_utc"], state["window_started_at_utc"])
            self.assertEqual(result["max_trials"], 8)
            self.assertEqual(apply_extension(root, auth), result)
            self.assertEqual(len(read(root / "status.json")["budget_extensions"]), 1)
            self.assertEqual(original, {str(p.relative_to(root)): digest(p) for p in (root / "trials").rglob("*") if p.is_file()})

    def test_rejects_live_state_pause_wrong_budget_and_scientific_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state, auth = self.fixture(root)
            for fields in ({"state": "running"}, {"state": "paused"}, {"window_hours": 11}, {"max_trials": 9},
                           {"historical_observations_imported": 1}, {"official_test_inference": True}):
                write(root / "status.json", {**state, **fields})
                with self.assertRaises(ValueError):
                    apply_extension(root, auth)
            write(root / "status.json", state)
            (root / "STOP.request").write_text("user pause")
            with self.assertRaisesRegex(ValueError, "pause"):
                apply_extension(root, auth)
            self.assertTrue((root / "STOP.request").exists())
            self.assertFalse((root / "extension_previous_status_20261101.json").exists())

    def test_rejects_rewritten_config_or_oversized_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state, auth = self.fixture(root)
            wrong = copy.deepcopy(auth); wrong["new_deadline_at_utc"] = "2026-11-02T22:31:50.263638+00:00"
            with self.assertRaises(ValueError):
                validate(state, wrong)
            write(root / "study_config.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "identity/config"):
                apply_extension(root, auth)

    def test_os_lock_excludes_duplicate_supervisors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runner.lock"
            with os_lock(path):
                with self.assertRaises(OSError):
                    with os_lock(path):
                        self.fail("A duplicate supervisor acquired the lock")
            with os_lock(path):
                pass

    def test_process_query_denial_is_never_exit_evidence(self):
        kernel = Mock()
        kernel.OpenProcess.return_value = 0
        with patch('ctypes.WinDLL', return_value=kernel, create=True), \
             patch('ctypes.get_last_error', return_value=5, create=True):
            with self.assertRaises(OSError):
                process_exited(123)
        with patch('ctypes.WinDLL', return_value=kernel, create=True), \
             patch('ctypes.get_last_error', return_value=87, create=True):
            self.assertTrue(process_exited(123))

    def test_supervisor_resumes_once_with_new_exact_deadline_after_exit(self):
        class FixedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 1, 13, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state, auth = self.fixture(root)
            auth.update(supervisor_sha256=digest(supervisor.__file__), initial_launcher_pid=122,
                        lock_roots=[f'lock_{i}' for i in range(5)])
            for folder in auth['lock_roots']:
                (root / folder).mkdir()
            write(root / supervisor.AUTH_FILE, auth)
            write(root / 'launch.json', {'launcher_pid': 456})
            with patch.object(supervisor, 'ROOT', root), patch.object(supervisor, 'PROJECT', root), \
                 patch.object(supervisor, 'datetime', FixedClock), patch.object(supervisor.time, 'sleep'), \
                 patch.object(supervisor, 'process_exited', return_value=True) as exited, \
                 patch.object(supervisor.subprocess, 'run', return_value=Mock(stdout='started')) as launch:
                supervisor.main()
                command = launch.call_args.args[0]
                self.assertIn('-Resume', command)
                self.assertNotIn('-ClearStop', command)
                self.assertNotIn('-Hours', command)
                self.assertEqual(command[-1], auth['new_deadline_at_utc'])
                self.assertEqual({c.args[0] for c in exited.call_args_list}, {122, 123})
                self.assertEqual(read(root / supervisor.PROGRESS_FILE)['state'], 'resume_launched')
                with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                    supervisor.main()
                self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
