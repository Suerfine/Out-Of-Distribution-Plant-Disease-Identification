"""Apply an explicitly authorized budget extension after the pinned runner exits.

This administrative supervisor never trains, changes scientific inputs, clears
STOP.request, or edits the pinned runner. It resumes the same study once.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42"
AUTH_FILE = "extension_authorization_20261101.json"
PROGRESS_FILE = "extension_status_20261101.json"
EXTENSION_ID = "user_20261101_add_10_hours"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


@contextlib.contextmanager
def os_lock(path):
    path = Path(path)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def process_exited(pid):
    """Access denial is an error, never evidence that a process exited."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
            return True
        raise OSError(error, f"Cannot positively establish exit for PID {pid}")
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value != 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def validate(state, auth):
    if auth["id"] != EXTENSION_ID or auth["user_instruction"] != "extend for 10 more hours":
        raise ValueError("Explicit extension authorization is required")
    start = datetime.fromisoformat(auth["window_started_at_utc"])
    old = datetime.fromisoformat(auth["previous_deadline_at_utc"])
    new = datetime.fromisoformat(auth["new_deadline_at_utc"])
    if start.tzinfo is None or old != start + timedelta(hours=10) or new != old + timedelta(hours=10):
        raise ValueError("Only the authorized additional ten hours may be applied")
    if state["window_started_at_utc"] != auth["window_started_at_utc"]:
        raise ValueError("Study start changed")
    if state["audit_identity"] != auth["audit_identity"] or state["max_trials"] != 8:
        raise ValueError("Study identity or eight-trial limit changed")
    if state["objective"]["gamma"] != 0 or state["objective"]["adaptive_decisions"] != "gamma_zero_only":
        raise ValueError("Gamma-zero objective changed")
    if state["historical_observations_imported"] != 0 or state["official_test_inference"] or state["automatic_final_refit"]:
        raise ValueError("Validation-only scope changed")
    applied = any(x["id"] == auth["id"] for x in state.get("budget_extensions", []))
    expected_hours, expected_deadline = (20, new) if applied else (10, old)
    if state["window_hours"] != expected_hours or datetime.fromisoformat(state["window_deadline_at_utc"]) != expected_deadline:
        raise ValueError("Unexpected budget; do not overwrite or extend it again")
    return applied


def stable_files(root):
    paths = [root / "status.json", root / "runner.pid", root / "training.log", root / "study.sqlite3"]
    paths += list((root / "trials").rglob("*.json"))
    paths += list((root / "trials").rglob("*.pt"))
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def apply_extension(root, auth):
    """Caller must hold all five study locks and have proven process exit."""
    root = Path(root)
    state = read(root / "status.json")
    if (root / "STOP.request").exists() or state["state"] == "paused":
        raise ValueError("User pause preserved; no extension resume")
    applied = validate(state, auth)
    if state["state"] != "window_complete" or state["phase"] != "checkpointed":
        raise ValueError("Only a checkpointed cap stop can be extended")
    if applied:
        return state
    if digest(root / "identity.json") != auth["identity_file_sha256"] or digest(root / "study_config.json") != auth["study_config_sha256"]:
        raise ValueError("Immutable identity/config changed")
    for name, expected in read(root / "identity.json").get("code_sha256", {}).items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"Pinned scientific code changed: {name}")
    before = stable_files(root)
    with contextlib.closing(sqlite3.connect((root / "study.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        trials = db.execute("SELECT number,state FROM trials ORDER BY number").fetchall()
    if len(trials) > 8 or any(n >= 8 for n, _ in trials):
        raise ValueError("Fresh trial ceiling exceeded")
    for folder in sorted((root / "trials").glob("trial_*")):
        if (folder / "terminal.json").exists():
            continue
        pair = read(folder / "pair_state.json") if (folder / "pair_state.json").exists() else {"history": []}
        epochs = {}
        for target in ("crop", "disease"):
            hp = folder / target / "history.json"
            history = read(hp) if hp.exists() else []
            epochs[target] = history[-1]["epoch"] if history else 0
            if history and not (folder / target / "last.pt").is_file():
                raise ValueError(f"Missing recovery checkpoint: {target}")
        paired = len(pair["history"])
        if not (epochs["disease"] <= epochs["crop"] <= epochs["disease"] + 1 and paired <= min(epochs.values()) <= paired + 1):
            raise ValueError("Recovery histories do not describe a pending paired boundary")
    snapshot = root / "extension_previous_status_20261101.json"
    if snapshot.exists():
        if read(snapshot) != state:
            raise ValueError("Pre-extension audit already exists with different state")
    else:
        write(snapshot, state)
    event = {"id": auth["id"], "additional_hours": 10,
             "previous_deadline_at_utc": auth["previous_deadline_at_utc"],
             "new_deadline_at_utc": auth["new_deadline_at_utc"],
             "applied_at_utc": datetime.now(timezone.utc).isoformat()}
    state.update(window_hours=20, window_deadline_at_utc=auth["new_deadline_at_utc"],
                 budget_extensions=[*state.get("budget_extensions", []), event])
    write(root / "status.json", state)
    after = stable_files(root)
    if {k: v for k, v in before.items() if k != "status.json"} != {k: v for k, v in after.items() if k != "status.json"}:
        raise RuntimeError("Unexpected mutation outside the administrative status file")
    write(root / "extension_applied_20261101.json", {**event, "sqlite_trials": trials,
        "all_five_locks_acquired": True, "process_exit_confirmed": True,
        "checkpoint_and_trial_files_unchanged": True, "before_status_sha256": digest(snapshot),
        "after_status_sha256": digest(root / "status.json"),
        "identity_file_sha256": auth["identity_file_sha256"], "scientific_identity_migrated": False})
    return state


def main():
    root = ROOT
    auth = read(root / AUTH_FILE)
    progress = {"id": auth["id"], "supervisor_pid": os.getpid(), "state": "waiting_for_original_cap",
                "new_deadline_at_utc": auth["new_deadline_at_utc"]}
    with os_lock(root / "extension_supervisor.lock"):
        try:
            if digest(__file__) != auth["supervisor_sha256"]:
                raise ValueError("Extension supervisor changed after authorization was recorded")
            previous = read(root / PROGRESS_FILE) if (root / PROGRESS_FILE).exists() else {}
            if previous.get("resume_attempted") or previous.get("state") in ("launching_authorized_resume", "resume_launched"):
                progress.update(resume_attempted=True, previous_attempt=previous)
                raise RuntimeError("A resume was already attempted; inspect its identity before any retry")
            write(root / PROGRESS_FILE, progress)
            while True:
                state = read(root / "status.json")
                applied = validate(state, auth)
                if (root / "STOP.request").exists() or state["state"] == "paused":
                    progress["state"] = "user_pause_preserved"
                    break
                if state["state"] == "selection_complete":
                    progress["state"] = "selection_already_complete"
                    break
                if state["state"] == "failed":
                    raise RuntimeError("Controller failure requires inspection; no automatic recovery")
                if datetime.now(timezone.utc) >= datetime.fromisoformat(auth["new_deadline_at_utc"]):
                    progress["state"] = "extended_deadline_expired_no_launch"
                    break
                if applied and state["state"] != "window_complete":
                    progress.update(state="extension_already_active", controller_pid=state["pid"])
                    break
                if state["state"] != "window_complete":
                    time.sleep(20)
                    continue
                pids = {state["pid"], int((root / "runner.pid").read_text()), auth["initial_launcher_pid"]}
                if not all(process_exited(pid) for pid in pids):
                    time.sleep(5)
                    continue
                first = stable_files(root)
                time.sleep(5)
                if first != stable_files(root):
                    continue
                with contextlib.ExitStack() as locks:
                    try:
                        for folder in auth["lock_roots"]:
                            locks.enter_context(os_lock(PROJECT / folder / "runner.lock"))
                    except OSError:
                        time.sleep(5)
                        continue
                    apply_extension(root, auth)
                progress.update(state="launching_authorized_resume", resume_attempted=True)
                write(root / PROGRESS_FILE, progress)
                command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                    str(PROJECT / "independent_cnn_baseline/run_optuna_gamma0.ps1"),
                    "-Detach", "-Resume", "-DeadlineUtc", auth["new_deadline_at_utc"]]
                result = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW, check=True)
                progress.update(state="resume_launched", launcher_output=result.stdout.strip(), launch=read(root / "launch.json"))
                break
        except BaseException as error:
            progress.update(state="failed_requires_attention", error=repr(error))
            raise
        finally:
            progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write(root / PROGRESS_FILE, progress)


if __name__ == "__main__":
    main()
