"""Execute generated MBPP solutions against their assert-tests in isolated subprocesses (pass@1)."""
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 10


def _run_one(program):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        env = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""}
        try:
            r = subprocess.run(["python3", "-I", path], cwd=d, env=env, capture_output=True, timeout=TIMEOUT,
                               stdin=subprocess.DEVNULL)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def run_tests(codes, items, workers=16):
    """codes: list of extracted code strings; items: eval dicts with 'tests' and 'setup'. Returns list[bool]."""
    programs = []
    for code, it in zip(codes, items):
        programs.append((it.get("setup") or "") + "\n" + code + "\n\n" + "\n".join(it["tests"]) + "\n")
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(_run_one, programs))
