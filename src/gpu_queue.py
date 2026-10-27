"""Single work queue over shared GPUs.

Jobs file: one job per line: `<mem_gb>\t<out_json_path>\t<command>`. The scheduler repeatedly picks the first job
(in file order = priority) whose output does not exist, that is not running, and whose memory need fits the free
memory on some GPU (per nvidia-smi, minus a margin), then launches it with CUDA_VISIBLE_DEVICES set. At most one launch
per GPU every LAUNCH_GAP seconds (lets the new process allocate before the next decision). Failed jobs are retried up
to MAX_TRIES times. The jobs file is re-read every loop, so jobs can be appended while the queue runs.
"""
import fcntl
import os
import subprocess
import sys
import time

MARGIN_GB = 0.25
LAUNCH_GAP = int(os.environ.get("LAUNCH_GAP", "75"))
MAX_TRIES = 40
MAX_RUNNING = int(os.environ.get("MAX_RUNNING", "14"))


def free_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(l.split(",")[0]): float(l.split(",")[1]) / 1024 for l in out.strip().splitlines()}


def locked_elsewhere(out):
    """True if another process (e.g. a job orphaned by a queue restart) holds the job's output lock."""
    lp = out + ".lock"
    if not os.path.exists(lp):
        return False
    with open(lp, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        except OSError:
            return True


def read_jobs(path):
    jobs = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        mem, out, cmd = line.split("\t")
        jobs.append((float(mem), out, cmd))
    return jobs


def main(jobs_path, log_dir, gpus):
    os.makedirs(log_dir, exist_ok=True)
    running = {}  # out -> (proc, gpu, mem)
    tries = {}
    last_launch = {g: 0.0 for g in gpus}
    while True:
        for out, (p, g, m) in list(running.items()):
            if p.poll() is not None:
                del running[out]
                if p.returncode != 0 or not os.path.exists(out):
                    tries[out] = tries.get(out, 0) + 1
                    print(f"[{time.strftime('%H:%M:%S')}] FAIL rc={p.returncode} try={tries[out]} {out}", flush=True)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] OK {out}", flush=True)
        jobs = read_jobs(jobs_path)
        undone = [j for j in jobs if not os.path.exists(j[1]) and j[1] not in running]
        orphans = [j for j in undone if locked_elsewhere(j[1])]  # running, launched by an earlier queue instance
        pending = [j for j in undone if tries.get(j[1], 0) < MAX_TRIES and j not in orphans]
        n_active = len(running) + len(orphans)
        if not pending and not running and not orphans:
            print("ALL DONE", flush=True)
            return
        fm = free_mem()
        now = time.time()
        for g in gpus:
            if n_active >= MAX_RUNNING:
                break
            if now - last_launch[g] < LAUNCH_GAP:
                continue
            avail = fm.get(g, 0) - MARGIN_GB
            for mem, out, cmd in pending:
                if mem <= avail:
                    name = os.path.basename(out).replace(".json", "")
                    lf = open(os.path.join(log_dir, name + ".log"), "a")
                    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                               RESERVE_GB=f"{mem - 0.3:.2f}")
                    p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env)
                    running[out] = (p, g, mem)
                    last_launch[g] = now
                    n_active += 1
                    pending.remove((mem, out, cmd))
                    print(f"[{time.strftime('%H:%M:%S')}] LAUNCH gpu{g} free={fm.get(g):.1f}GB need={mem} {name}", flush=True)
                    break
        time.sleep(10)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")])
