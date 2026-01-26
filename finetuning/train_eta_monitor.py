#!/usr/bin/env python3
# train_eta_monitor.py — read training logs from stdin, print ETA every N steps.

import sys, re, time, argparse, math

parser = argparse.ArgumentParser()
parser.add_argument("--total", type=int, required=True, help="Total steps (same as --iters)")
parser.add_argument("--every", type=int, default=50, help="Print ETA every N steps")
parser.add_argument("--warmup", type=int, default=20, help="Ignore first K steps in ETA calc")
args = parser.parse_args()

# Match common step prints (case-insensitive):
#   "step 123", "step 123/3000", "iter 123", "iteration: 123", "global_step=123", etc.
PAT = re.compile(
    r"(?:step|iter|iteration|global_step)\s*[:= ]\s*(\d+)(?:\s*/\s*(\d+))?",
    re.IGNORECASE,
)

total = args.total
every = args.every
warmup = args.warmup

t0 = time.time()
last_step = 0
ema_rate = None  # exponential moving average of steps/sec
alpha = 0.2      # smoothing factor

def fmt_eta(seconds: float) -> str:
    if math.isinf(seconds) or seconds < 0:
        return "??:??:??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

for line in sys.stdin:
    # forward the original log line to stdout unchanged
    sys.stdout.write(line)
    sys.stdout.flush()

    m = PAT.search(line)
    if not m:
        continue

    step = int(m.group(1))
    if step <= 0 or step == last_step:
        continue

    # If the log shows "step X/Y", prefer that Y as total
    if m.group(2):
        try:
            total = int(m.group(2))
        except Exception:
            pass

    now = time.time()
    elapsed = now - t0
    if elapsed <= 0:
        continue

    # instantaneous rate
    rate = step / elapsed  # steps per second
    ema_rate = rate if ema_rate is None else (alpha * rate + (1 - alpha) * ema_rate)

    # only print every N steps, and skip early warmup for cleaner ETA
    if step >= warmup and step % every == 0:
        rem = max(total - step, 0)
        eta_sec = rem / ema_rate if ema_rate and ema_rate > 0 else float("inf")
        print(f"[ETA] step {step}/{total} | {ema_rate:.2f} it/s | ETA {fmt_eta(eta_sec)}", flush=True)

    last_step = step
