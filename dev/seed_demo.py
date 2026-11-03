#!/usr/bin/env python3
"""Fill a data directory with made-up history so the dashboard can be tried without GPUs.

    python3 dev/seed_demo.py /tmp/gpu-demo
    PATH="$PWD/tests/fake_bin:$PATH" python3 gpu_dash.py --data-dir /tmp/gpu-demo

Generates 3 GPUs x 95 days of one-minute samples: nine users, long training runs,
notebooks left idle on a GPU, GPUs shared by two people, and outages of each kind
(a reboot, a spell of nvidia-smi errors, and logger crashes restarted by cron).
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gpu_dash  # noqa: E402

USERS = ["alice", "bob", "chen", "dana", "eli", "fatima", "gus", "hana", "ivan"]
WEIGHTS = [30, 18, 10, 12, 8, 7, 5, 4, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--days", type=int, default=95)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    os.makedirs(a.data_dir, exist_ok=True)
    store = gpu_dash.Store(os.path.join(a.data_dir, "usage.db"))
    now = int(time.time()) // 60 * 60 - 120
    start = now - a.days * 86400
    first_seen = {u: start + i * 3 * 86400 for i, u in enumerate(USERS)}
    # times without samples, and the events that explain them
    boot = start - 2 * 86400
    reboot = (now - 9 * 86400, now - 9 * 86400 + 7 * 3600)        # node down, then rebooted
    smi_fail = (now - 4 * 86400 - 3 * 3600, now - 4 * 86400 - 3 * 3600 + 40 * 60)  # driver trouble
    crashes = [(now - 20 * 86400, now - 20 * 86400 + 12 * 60),       # logger killed, cron restarted it
               (now - 6 * 3600, now - 6 * 3600 + 14 * 60)]
    gaps = [reboot, smi_fail] + crashes
    events = [(start, "start", '{"boot": %d}' % boot),
              (reboot[1], "start", '{"boot": %d}' % (reboot[1] - 240))]
    events += [(t, "error", "nvidia-smi failed: Unable to determine the device handle for GPU0000:41:00.0: "
                            "Unknown Error") for t in range(smi_fail[0], smi_fail[1], 600)]
    events += [(b, "start", '{"boot": %d}' % (reboot[1] - 240 if b > reboot[1] else boot)) for _, b in crashes]

    def is_down(ts):
        return any(a <= ts < b for a, b in gaps)

    procs = {}  # (ts, gpu) -> [(user, pid, mem, sm)]
    for g in range(3):
        t = start
        while t < now:
            t += int(rnd.expovariate(1 / (5 * 3600)))
            cands = [u for u in USERS if first_seen[u] <= t] or USERS[:1]
            user = rnd.choices(cands, [WEIGHTS[USERS.index(c)] for c in cands])[0]
            dur = int(min(60, rnd.lognormvariate(1.6, 0.9)) * 3600)
            idle = (user == "chen" and rnd.random() < 0.7) or rnd.random() < 0.08
            mem = rnd.choice([8, 16, 24, 40, 60]) * 1024
            sm = rnd.uniform(0, 3) if idle else rnd.uniform(55, 92)
            # sometimes a second person squeezes onto the same GPU for part of the run
            partner = rnd.choice([c for c in cands if c != user] or [user]) if rnd.random() < 0.15 else None
            p_start, p_end = t + rnd.randint(0, dur // 2), t + dur
            p_idle = rnd.random() < 0.4
            for ts in range(t - t % 60, min(t + dur, now), 60):
                if is_down(ts):
                    continue
                jitter = rnd.uniform(-6, 6) if not idle else 0
                procs.setdefault((ts, g), []).append((user, 10000 + g, mem, max(0.0, sm + jitter)))
                if partner and p_start <= ts < p_end:
                    procs[(ts, g)].append((partner, 20000 + g, 12 * 1024,
                                           rnd.uniform(0, 2) if p_idle else rnd.uniform(20, 40)))
            t += dur

    rows_g, rows_p, seen = [], [], {}
    for g in range(3):
        for ts in range(start, now, 60):
            if is_down(ts):
                continue
            here = procs.get((ts, g), [])
            util = min(100.0, sum(p[3] for p in here))
            power = 58 + util * 3.2 + rnd.uniform(-8, 8) if here else rnd.uniform(52, 62)
            rows_g.append((ts, 60, g, round(util), sum(p[2] for p in here) + 4, 81920,
                           round(power, 1), round(31 + util * 0.38 + rnd.uniform(-2, 2))))
            for user, pid, mem, sm in here:
                rows_p.append((ts, g, user, pid, mem, round(sm, 1)))
                seen.setdefault(user, ts)
    with store.con:
        store.con.executemany("INSERT INTO gpu (ts, dt, gpu, util, mem_used, mem_total, power, temp) "
                              "VALUES (?,?,?,?,?,?,?,?)", rows_g)
        store.con.executemany("INSERT INTO proc (ts, gpu, user, pid, mem, sm) VALUES (?,?,?,?,?,?)", rows_p)
        store.con.executemany("INSERT OR IGNORE INTO users VALUES (?,?)", seen.items())
        store.con.executemany("INSERT INTO events VALUES (?,?,?)", events)
    print("wrote %d GPU samples and %d process samples to %s"
          % (len(rows_g), len(rows_p), store.path))


if __name__ == "__main__":
    main()
