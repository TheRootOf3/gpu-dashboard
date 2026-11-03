#!/usr/bin/env python3
"""
gpu_dash.py - per-user GPU usage logger and web dashboard for a single shared node.

Needs no root and no extra packages: only Python 3.7+ and nvidia-smi.

  python3 gpu_dash.py                  # log every 60 s and serve http://127.0.0.1:8050
  python3 gpu_dash.py --report 7d      # per-user usage for 24h/7d/30d/90d (add --csv for CSV)
  python3 gpu_dash.py --export DIR     # copy the database and every sample (CSV) into DIR

Data is kept in --data-dir (default: $GPU_DASH_DIR, else ~/.gpu-dash). Only one copy runs
per data directory - extra copies exit at once - so it is safe to (re)start it from your
own crontab every few minutes as a watchdog. See README.md.
"""
import argparse
import csv
import datetime as dt
import fcntl
import io
import json
import os
import pwd
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

IDLE_UTIL = 5        # a user holding a GPU below this utilization (%) counts as idle
LIVE_MAX_AGE = 5     # seconds a live nvidia-smi snapshot may be reused
HIST_MAX_AGE = 30    # seconds a history query result may be reused
SCHEMA_VERSION = 3

# range -> (timeline bin seconds, column bucket, number of columns)
RANGES = {
    "24h": (600, "hour", 24),
    "7d": (3600, "day", 7),
    "30d": (4 * 3600, "day", 30),
    "90d": (12 * 3600, "day", 90),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS gpu   (ts INTEGER, dt REAL, gpu INTEGER, util REAL, mem_used REAL,
                                  mem_total REAL, power REAL, temp REAL);
CREATE TABLE IF NOT EXISTS proc  (ts INTEGER, gpu INTEGER, user TEXT, pid INTEGER, mem REAL,
                                  sm REAL);
CREATE TABLE IF NOT EXISTS users (user TEXT PRIMARY KEY, first_seen INTEGER);
CREATE TABLE IF NOT EXISTS events (ts INTEGER, kind TEXT, detail TEXT);  -- logger starts, failed samples
CREATE INDEX IF NOT EXISTS gpu_ts    ON gpu (ts, gpu);
CREATE INDEX IF NOT EXISTS proc_ts   ON proc(ts, gpu);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""
# Columns added after the first version; older databases are upgraded in place.
MIGRATIONS = {"gpu": [("power", "REAL"), ("temp", "REAL")], "proc": [("sm", "REAL")]}

GPU_FIELDS = ["index", "uuid", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "name"]
BASIC_GPU_FIELDS = ["index", "uuid", "utilization.gpu", "memory.used", "memory.total", "name"]

# CSV exports: query (first column is the rowid used for paging) and header
CSV_EXPORTS = {
    "gpu": ("SELECT rowid, datetime(ts, 'unixepoch', 'localtime'), ts, gpu, util, mem_used, "
            "mem_total, power, temp, dt FROM gpu WHERE rowid > ? ORDER BY rowid LIMIT ?",
            ["time", "unix_time", "gpu", "util_pct", "mem_used_mib", "mem_total_mib",
             "power_w", "temp_c", "sample_seconds"], "gpu-samples"),
    "proc": ("SELECT rowid, datetime(ts, 'unixepoch', 'localtime'), ts, gpu, user, pid, mem, sm "
             "FROM proc WHERE rowid > ? ORDER BY rowid LIMIT ?",
             ["time", "unix_time", "gpu", "user", "pid", "mem_mib", "sm_util_pct"],
             "process-samples"),
    "events": ("SELECT rowid, datetime(ts, 'unixepoch', 'localtime'), ts, kind, detail "
               "FROM events WHERE rowid > ? ORDER BY rowid LIMIT ?",
               ["time", "unix_time", "event", "detail"], "logger-events"),
}
SUMMARY_HEADER = ["user", "gpu_hours", "share_pct", "avg_util_pct", "idle_held_hours",
                  "shared_time_pct", "energy_kwh", "peak_mem_gb", "last_active"]


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, file=sys.stderr, flush=True)


def _boot_time():
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


BOOT_TIME = _boot_time()
CLK_TCK = os.sysconf("SC_CLK_TCK")


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None  # "[N/A]", "[Not Supported]", "-", ...


def r1(x):
    return None if x is None else round(x, 1)


# --------------------------------------------------------------------------- reading the GPUs

def run(args, timeout=30):
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("%s failed: %s" % (" ".join(args[:2]), (r.stderr or r.stdout).strip()[:300]))
    return r.stdout


def smi_csv(query):
    out = run(["nvidia-smi", query, "--format=csv,noheader,nounits"])
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def parse_pmon(text):
    """Per-process SM utilization from `nvidia-smi pmon -c 1 -s u`, as {(gpu, pid): percent}.

    Columns differ between driver versions, so they are read from the header line."""
    cols, out = None, {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            toks = line.lstrip()[1:].split()
            if "gpu" in toks and "pid" in toks and "sm" in toks:
                cols = toks
            continue
        if not cols:
            continue
        rec = dict(zip(cols, line.split()))
        try:
            key = (int(rec["gpu"]), int(rec["pid"]))
        except (KeyError, ValueError):
            continue  # "-": no process on this GPU
        out[key] = max(out.get(key, 0.0), num(rec.get("sm")) or 0.0)  # "-" = no activity
    return out


def proc_info(pid):
    """Owner, command line and running time of a process, read from /proc."""
    info = {"user": "(unknown)", "cmd": "", "age": None}
    base = "/proc/%d" % pid
    try:
        uid = os.stat(base).st_uid
        try:
            info["user"] = pwd.getpwuid(uid).pw_name
        except KeyError:
            info["user"] = str(uid)
    except OSError:
        pass
    try:
        with open(base + "/cmdline", "rb") as f:
            info["cmd"] = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()[:400]
    except OSError:
        pass
    try:
        with open(base + "/stat") as f:
            stat = f.read()
        started = BOOT_TIME + int(stat[stat.rfind(")") + 2:].split()[19]) / CLK_TCK
        info["age"] = max(0, int(time.time() - started))
    except (OSError, ValueError, IndexError):
        pass
    return info


class Sampler:
    """Takes snapshots through nvidia-smi and remembers which optional queries work here."""

    def __init__(self):
        self.fields = GPU_FIELDS
        self.pmon_failures = 0
        self.lock = threading.Lock()

    def _gpu_rows(self):
        try:
            rows = smi_csv("--query-gpu=" + ",".join(self.fields))
        except RuntimeError as err:
            if self.fields is BASIC_GPU_FIELDS:
                raise
            rows = smi_csv("--query-gpu=" + ",".join(BASIC_GPU_FIELDS))  # raises if smi is broken
            self.fields = BASIC_GPU_FIELDS
            log("nvidia-smi cannot report power/temperature here (%s); logging without them" % err)
        n = len(self.fields) - 1  # the name comes last and may itself contain commas
        out = []
        for row in rows:
            if len(row) >= len(self.fields):
                rec = dict(zip(self.fields[:n], row[:n]))
                rec["name"] = ", ".join(row[n:])
                out.append(rec)
        return out

    def _pmon(self):
        if self.pmon_failures >= 3:
            return {}  # not supported here (e.g. MIG or an old driver); stop trying
        try:
            res = parse_pmon(run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], timeout=20))
            self.pmon_failures = 0
            return res
        except (RuntimeError, OSError, subprocess.SubprocessError) as err:
            self.pmon_failures += 1
            if self.pmon_failures == 3:
                log("per-process utilization unavailable (%s); using whole-GPU utilization" % err)
            return {}

    def snapshot(self):
        with self.lock:
            gpus = {}
            for r in self._gpu_rows():
                gpus[r["uuid"]] = {
                    "index": int(r["index"]), "name": r["name"],
                    "util": num(r.get("utilization.gpu")), "mem_used": num(r.get("memory.used")),
                    "mem_total": num(r.get("memory.total")), "temp": num(r.get("temperature.gpu")),
                    "power": num(r.get("power.draw")), "power_limit": num(r.get("power.limit")),
                    "procs": [],
                }
            apps = smi_csv("--query-compute-apps=gpu_uuid,pid,used_memory")
            sm = self._pmon() if apps else {}
            for row in apps:
                if len(row) < 3 or row[0] not in gpus:
                    continue
                try:
                    pid = int(row[1])
                except ValueError:
                    continue
                g = gpus[row[0]]
                # sm stays None (unknown) if pmon could not see this process
                p = {"pid": pid, "mem": num(row[2]), "sm": sm.get((g["index"], pid))}
                p.update(proc_info(pid))
                g["procs"].append(p)
        out = sorted(gpus.values(), key=lambda g: g["index"])
        for g in out:
            g["procs"].sort(key=lambda p: -(p["mem"] or 0))
        return out


# --------------------------------------------------------------------------- storage

class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.con = sqlite3.connect(path, check_same_thread=False, timeout=60)
        with self.lock, self.con:
            self.con.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():
                have = {r[1] for r in self.con.execute("PRAGMA table_info(%s)" % table)}
                for name, typ in cols:
                    if name not in have:
                        self.con.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, name, typ))
            self.con.execute("PRAGMA user_version = %d" % SCHEMA_VERSION)

    def insert(self, ts, interval, gpus):
        rows_g = [(ts, interval, g["index"], g["util"], g["mem_used"], g["mem_total"],
                   g["power"], g["temp"]) for g in gpus]
        rows_p = [(ts, g["index"], p["user"], p["pid"], p["mem"], p["sm"])
                  for g in gpus for p in g["procs"]]
        with self.lock, self.con:
            self.con.executemany("INSERT INTO gpu (ts, dt, gpu, util, mem_used, mem_total, power, temp) "
                                 "VALUES (?,?,?,?,?,?,?,?)", rows_g)
            self.con.executemany("INSERT INTO proc (ts, gpu, user, pid, mem, sm) VALUES (?,?,?,?,?,?)",
                                 rows_p)
            self.con.executemany("INSERT OR IGNORE INTO users VALUES (?,?)",
                                 {(r[2], ts) for r in rows_p})

    def event(self, ts, kind, detail=""):
        with self.lock, self.con:
            self.con.execute("INSERT INTO events VALUES (?,?,?)", (ts, kind, detail))

    def slots(self):
        """Stable colour slot per user: the order in which users were first seen."""
        with self.lock:
            rows = self.con.execute("SELECT user FROM users ORDER BY first_seen, user").fetchall()
        return {u: i for i, (u,) in enumerate(rows)}

    def info(self):
        with self.lock:
            c = self.con
            n_gpu = c.execute("SELECT COUNT(*) FROM gpu").fetchone()[0]
            n_proc = c.execute("SELECT COUNT(*) FROM proc").fetchone()[0]
            n_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            first, last = c.execute("SELECT MIN(ts), MAX(ts) FROM gpu").fetchone()
        size = sum(os.path.getsize(p) for p in (self.path, self.path + "-journal", self.path + "-wal")
                   if os.path.exists(p))
        days = (last - first) / 86400 if first is not None else 0
        return {"path": self.path, "bytes": size, "gpu_rows": n_gpu, "proc_rows": n_proc,
                "users": n_users, "first": first, "last": last,
                "bytes_per_day": size / days if days >= 1 else None}

    def backup(self, dest):
        """Consistent copy of the whole database, safe while the logger is writing."""
        if os.path.exists(dest):
            os.remove(dest)
        dst = sqlite3.connect(dest)
        try:
            with self.lock:
                self.con.backup(dst)
        finally:
            dst.close()

    def iter_csv(self, table, batch=50000):
        """Every row of a table as CSV, in chunks, without blocking the logger for long."""
        sql, header, _ = CSV_EXPORTS[table]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        last = 0
        while True:
            with self.lock:
                rows = self.con.execute(sql, (last, batch)).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            w.writerows(r[1:] for r in rows)
            yield buf.getvalue().encode()
            buf.seek(0)
            buf.truncate()
        if buf.tell():
            yield buf.getvalue().encode()


# --------------------------------------------------------------------------- app

class App:
    def __init__(self, store, interval, sampler=None):
        self.store, self.interval = store, interval
        self.sampler = sampler or Sampler()
        self.host = socket.gethostname()
        self._live = (0.0, None, None)  # (time, gpus, error)
        self._live_lock = threading.Lock()
        self._hist = {}
        self._hist_lock = threading.Lock()
        self._last_err = (0.0, None)  # (time, message) of the last error written to the events table

    def sample_once(self):
        t0 = time.time()
        try:
            gpus = self.sampler.snapshot()
            with self._live_lock:
                self._live = (t0, gpus, None)
            self.store.insert(int(t0), self.interval, gpus)
            self._last_err = (0.0, None)
        except Exception as e:  # keep sampling whatever happens
            msg = str(e)[:300]
            with self._live_lock:
                self._live = (t0, None, msg)
            log("sample failed:", msg)
            # remember failures (at most one row per 10 minutes) so the uptime chart can explain gaps
            if msg != self._last_err[1] or t0 - self._last_err[0] >= 600:
                self._last_err = (t0, msg)
                try:
                    self.store.event(int(t0), "error", msg)
                except Exception:
                    pass
        return t0

    def sample_forever(self):
        while True:
            t0 = self.sample_once()
            time.sleep(max(1.0, self.interval - (time.time() - t0)))

    def live(self):
        with self._live_lock:
            t, gpus, err = self._live
            if time.time() - t > LIVE_MAX_AGE:
                try:
                    gpus, err = self.sampler.snapshot(), None
                except Exception as e:
                    gpus, err = None, str(e)
                t = time.time()
                self._live = (t, gpus, err)
        return {"host": self.host, "now": int(t), "interval": self.interval,
                "tz_offset": time.localtime(t).tm_gmtoff, "idle_util": IDLE_UTIL,
                "gpus": gpus or [], "error": err, "slots": self.store.slots()}

    def history(self, rng):
        if rng not in RANGES:
            rng = "7d"
        with self._hist_lock:
            hit = self._hist.get(rng)
            if hit and time.time() - hit[0] < HIST_MAX_AGE:
                return hit[1]
        res = self._compute_history(rng)
        with self._hist_lock:
            self._hist[rng] = (time.time(), res)
        return res

    def _compute_history(self, rng, now=None):
        bin_s, kind, _ = RANGES[rng]
        now = time.time() if now is None else now
        starts, fmt = range_window(rng, now)
        since, end = int(starts[0].timestamp()), int(starts[-1].timestamp())
        keys = [s.strftime(fmt) for s in starts[:-1]]
        nbins = -(-(end - since) // bin_s)
        windows = {r: int(range_window(r, now)[0][0].timestamp()) for r in RANGES}

        with self.store.lock:
            c = self.store.con
            c.execute("DROP TABLE IF EXISTS temp.s")
            # One row per (sample, GPU, user). When users share a GPU:
            #   sec      - their share of the GPU time, split by GPU memory held (evenly if unknown)
            #   cshare   - their share of the compute, split by per-process SM use (else as sec)
            #   own_util - their own SM utilization if known, else the whole GPU's
            c.execute("""
                CREATE TEMP TABLE s AS
                WITH u AS (SELECT ts, gpu, user, SUM(mem) AS mem, SUM(sm) AS sm FROM proc
                           WHERE ts >= ? AND ts < ? GROUP BY ts, gpu, user),
                     n AS (SELECT ts, gpu, COUNT(*) AS k, COUNT(mem) AS kmem, SUM(mem) AS tmem,
                                  COUNT(sm) AS ksm, SUM(sm) AS tsm
                           FROM u GROUP BY ts, gpu)
                SELECT u.ts, u.gpu, u.user, u.mem, g.dt, g.power, n.k,
                       g.dt * CASE WHEN n.kmem = n.k AND n.tmem > 0 THEN u.mem / n.tmem
                                   ELSE 1.0 / n.k END AS sec,
                       CASE WHEN n.ksm = n.k AND n.tsm > 0 THEN u.sm / n.tsm
                            WHEN n.kmem = n.k AND n.tmem > 0 THEN u.mem / n.tmem
                            ELSE 1.0 / n.k END AS cshare,
                       CASE WHEN u.sm IS NOT NULL THEN MIN(u.sm, 100.0) ELSE g.util END AS own_util
                FROM u JOIN n ON n.ts = u.ts AND n.gpu = u.gpu
                       JOIN gpu g ON g.ts = u.ts AND g.gpu = u.gpu""", (since, end))
            users = c.execute("""
                SELECT user, SUM(sec),
                       SUM(CASE WHEN own_util < ? THEN sec ELSE 0 END),
                       SUM(own_util * sec) / SUM(CASE WHEN own_util IS NOT NULL THEN sec END),
                       MAX(mem), MAX(ts),
                       SUM(power * dt * cshare),
                       SUM(CASE WHEN k > 1 THEN dt ELSE 0 END) / SUM(dt)
                FROM s GROUP BY user ORDER BY 2 DESC""", (IDLE_UTIL,)).fetchall()
            shared_sec = c.execute("SELECT SUM(sec) FROM s WHERE k > 1").fetchone()[0] or 0
            per_bucket = c.execute(
                "SELECT strftime(?, ts, 'unixepoch', 'localtime'), user, SUM(sec) "
                "FROM s GROUP BY 1, 2", (fmt,)).fetchall()
            tl_users = c.execute(
                "SELECT gpu, (ts - ?) / ?, user, SUM(sec) FROM s GROUP BY 1, 2, 3",
                (since, bin_s)).fetchall()
            tl_gpu = c.execute("""
                SELECT gpu, (ts - ?) / ?, SUM(dt),
                       SUM(util * dt) / SUM(CASE WHEN util IS NOT NULL THEN dt END),
                       AVG(mem_used), MAX(mem_total), AVG(power), MAX(power), AVG(temp),
                       SUM(power * dt)
                FROM gpu WHERE ts >= ? AND ts < ? GROUP BY 1, 2""",
                               (since, bin_s, since, end)).fetchall()
            first_sample = c.execute("SELECT MIN(ts) FROM gpu").fetchone()[0]
            c.execute("DROP TABLE temp.s")
            # recording periods: one row per sample time (from a little before the range)
            samples = c.execute("SELECT ts, MAX(dt) FROM gpu WHERE ts >= ? AND ts < ? "
                                "GROUP BY ts ORDER BY ts", (since - 3600, end)).fetchall()
            events = c.execute("SELECT ts, kind, detail FROM events WHERE ts >= ? AND ts < ? ORDER BY ts",
                               (since - 3600, end)).fetchall()
            # utilization of the theoretical maximum, per GPU, for every range at once
            per = c.execute(
                "SELECT gpu, " + ", ".join(
                    ["SUM(CASE WHEN ts >= ? THEN util * dt END), "
                     "SUM(CASE WHEN ts >= ? AND util IS NOT NULL THEN dt END)"] * len(RANGES)) +
                " FROM gpu WHERE ts >= ? GROUP BY gpu ORDER BY gpu",
                [w for r in RANGES for w in (windows[r], windows[r])] + [min(windows.values())]).fetchall()

        live = {g["index"]: g for g in (self._live[1] or [])}
        gpu_ids = sorted({r[0] for r in tl_gpu} | set(live))

        buckets = [{"t": int(s.timestamp()), "users": {}} for s in starts[:-1]]
        pos = {k: i for i, k in enumerate(keys)}
        for key, user, sec in per_bucket:
            if key in pos:
                buckets[pos[key]]["users"][user] = round(sec / 3600, 4)

        rows = {g: {"index": g, "mem_total": (live.get(g) or {}).get("mem_total"),
                    "power_limit": (live.get(g) or {}).get("power_limit"), "cells": [None] * nbins}
                for g in gpu_ids}
        for g, b, sampled, util, mem, mem_total, power, pmax, temp, energy in tl_gpu:
            if 0 <= b < nbins:
                rows[g]["cells"][b] = {"s": round(sampled), "u": r1(util), "m": r1(mem), "p": r1(power),
                                       "pm": r1(pmax), "t": r1(temp),
                                       "e": None if energy is None else round(energy), "users": {}}
                if mem_total and not rows[g]["mem_total"]:
                    rows[g]["mem_total"] = mem_total
        for g, b, user, sec in tl_users:
            if g in rows and 0 <= b < nbins and rows[g]["cells"][b] is not None:
                rows[g]["cells"][b]["users"][user] = round(sec)

        sampled = sum(r[2] for r in tl_gpu)
        util_den = sum(r[2] for r in tl_gpu if r[3] is not None)
        util_num = sum(r[2] * r[3] for r in tl_gpu if r[3] is not None)
        energy = [r[9] for r in tl_gpu if r[9] is not None]
        jobs = [u[6] for u in users if u[6] is not None]
        return {
            "range": rng, "kind": kind, "since": since, "end": end, "now": int(now),
            "tz_offset": time.localtime(now).tm_gmtoff, "bin": bin_s,
            "bucket_hours": 1 if kind == "hour" else 24, "n_gpus": len(gpu_ids) or 1,
            "interval": self.interval, "idle_util": IDLE_UTIL,
            "kpi": {"used_h": sum(u[1] for u in users) / 3600, "sampled_h": sampled / 3600,
                    "idle_h": sum(u[2] for u in users) / 3600, "shared_h": shared_sec / 3600,
                    "avg_util": util_num / util_den if util_den else None, "users": len(users),
                    "energy_kwh": sum(energy) / 3.6e6 if energy else None,
                    "jobs_kwh": sum(jobs) / 3.6e6 if jobs else None},
            "users": [{"user": u, "hours": h / 3600, "idle_h": i / 3600, "avg_util": a,
                       "peak_mem": m, "last_seen": ls, "kwh": None if e is None else e / 3.6e6,
                       "shared": sh} for u, h, i, a, m, ls, e, sh in users],
            "buckets": buckets,
            "timeline": [rows[g] for g in gpu_ids],
            "uptime": compute_uptime(samples, events, since, min(end, int(now)), first_sample,
                                     self.interval),
            "periods": period_utilization(per, windows, now),
            "slots": self.store.slots(), "first_sample": first_sample,
            "host": self.host, "db": self.store.info(),
        }


def range_window(rng, now):
    """Local-time bucket starts for a range (ending with the end of the current bucket)."""
    _, kind, n = RANGES[rng]
    local_now = dt.datetime.fromtimestamp(now)
    if kind == "hour":
        first = local_now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=n - 1)
        return [first + dt.timedelta(hours=i) for i in range(n + 1)], "%Y-%m-%d %H"
    first = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(days=n - 1)
    return [first + dt.timedelta(days=i) for i in range(n + 1)], "%Y-%m-%d"


def period_utilization(rows, windows, now):
    """Average utilization as a % of the theoretical maximum (100% for the whole period), per GPU
    and for all GPUs, over the time the logger was recording; plus how much of the period that was."""
    names = list(RANGES)
    out = {"ranges": names, "since": windows, "gpus": [], "all": {"util": {}, "logged": {}}}
    totals = {r: [0.0, 0.0, 0] for r in names}
    for row in rows:
        g = {"index": row[0], "util": {}, "logged": {}}
        for i, r in enumerate(names):
            busy, logged = row[1 + 2 * i], row[2 + 2 * i]
            span = max(1.0, now - windows[r])
            g["util"][r] = busy / logged if logged else None
            g["logged"][r] = min(1.0, (logged or 0) / span)
            if logged:
                totals[r][0] += busy
                totals[r][1] += logged
            totals[r][2] += 1
        out["gpus"].append(g)
    for r in names:
        busy, logged, n = totals[r]
        out["all"]["util"][r] = busy / logged if logged else None
        out["all"]["logged"][r] = min(1.0, logged / (n * max(1.0, now - windows[r]))) if n else 0.0
    return out


def compute_uptime(samples, events, since, stop, first_sample, interval):
    """Split [since, stop] into the periods when samples were recorded and the gaps between them.

    samples: [(ts, dt)] in time order. Missing a single sample is tolerated; longer gaps are
    outages. Each outage is matched with events that explain it: failed samples (the logger was
    running but nvidia-smi was not answering) and logger starts (with the node's boot time)."""
    start = since if first_sample is None else max(since, first_sample)
    res = {"start": start, "stop": stop, "up": [], "down": [], "pct": None, "starts": []}
    if first_sample is None or start >= stop:
        return res
    tol = max([d for _, d in samples] or [interval])
    covered = []
    for ts, d in samples:
        if covered and ts <= covered[-1][1] + d:
            covered[-1][1] = max(covered[-1][1], ts + d)
        else:
            covered.append([ts, ts + d])
    covered = [[max(a, start), min(b, stop)] for a, b in covered if b > start and a < stop]
    down, prev = [], start
    for a, b in covered:
        if a - prev > tol:
            down.append([prev, a])
        prev = max(prev, b)
    if stop - prev > tol:
        down.append([prev, stop])
    edges = [start] + [x for gap in down for x in gap] + [stop]
    res["up"] = [[edges[i], edges[i + 1]] for i in range(0, len(edges), 2) if edges[i + 1] > edges[i]]
    res["pct"] = 1 - sum(b - a for a, b in down) / (stop - start)
    res["starts"] = [ts for ts, kind, _ in events if kind == "start" and start <= ts <= stop]
    for a, b in down:
        o = {"a": a, "b": b, "ongoing": b >= stop, "error": None, "errors": 0,
             "restart": None, "reboot": None}
        errs = [e for e in events if e[1] == "error" and a - tol <= e[0] <= b + tol]
        if errs:
            o["error"], o["errors"] = errs[-1][2], len(errs)
        st = [e for e in events if e[1] == "start" and a - tol <= e[0] <= b + 3 * tol]
        if st:
            o["restart"] = st[0][0]
            try:
                boot = json.loads(st[0][2] or "{}").get("boot")
            except (ValueError, AttributeError):
                boot = None
            if boot and a - tol <= boot <= b + tol:
                o["reboot"] = boot
        res["down"].append(o)
    return res


def summary_rows(d):
    total = d["kpi"]["used_h"]
    for u in d["users"]:
        yield [u["user"], round(u["hours"], 3), round(100 * u["hours"] / total, 1) if total else 0,
               None if u["avg_util"] is None else round(u["avg_util"], 1), round(u["idle_h"], 3),
               None if u["shared"] is None else round(100 * u["shared"], 1),
               None if u["kwh"] is None else round(u["kwh"], 3),
               None if u["peak_mem"] is None else round(u["peak_mem"] / 1024, 2),
               dt.datetime.fromtimestamp(u["last_seen"]).isoformat(sep=" ")]


def summary_csv(d):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SUMMARY_HEADER)
    w.writerows(summary_rows(d))
    return buf.getvalue()


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0


def print_report(d):
    k, db = d["kpi"], d["db"]
    print("GPU usage on %s, last %s (%d GPUs)" % (d["host"], d["range"], d["n_gpus"]))
    print("%.1f GPU-hours used of %.1f logged, %.1f held while idle, %.1f shared"
          % (k["used_h"], k["sampled_h"], k["idle_h"], k["shared_h"]))
    if k["energy_kwh"] is not None:
        print("GPU energy %.1f kWh (%.1f kWh during jobs)" % (k["energy_kwh"], k["jobs_kwh"] or 0))
    print()
    print("%-16s %9s %6s %9s %11s %7s %9s %9s" % ("user", "GPU-hours", "share", "avg util",
                                                  "held idle h", "shared", "energy", "peak mem"))
    for u in d["users"]:
        print("%-16s %9.1f %5.0f%% %9s %11.1f %6.0f%% %9s %9s" % (
            u["user"][:16], u["hours"], 100 * u["hours"] / (k["used_h"] or 1),
            "-" if u["avg_util"] is None else "%.0f%%" % u["avg_util"], u["idle_h"],
            100 * (u["shared"] or 0), "-" if u["kwh"] is None else "%.1f kWh" % u["kwh"],
            "-" if u["peak_mem"] is None else "%.1f GB" % (u["peak_mem"] / 1024)))
    if not d["users"]:
        print("(no GPU use recorded in this period)")

    p = d["periods"]
    print("\nutilization as % of the theoretical maximum (over logged time)")
    print("%-10s" % "" + "".join("%9s" % r for r in p["ranges"]))
    for name, g in [("GPU %d" % g["index"], g) for g in p["gpus"]] + [("all GPUs", p["all"])]:
        print("%-10s" % name + "".join("%9s" % ("-" if g["util"][r] is None else "%.0f%%" % g["util"][r])
                                       for r in p["ranges"]))

    u = d["uptime"]
    if u["pct"] is not None:
        longest = max([o["b"] - o["a"] for o in u["down"]] or [0])
        print("\nlogger uptime %.1f%% (%d outages, longest %.1f h)"
              % (100 * u["pct"], len(u["down"]), longest / 3600))
    print("database: %s (%s, %d GPU samples, %d process samples)"
          % (db["path"], fmt_bytes(db["bytes"]), db["gpu_rows"], db["proc_rows"]))


def export_all(store, host, outdir):
    os.makedirs(outdir, exist_ok=True)
    stem = "%s_%s" % (host, time.strftime("%Y%m%d"))
    paths = [os.path.join(outdir, "gpu-usage_%s.sqlite" % stem)]
    store.backup(paths[0])
    for table, (_, _, name) in CSV_EXPORTS.items():
        path = os.path.join(outdir, "%s_%s.csv" % (name, stem))
        with open(path, "wb") as f:
            for chunk in store.iter_csv(table):
                f.write(chunk)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- web

def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _download(self, chunks, ctype, filename, length=None):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % filename)
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)

        def do_GET(self):
            url = urlparse(self.path)
            p, q = url.path, parse_qs(url.query)
            stem = "%s_%s" % (app.host, time.strftime("%Y%m%d"))
            try:
                if p.endswith("/api/live"):
                    self._send(200, json.dumps(app.live()).encode(), "application/json")
                elif p.endswith("/api/history"):
                    d = app.history(q.get("range", ["7d"])[0])
                    self._send(200, json.dumps(d).encode(), "application/json")
                elif p.endswith("/export/usage.sqlite"):
                    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
                    os.close(fd)
                    try:
                        app.store.backup(tmp)
                        with open(tmp, "rb") as f:
                            self._download(iter(lambda: f.read(1 << 20), b""), "application/vnd.sqlite3",
                                           "gpu-usage_%s.sqlite" % stem, os.path.getsize(tmp))
                    finally:
                        os.unlink(tmp)
                elif any(p.endswith("/export/%s.csv" % t) for t in CSV_EXPORTS):
                    table = p.rsplit("/", 1)[1][:-4]
                    self._download(app.store.iter_csv(table), "text/csv; charset=utf-8",
                                   "%s_%s.csv" % (CSV_EXPORTS[table][2], stem))
                elif p.endswith("/export/summary.csv"):
                    d = app.history(q.get("range", ["7d"])[0])
                    body = summary_csv(d).encode()
                    self._download([body], "text/csv; charset=utf-8",
                                   "user-summary-%s_%s.csv" % (d["range"], stem), len(body))
                elif p.endswith("/") or p.endswith("/index.html"):
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser went away (e.g. a cancelled download)
            except Exception as e:
                log("request %s failed: %s" % (p, e))
                try:
                    self._send(500, str(e).encode(), "text/plain")
                except Exception:
                    pass

        def log_message(self, *a):
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-user GPU usage logger and dashboard.")
    ap.add_argument("--data-dir", default=os.environ.get("GPU_DASH_DIR") or os.path.expanduser("~/.gpu-dash"),
                    help="where the database lives (default: $GPU_DASH_DIR, else ~/.gpu-dash); "
                         "created if missing")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (default 127.0.0.1; 0.0.0.0 exposes it, without login, "
                         "to everyone who can reach this machine)")
    ap.add_argument("--interval", type=int, default=60, help="seconds between samples (default 60)")
    ap.add_argument("--report", metavar="RANGE", choices=list(RANGES),
                    help="print a per-user summary for 24h, 7d, 30d or 90d and exit")
    ap.add_argument("--csv", action="store_true", help="with --report: print CSV instead of a table")
    ap.add_argument("--export", metavar="DIR",
                    help="write a copy of the database plus all samples as CSV into DIR and exit")
    a = ap.parse_args(argv)

    data_dir = os.path.abspath(os.path.expanduser(a.data_dir))
    os.makedirs(data_dir, exist_ok=True)
    if a.report or a.export:
        store = Store(os.path.join(data_dir, "usage.db"))
        try:
            app = App(store, a.interval)
            if a.report:
                d = app.history(a.report)
                if a.csv:
                    sys.stdout.write(summary_csv(d))
                else:
                    print_report(d)
            else:
                for path in export_all(store, app.host, a.export):
                    print("%-60s %s" % (path, fmt_bytes(os.path.getsize(path))))
            sys.stdout.flush()
        except BrokenPipeError:  # e.g. `--report 7d | head`
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        finally:
            store.con.close()
        return

    lock = open(os.path.join(data_dir, "gpu_dash.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        print("gpu_dash is already running for %s" % data_dir)
        return
    store = Store(os.path.join(data_dir, "usage.db"))
    app = App(store, a.interval)
    try:
        server = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    except OSError as e:
        log("cannot listen on %s:%d (%s); try another --port" % (a.host, a.port, e))
        sys.exit(1)
    server.daemon_threads = True
    # remembered so the uptime chart can tell restarts and reboots apart from other gaps
    store.event(int(time.time()), "start",
                json.dumps({"pid": os.getpid(), "boot": BOOT_TIME, "interval": a.interval}))
    threading.Thread(target=app.sample_forever, daemon=True).start()
    log("sampling every %ds into %s; dashboard at http://%s:%d/"
        % (a.interval, store.path, a.host, a.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU usage</title>
<style>
:root {
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10); --wash:rgba(11,11,11,.05);
  --track:#efeee9; --accent:#2a78d6; --accent-track:#cde2fb; --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#6250d6; --s8:#e34948; --other:#a9a8a1;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --wash:rgba(255,255,255,.06);
    --track:#262624; --accent:#3987e5; --accent-track:#0d366b;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767; --other:#62615c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --wash:rgba(255,255,255,.06);
  --track:#262624; --accent:#3987e5; --accent-track:#0d366b;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767; --other:#62615c;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
       font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1200px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:20px; font-weight:600; margin:0; }
h1 .host { color:var(--ink2); font-weight:400; }
h2 { font-size:15px; font-weight:600; margin:0; }
h3 { font-size:14px; font-weight:600; margin:0; }
h4 { font-size:13px; font-weight:600; margin:0; }
.sub { color:var(--ink2); font-size:13px; }
.small { font-size:12px; }
code { font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
header, .sec-h { display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:8px 16px; }
.sec-h { align-items:center; margin:28px 0 12px; }
.sec-h h2 span { color:var(--ink2); font-weight:400; margin-left:6px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; min-width:0; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }
.stack { display:grid; gap:12px; }
#err { display:none; margin-top:16px; border-color:var(--warning); }
#err.on { display:block; }

/* live GPU cards */
.gh { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
.gh .sub { font-size:12px; margin-top:1px; }
.chip { display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:500; white-space:nowrap;
        padding:2px 10px; border-radius:999px; border:1px solid var(--border); }
.chip .ic { font-size:10px; line-height:1; }
.chip.good .ic { color:var(--good); } .chip.warn .ic { color:var(--warning); } .chip.busy .ic { color:var(--accent); }
.meter { margin-top:12px; }
.ml { display:flex; justify-content:space-between; gap:8px; font-size:12px; color:var(--ink2); }
.ml b { color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums; }
.mtrack { height:8px; border-radius:4px; background:var(--accent-track); margin-top:4px; overflow:hidden; }
.mfill { height:100%; background:var(--accent); border-radius:4px; transition:width .4s; }
.gx { margin-top:8px; font-size:12px; color:var(--muted); }
.procs { margin-top:12px; border-top:1px solid var(--grid); padding-top:10px; display:grid;
         grid-template-columns:minmax(0,1fr); gap:10px; }
.procs > div { min-width:0; }
.pl { display:flex; align-items:center; gap:8px; font-size:13px; }
.pu { font-weight:600; } .pm { color:var(--ink2); font-variant-numeric:tabular-nums; white-space:nowrap; }
.pa { color:var(--muted); margin-left:auto; white-space:nowrap; font-size:12px; }
.tag { font-size:11px; color:var(--ink2); white-space:nowrap; }
.tag .ic { color:var(--warning); font-size:9px; margin-right:3px; }
.sw { width:10px; height:10px; border-radius:3px; flex:none; display:inline-block; }
.cmd { font:11.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--muted);
       white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding-left:18px; margin-top:1px; }

/* range filter */
.seg { display:inline-flex; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:2px; }
.seg button { font:inherit; font-size:13px; border:0; background:transparent; color:var(--ink2);
              padding:5px 12px; border-radius:6px; cursor:pointer; }
.seg button:hover { background:var(--wash); }
.seg button[aria-pressed="true"] { background:var(--ink); color:var(--surface); font-weight:600; }

/* stat tiles */
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }
.tile .lab { color:var(--ink2); font-size:13px; }
.tile .val { font-size:28px; font-weight:600; margin:2px 0 0; letter-spacing:-.01em; }
.tile .val small { font-size:15px; font-weight:500; color:var(--ink2); margin-left:3px; }
.tile .sub { font-size:12px; }

/* charts */
.ch { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:baseline; gap:6px 16px; margin-bottom:10px; }
.legend { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12px; color:var(--ink2); }
.legend .li { display:inline-flex; align-items:center; gap:6px; }
.legend .sw { border-radius:2px; }
svg { display:block; }
svg text { font-family:inherit; }
.grid { stroke:var(--grid); stroke-width:1; shape-rendering:crispEdges; }
.axis { stroke:var(--axis); stroke-width:1; shape-rendering:crispEdges; }
.tick { fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
.rl { fill:var(--ink2); font-size:12px; }
.bar { pointer-events:none; }
.hit { fill:transparent; outline:none; cursor:default; }
.hit:hover, .hit:focus-visible { fill:var(--wash); }
.trk { fill:var(--track); }
.hov { fill:none; stroke:var(--ink); stroke-width:1.5; pointer-events:none; }
.line { fill:none; stroke:var(--accent); stroke-width:2; stroke-linejoin:round; stroke-linecap:round; pointer-events:none; }
.area { fill:var(--accent); opacity:.1; pointer-events:none; }
.pt { fill:var(--accent); pointer-events:none; }
.cross { stroke:var(--ink2); stroke-width:1; pointer-events:none; }
.cdot { fill:var(--accent); stroke:var(--surface); stroke-width:2; pointer-events:none; }
#metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:8px 28px; }
.mh { display:flex; justify-content:space-between; gap:8px; font-size:12px; color:var(--ink2); margin:12px 0 2px; }
.mh .dim { font-variant-numeric:tabular-nums; }
.empty { color:var(--muted); font-size:13px; padding:24px 0; text-align:center; }
.hist { transition:opacity .2s; }
body.loading .hist { opacity:.6; }

/* table */
.tablewrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; font-weight:500; color:var(--ink2); font-size:12px; padding:6px 10px; border-bottom:1px solid var(--axis); white-space:nowrap; }
td { padding:8px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th:first-child, td:first-child { padding-left:0; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
td .sw { margin-right:8px; vertical-align:-1px; }
.sbar { display:inline-block; width:64px; height:6px; border-radius:3px; background:var(--track);
        vertical-align:middle; margin-right:8px; overflow:hidden; }
.sbar span { display:block; height:100%; border-radius:3px; }
.dim { color:var(--muted); }
.note { color:var(--muted); font-size:12px; margin-top:10px; }
td.wrap { white-space:normal; min-width:240px; color:var(--ink2); }

/* utilization of theoretical maximum */
.ptable td.num { min-width:110px; }
.ptable .pv { font-weight:600; font-size:15px; }
.ptable .pbar { display:block; height:4px; border-radius:2px; background:var(--accent-track); margin-top:5px; overflow:hidden; }
.ptable .pbar span { display:block; height:100%; background:var(--accent); border-radius:2px; }
.ptable .pl2 { display:block; font-size:11px; color:var(--muted); margin-top:3px; }
.ptable tr.all td { font-weight:600; border-bottom:0; }
.ptable tr.all td:first-child { color:var(--ink); }

/* logger uptime */
.upseg { fill:var(--good); }
.downseg { fill:var(--critical); }
.startmk { fill:var(--ink2); }
.sw.k-up { background:var(--good); } .sw.k-down { background:var(--critical); } .sw.k-pre { background:var(--track); }
.legend .mk { color:var(--ink2); font-size:9px; }
#uptime-stat b { color:var(--ink); }
#outages { margin-top:8px; }

/* data section */
.btn { display:inline-flex; align-items:center; gap:6px; padding:6px 12px; border:1px solid var(--border);
       border-radius:8px; background:var(--surface); color:var(--ink); text-decoration:none; font-size:13px; white-space:nowrap; }
.btn:hover { background:var(--wash); }
.btn .ar { color:var(--ink2); }
.dstats { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:16px; margin-bottom:16px; }
.dstat .lab { color:var(--ink2); font-size:12px; }
.dstat .dval { font-size:18px; font-weight:600; margin-top:1px; }
.dl { display:flex; flex-wrap:wrap; gap:8px; }
footer { color:var(--muted); font-size:12px; margin-top:24px; }

/* tooltip */
#tip { position:fixed; z-index:10; pointer-events:none; background:var(--surface); color:var(--ink);
       border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-size:12px;
       box-shadow:0 4px 16px rgba(0,0,0,.14); min-width:150px; max-width:300px; }
#tip .tt { color:var(--ink2); margin-bottom:4px; }
#tip .tr { display:flex; align-items:center; gap:8px; padding:1px 0; }
#tip .key { width:12px; height:3px; border-radius:2px; flex:none; }
#tip .tr b { font-variant-numeric:tabular-nums; min-width:50px; }
#tip .tr span:last-child { color:var(--ink2); overflow:hidden; text-overflow:ellipsis; }
#tip .tf { color:var(--muted); }
#tip .tr + .tf { margin-top:5px; border-top:1px solid var(--grid); padding-top:5px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>GPU usage <span class="host" id="host"></span></h1>
    <div class="sub" id="updated"></div>
  </header>
  <div class="card" id="err"></div>

  <div class="sec-h"><h2>Right now<span id="freecount"></span></h2></div>
  <div class="cards" id="live"></div>

  <div class="sec-h"><h2>Utilization of theoretical maximum</h2></div>
  <div class="card hist">
    <div class="tablewrap"><table id="periods" class="ptable"></table></div>
    <div class="note" id="pnote"></div>
  </div>

  <div class="sec-h">
    <h2>History</h2>
    <div class="seg" role="group" aria-label="Time range">
      <button data-r="24h">24 hours</button><button data-r="7d">7 days</button><button data-r="30d">30 days</button><button data-r="90d">90 days</button>
    </div>
  </div>
  <div class="stack hist">
    <div class="tiles" id="kpis"></div>
    <div class="card">
      <div class="ch"><h3 id="cols-title">GPU-hours by user</h3><div class="legend"></div></div>
      <div id="cols"></div>
    </div>
    <div class="card">
      <div class="ch"><div><h3>Who held each GPU</h3><div class="sub small">Bar height is how much of the time the GPU was in use; colours show who held it, split by GPU memory when shared</div></div><div class="legend"></div></div>
      <div id="timeline"></div>
    </div>
    <div class="card">
      <div class="ch"><div><h3>Utilization, memory and power per GPU</h3><div class="sub small">Average per time slot; hover to compare the GPUs at the same moment</div></div></div>
      <div id="metrics"></div>
    </div>
    <div class="card">
      <div class="ch"><h3>Per-user summary</h3><a class="btn" id="sumcsv" href="export/summary.csv" download><span class="ar">↓</span>CSV</a></div>
      <div class="tablewrap"><table id="usertable"></table></div>
      <div class="note" id="tnote"></div>
    </div>
    <div class="card">
      <div class="ch"><div><h3>Logger uptime</h3><div class="sub small" id="uptime-stat"></div></div><div class="legend" id="uptime-legend"></div></div>
      <div id="uptime"></div>
      <div class="tablewrap"><table id="outages"></table></div>
      <div class="note" id="outnote"></div>
    </div>
  </div>

  <div class="sec-h"><h2>Data</h2></div>
  <div class="card">
    <div class="dstats" id="dbstats"></div>
    <div class="dl">
      <a class="btn" href="export/usage.sqlite" download><span class="ar">↓</span>Database (.sqlite)</a>
      <a class="btn" href="export/gpu.csv" download><span class="ar">↓</span>GPU samples (.csv)</a>
      <a class="btn" href="export/proc.csv" download><span class="ar">↓</span>Process samples (.csv)</a>
      <a class="btn" href="export/events.csv" download><span class="ar">↓</span>Logger events (.csv)</a>
    </div>
    <div class="note" id="dbnote"></div>
  </div>
  <footer id="foot"></footer>
</main>
<div id="tip" role="tooltip" hidden></div>
<script>
(() => {
"use strict";
const RANGES = ["24h", "7d", "30d", "90d"];
const OTHER = "\u0000other";
let range = RANGES.includes(location.hash.slice(1)) ? location.hash.slice(1) : "7d";
let live = null, hist = null, slots = {}, TZ = 0, mcharts = [];

const $ = s => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, cls, ...kids) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  for (const k of kids) if (k != null && k !== "") e.append(k);
  return e;
}
function sv(tag, attrs, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}

// ---- identity colour: fixed per user (first-seen order), never by rank
const keyOf = u => (slots[u] != null && slots[u] < 8) ? u : OTHER;
const colorOf = k => (k !== OTHER && slots[k] != null && slots[k] < 8) ? `var(--s${slots[k] + 1})` : "var(--other)";
const nameOf = k => k === OTHER ? "Other users" : k;
function swatch(k) { const s = el("span", "sw"); s.style.background = colorOf(k); return s; }
const slotOrder = (a, b) => (a === OTHER) - (b === OTHER) || (slots[a] ?? 1e9) - (slots[b] ?? 1e9);

// ---- formatting
const nf1 = new Intl.NumberFormat(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
function fmtH(h) {
  if (h == null) return "–";
  if (h < 1) return Math.round(h * 60) + " min";
  if (h < 100) return nf1.format(h) + " h";
  return Math.round(h).toLocaleString() + " h";
}
function kwhParts(k) {
  if (k == null) return ["–", null];
  if (k < 1) return [String(Math.round(k * 1000)), "Wh"];
  return [k < 100 ? nf1.format(k) : Math.round(k).toLocaleString(), "kWh"];
}
const fmtKWh = k => { const [v, u] = kwhParts(k); return u ? v + " " + u : v; };
const fmtPct = x => x == null ? "–" : Math.round(x) + "%";
const fmtGB = mib => mib == null ? "–" : (mib / 1024).toFixed(1) + " GB";
function fmtBytes(b) {
  if (b == null) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return (i ? b.toFixed(b < 10 ? 2 : 1) : b) + " " + u[i];
}
function fmtAge(s) {
  if (s == null) return "";
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return d ? `${d} d ${h} h` : h ? `${h} h ${m} min` : m ? `${m} min` : "<1 min";
}
// all times are shown in the node's local time
const fmtT = (t, o) => new Date((t + TZ) * 1000).toLocaleString(undefined, Object.assign({timeZone: "UTC"}, o));
const HM = {hour: "2-digit", minute: "2-digit", hourCycle: "h23"};
const DAY = {weekday: "short", day: "numeric"};
const DATE = {day: "numeric", month: "short"};
const LONGDAY = {weekday: "short", day: "numeric", month: "short"};
const YEAR = {day: "numeric", month: "short", year: "numeric"};
function ago(t, now, interval) {
  const s = now - t;
  if (s <= 3 * interval) return "now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 2 * 86400) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " d ago";
}
function nice(max) {
  if (!(max > 0)) max = 1;
  const raw = max / 4, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw - 1e-9);
  return {step, top: Math.ceil(max / step - 1e-9) * step};
}
const fmtTick = v => Number.isInteger(v) ? v.toLocaleString() : v.toFixed(1);
function topRounded(x, y, w, h, r) {   // rounded data-end on top, square at the baseline
  r = Math.max(0, Math.min(r, w / 2, h));
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
const rectPath = (x, y, w, h) => `M${x},${y}h${w}v${h}h${-w}Z`;
const sum = o => Object.values(o).reduce((a, b) => a + b, 0);
function aggregate(users) {   // per-user values -> per-legend-key values (tail folds into Other)
  const v = {};
  for (const u in users) { const k = keyOf(u); v[k] = (v[k] || 0) + users[u]; }
  return v;
}

// ---- tooltip: rows are {value, label, color?}; values lead, labels follow
const tip = $("#tip");
const userRow = (k, value) => ({value, label: nameOf(k), color: colorOf(k)});
function tipShow(x, y, title, rows, foot) {
  tip.textContent = "";
  tip.append(el("div", "tt", title));
  for (const r of rows) {
    const k = el("span", "key");
    if (r.color) k.style.background = r.color; else k.style.visibility = "hidden";
    tip.append(el("div", "tr", k, el("b", null, r.value), el("span", null, r.label)));
  }
  if (foot) tip.append(el("div", "tf", foot));
  tip.hidden = false;
  const b = tip.getBoundingClientRect();
  let L = x + 14, T = y + 14;
  if (L + b.width > innerWidth - 8) L = x - b.width - 14;
  if (T + b.height > innerHeight - 8) T = y - b.height - 14;
  tip.style.left = Math.max(8, L) + "px";
  tip.style.top = Math.max(8, T) + "px";
}
const tipHide = () => { tip.hidden = true; };

// ---- live cards
function meter(label, v, max, text) {
  const pct = (v == null || !max) ? 0 : Math.max(0, Math.min(100, v / max * 100));
  const fill = el("div", "mfill"); fill.style.width = pct + "%";
  const track = el("div", "mtrack", fill);
  track.setAttribute("role", "meter"); track.setAttribute("aria-label", label);
  track.setAttribute("aria-valuenow", Math.round(pct)); track.setAttribute("aria-valuemin", 0); track.setAttribute("aria-valuemax", 100);
  return el("div", "meter", el("div", "ml", el("span", null, label), el("b", null, text)), track);
}
function renderLive() {
  TZ = live.tz_offset;
  $("#host").textContent = "· " + live.host;
  document.title = "GPU usage · " + live.host;
  const err = $("#err");
  err.textContent = live.error ? "Could not read the GPUs: " + live.error : "";
  err.classList.toggle("on", !!live.error);
  const box = $("#live"); box.textContent = "";
  const free = live.gpus.filter(g => !g.procs.length).length;
  const shared = live.gpus.filter(g => new Set(g.procs.map(p => p.user)).size > 1).length;
  $("#freecount").textContent = live.gpus.length ? `${free} of ${live.gpus.length} free` + (shared ? ` · ${shared} shared` : "") : "";
  for (const g of live.gpus) {
    const nUsers = new Set(g.procs.map(p => p.user)).size;
    const idle = g.util != null && g.util < live.idle_util;
    const sharedTxt = nUsers > 1 ? ` · shared by ${nUsers}` : "";
    const st = !g.procs.length ? ["good", "●", "Free"] : idle ? ["warn", "▲", "Held, idle" + sharedTxt] : ["busy", "●", "In use" + sharedTxt];
    const head = el("div", "gh",
      el("div", null, el("h3", null, "GPU " + g.index), el("div", "sub", g.name)),
      el("span", "chip " + st[0], el("span", "ic", st[1]), st[2]));
    const memText = g.mem_used == null ? "–" : `${(g.mem_used / 1024).toFixed(1)} / ${Math.round(g.mem_total / 1024)} GB`;
    const memFree = g.mem_used == null || g.mem_total == null ? "" : ` · ${((g.mem_total - g.mem_used) / 1024).toFixed(1)} GB free`;
    const power = g.power == null ? null : Math.round(g.power) + (g.power_limit ? " / " + Math.round(g.power_limit) : "") + " W";
    const extra = [g.temp != null ? Math.round(g.temp) + " °C" : null, power].filter(Boolean).join(" · ");
    const list = el("div", "procs");
    if (!g.procs.length) list.append(el("div", "sub small", "No processes, available"));
    for (const p of g.procs) {
      const cmd = el("div", "cmd", p.cmd || "pid " + p.pid);
      cmd.title = (p.cmd || "") + "  (pid " + p.pid + ")";
      const smTxt = p.sm == null ? null : el("span", "pm", "SM " + Math.round(p.sm) + "%");
      const idleTag = p.sm != null && p.sm < live.idle_util ? el("span", "tag", el("span", "ic", "▲"), "idle") : null;
      list.append(el("div", null,
        el("div", "pl", swatch(keyOf(p.user)), el("span", "pu", p.user), el("span", "pm", fmtGB(p.mem)), smTxt, idleTag,
           el("span", "pa", p.age != null ? "running " + fmtAge(p.age) : "")),
        cmd));
    }
    box.append(el("div", "card", head,
      meter("Utilization", g.util, 100, fmtPct(g.util)),
      meter("Memory" + memFree, g.mem_used, g.mem_total, memText),
      extra ? el("div", "gx", extra) : null, list));
  }
  $("#updated").textContent = "Updated " + fmtT(live.now, Object.assign({second: "2-digit"}, HM));
}

// ---- history
function seriesKeys(d) {   // users present in range, in stable slot order; tail folds into "Other"
  return [...new Set(d.users.map(u => keyOf(u.user)))].sort(slotOrder);
}
function renderLegends(d) {
  for (const box of document.querySelectorAll(".legend:not(#uptime-legend)")) {
    box.textContent = "";
    for (const k of seriesKeys(d)) box.append(el("span", "li", swatch(k), nameOf(k)));
  }
}
function renderKpis(d) {
  const k = d.kpi, box = $("#kpis"); box.textContent = "";
  const capH = d.n_gpus * (d.end - d.since) / 3600;
  const occ = k.sampled_h ? k.used_h / k.sampled_h * 100 : null;
  const top = d.users[0];
  const num = h => h < 100 ? nf1.format(h) : Math.round(h).toLocaleString();
  const tile = (lab, val, unit, sub) => el("div", "card tile", el("div", "lab", lab),
    el("div", "val", val, unit ? el("small", null, unit) : null), el("div", "sub", sub));
  const logged = k.sampled_h < capH * 0.9 ? ` (${num(k.sampled_h)} h logged)` : "";
  const sharedTxt = k.used_h && k.shared_h ? ` · ${Math.round(k.shared_h / k.used_h * 100)}% shared` : "";
  const [ev, eu] = kwhParts(k.energy_kwh);
  box.append(
    tile("GPU-hours used", num(k.used_h), "h",
         occ == null ? "No samples in this period" : `GPUs busy ${Math.round(occ)}% of the time${logged}${sharedTxt}`),
    tile("Average utilization", fmtPct(k.avg_util), null, "Across all GPUs, including idle time"),
    tile("Held while idle", num(k.idle_h), "h",
         k.used_h ? `${Math.round(k.idle_h / k.used_h * 100)}% of used time at <${d.idle_util}% utilization` : `Held at <${d.idle_util}% utilization`),
    tile("GPU energy", ev, eu,
         k.energy_kwh == null ? "No power readings in this period" :
         `${Math.round((k.jobs_kwh || 0) / k.energy_kwh * 100)}% of it while running jobs`),
    tile("Active users", String(k.users), null,
         top ? `Top: ${top.user} (${Math.round(top.hours / k.used_h * 100)}%)` : "Nobody used a GPU"));
}

function renderCols(d) {
  const box = $("#cols"); box.textContent = "";
  $("#cols-title").textContent = "GPU-hours by user, per " + (d.kind === "hour" ? "hour" : "day");
  const W = Math.max(280, box.clientWidth), H = 250, L = 40, R = 6, T = 16, B = 26;
  const pw = W - L - R, ph = H - T - B, base = T + ph;
  const cap = d.n_gpus * d.bucket_hours;
  const {step, top} = nice(cap);
  const y = v => base - v / top * ph;
  const svg = sv("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
                         "aria-label": "Stacked columns of GPU-hours per user; the table below lists the totals"});
  for (let v = step; v <= top + 1e-9; v += step) {
    svg.append(sv("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid"}));
    svg.append(sv("text", {x: L - 8, y: y(v) + 4, class: "tick", "text-anchor": "end"}, fmtTick(v)));
  }
  svg.append(sv("text", {x: L - 8, y: base + 4, class: "tick", "text-anchor": "end"}, "0"));
  svg.append(sv("line", {x1: L, x2: W - R, y1: y(cap), y2: y(cap), class: "axis"}));
  svg.append(sv("text", {x: W - R, y: y(cap) - 5, class: "tick", "text-anchor": "end"},
                `capacity: ${d.n_gpus} GPUs × ${d.bucket_hours} h`));

  const keys = seriesKeys(d), n = d.buckets.length, band = pw / n;
  const bw = Math.max(2, Math.min(24, band * 0.62));
  const label = t => d.kind === "hour" ? fmtT(t, LONGDAY) + ", " + fmtT(t, HM) + "–" + fmtT(t + 3600, HM) : fmtT(t, LONGDAY);
  d.buckets.forEach((b, i) => {
    const vals = aggregate(b.users);
    const segs = keys.filter(k => vals[k] > 0);
    const total = segs.reduce((a, k) => a + vals[k], 0);
    const hit = sv("rect", {x: L + i * band, y: T, width: band, height: ph, class: "hit", tabindex: 0,
                            "aria-label": `${label(b.t)}: ${fmtH(total)}`});
    const show = (cx, cy) => tipShow(cx, cy, label(b.t),
      segs.slice().reverse().map(k => userRow(k, fmtH(vals[k]))),
      total ? `Total ${fmtH(total)} of ${cap} GPU-hours` : "No GPU use");
    hit.addEventListener("pointermove", e => show(e.clientX, e.clientY));
    hit.addEventListener("pointerleave", tipHide);
    hit.addEventListener("focus", () => { const r = hit.getBoundingClientRect(); show(r.left + r.width / 2, r.top + 24); });
    hit.addEventListener("blur", tipHide);
    svg.append(hit);
    const x = L + i * band + (band - bw) / 2;
    let cum = 0;
    segs.forEach((k, j) => {
      const y0 = y(cum), y1 = y(cum + vals[k]); cum += vals[k];
      const yb = y0 - (j > 0 ? 2 : 0), h = yb - y1;          // 2px surface gap between segments
      if (h < 0.75) return;
      const p = sv("path", {class: "bar", d: j === segs.length - 1 ? topRounded(x, y1, bw, h, 4) : rectPath(x, y1, bw, h)});
      p.style.fill = colorOf(k);
      svg.append(p);
    });
  });
  svg.append(sv("line", {x1: L, x2: W - R, y1: base, y2: base, class: "axis"}));
  const need = d.kind === "hour" ? 46 : (n <= 14 ? 50 : 44);
  const strides = d.kind === "hour" ? [1, 2, 3, 4, 6, 12] : [1, 2, 7, 14, 30];
  const stride = strides.find(s => s * band >= need) || strides[strides.length - 1];
  const tickFmt = d.kind === "hour" ? HM : (n <= 14 ? DAY : DATE);
  d.buckets.forEach((b, i) => {
    if ((n - 1 - i) % stride) return;
    const cx = L + (i + 0.5) * band, nearEnd = cx > W - R - 22;   // keep the last label inside the chart
    svg.append(sv("text", {x: nearEnd ? W - R : cx, y: base + 17, class: "tick", "text-anchor": nearEnd ? "end" : "middle"}, fmtT(b.t, tickFmt)));
  });
  box.append(svg);
}

// tick positions along a time axis pw pixels wide, at column-bucket boundaries
function timeTicks(d, pw, minPx) {
  const n = d.buckets.length, span = d.end - d.since;
  const strides = d.kind === "hour" ? [1, 2, 3, 4, 6, 12] : [1, 2, 7, 14, 30];
  const stride = strides.find(s => s * pw / n >= minPx) || strides[strides.length - 1];
  const fmt = d.kind === "hour" ? HM : (n <= 14 ? DAY : DATE);
  const out = [];
  d.buckets.forEach((b, i) => {
    if ((n - 1 - i) % stride) return;
    const x = (b.t - d.since) / span * pw;
    if (x <= pw - 40) out.push({x, label: fmtT(b.t, fmt)});
  });
  return out;
}
function drawTimeTicks(svg, d, L, pw, y) {
  for (const t of timeTicks(d, pw, 56)) {
    svg.append(sv("line", {x1: L + t.x, x2: L + t.x, y1: y, y2: y + 4, class: "axis"}));
    svg.append(sv("text", {x: L + t.x + 3, y: y + 15, class: "tick"}, t.label));
  }
}
const slotTitle = (d, gpu, i) => {
  const t0 = d.since + i * d.bin;
  return `GPU ${gpu} · ${fmtT(t0, LONGDAY)}, ${fmtT(t0, HM)}–${fmtT(t0 + d.bin, HM)}`;
};

function renderTimeline(d) {
  const box = $("#timeline"); box.textContent = "";
  const rows = d.timeline;
  if (!rows.length) { box.append(el("div", "empty", "No samples yet.")); return; }
  const W = Math.max(280, box.clientWidth), L = 52, R = 6, rowH = 28, gapY = 10, B = 24;
  const H = rows.length * (rowH + gapY) - gapY + B;
  const pw = W - L - R, nb = rows[0].cells.length, bw = pw / nb;
  const gap = bw >= 6 ? 2 : bw >= 3 ? 1 : 0, w = Math.max(0.6, bw - gap);
  const svg = sv("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
                         "aria-label": "Timeline of GPU occupancy coloured by user"});
  rows.forEach((row, ri) => {
    const y0 = ri * (rowH + gapY);
    svg.append(sv("text", {x: 0, y: y0 + rowH / 2 + 4, class: "rl"}, "GPU " + row.index));
    row.cells.forEach((c, i) => {
      if (!c) return;
      const x = L + i * bw;
      svg.append(sv("rect", {x, y: y0, width: w, height: rowH, rx: Math.min(2, w / 2), class: "trk"}));
      const busy = sum(c.users);
      if (!busy) return;
      const barH = Math.max(2, Math.min(1, busy / c.s) * rowH);
      const vals = aggregate(c.users), keys = Object.keys(vals).sort(slotOrder);
      let yb = y0 + rowH;                                    // stack users bottom-up, slot order
      keys.forEach((k, j) => {
        const h = barH * vals[k] / busy - (j > 0 ? 2 : 0);
        if (h < 0.75) { yb -= barH * vals[k] / busy; return; }
        if (j > 0) yb -= 2;                                  // 2px surface gap between users
        const p = sv("path", {class: "bar", d: j === keys.length - 1 ? topRounded(x, yb - h, w, h, 4) : rectPath(x, yb - h, w, h)});
        p.style.fill = colorOf(k);
        svg.append(p);
        yb -= h;
      });
    });
  });
  const axisY = rows.length * (rowH + gapY) - gapY;
  drawTimeTicks(svg, d, L, pw, axisY + 2);
  const hov = sv("rect", {class: "hov", width: w + 2, height: rowH + 2, rx: 2, visibility: "hidden"});
  svg.append(hov);
  svg.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    const ri = Math.floor(py / (rowH + gapY)), i = Math.floor((px - L) / bw);
    if (px < L || ri < 0 || ri >= rows.length || py - ri * (rowH + gapY) > rowH || i < 0 || i >= nb) {
      hov.setAttribute("visibility", "hidden"); tipHide(); return;
    }
    hov.setAttribute("x", L + i * bw - 1); hov.setAttribute("y", ri * (rowH + gapY) - 1);
    hov.setAttribute("visibility", "visible");
    const c = rows[ri].cells[i], title = slotTitle(d, rows[ri].index, i);
    if (!c) { tipShow(e.clientX, e.clientY, title, [], d.since + i * d.bin > d.now ? "Not yet" : "No data (logger not running)"); return; }
    const users = c.users, busy = sum(users), names = Object.keys(users).sort((a, b) => users[b] - users[a]);
    const foot = busy ? `In use ${Math.round(Math.min(1, busy / c.s) * 100)}% of the time` +
      (names.length > 1 ? " · shared" : "") + ` · avg utilization ${fmtPct(c.u)}` : "Free the whole time";
    tipShow(e.clientX, e.clientY, title, names.map(u => userRow(u, fmtH(users[u] / 3600))), foot);
  });
  svg.addEventListener("pointerleave", () => { hov.setAttribute("visibility", "hidden"); tipHide(); });
  box.append(svg);
}

// small multiples: one column per GPU, one row per metric, shared y-scale per metric
function renderMetrics(d) {
  const box = $("#metrics"); box.textContent = ""; mcharts = [];
  const rows = d.timeline;
  if (!rows.length) { box.append(el("div", "empty", "No samples yet.")); return; }
  const cells = rows.flatMap(r => r.cells.filter(Boolean));
  const memTop = Math.max(1, ...rows.map(r => (r.mem_total || 0) / 1024), ...cells.map(c => (c.m || 0) / 1024));
  const pTop = Math.max(0, ...rows.map(r => r.power_limit || 0), ...cells.map(c => c.pm || 0));
  const metrics = [
    {label: "Utilization (%)", top: 100, v: c => c.u,
     sum: r => { const [a, s] = r.cells.reduce(([a, s], c) => c && c.u != null ? [a + c.u * c.s, s + c.s] : [a, s], [0, 0]); return s ? `avg ${Math.round(a / s)}%` : ""; }},
    {label: "Memory (GB)", top: Math.round(memTop), v: c => c.m == null ? null : c.m / 1024,
     sum: r => { const m = Math.max(0, ...r.cells.map(c => c && c.m != null ? c.m : 0)); return m ? `peak ${fmtGB(m)}` : ""; }},
  ];
  if (pTop > 0) metrics.push({label: "Power (W)", top: nice(pTop).top, v: c => c.p,
     sum: r => { const e = r.cells.reduce((a, c) => a + (c && c.e != null ? c.e : 0), 0); return e ? `${fmtKWh(e / 3.6e6)} used` : ""; }});
  for (const row of rows) {
    const col = el("div", null, el("h4", null, "GPU " + row.index));
    for (const m of metrics) {
      const wrap = el("div", null, el("div", "mh", el("span", null, m.label), el("span", "dim", m.sum(row))));
      col.append(wrap);
      mcharts.push({row, m, wrap});
    }
    box.append(col);
  }
  for (const mc of mcharts) drawMini(d, mc);
}
function drawMini(d, mc) {
  const {row, m, wrap} = mc;
  const W = Math.max(200, wrap.clientWidth), H = 104, L = 30, R = 4, T = 6, B = 20;
  const pw = W - L - R, ph = H - T - B, base = T + ph, nb = row.cells.length, bw = pw / nb;
  const y = v => base - Math.max(0, Math.min(v, m.top)) / m.top * ph;
  const x = i => L + (i + 0.5) * bw;
  const svg = sv("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
                         "aria-label": `${m.label} over time, GPU ${row.index}`});
  for (const v of [m.top / 2, m.top]) {
    svg.append(sv("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid"}));
    svg.append(sv("text", {x: L - 6, y: y(v) + 4, class: "tick", "text-anchor": "end"}, fmtTick(v)));
  }
  svg.append(sv("text", {x: L - 6, y: base + 4, class: "tick", "text-anchor": "end"}, "0"));
  let line = "", area = "", seg = [];
  const flush = () => {
    if (seg.length === 1) svg.append(sv("circle", {cx: seg[0][0], cy: seg[0][1], r: 1.5, class: "pt"}));
    else if (seg.length > 1) {
      const pts = seg.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L");
      line += "M" + pts;
      area += `M${seg[0][0].toFixed(1)},${base}L${pts}L${seg[seg.length - 1][0].toFixed(1)},${base}Z`;
    }
    seg = [];
  };
  row.cells.forEach((c, i) => { const v = c ? m.v(c) : null; if (v == null) flush(); else seg.push([x(i), y(v)]); });
  flush();
  svg.append(sv("path", {d: area || "M0,0", class: "area"}), sv("path", {d: line || "M0,0", class: "line"}));
  svg.append(sv("line", {x1: L, x2: W - R, y1: base, y2: base, class: "axis"}));
  drawTimeTicks(svg, d, L, pw, base);
  mc.cross = sv("line", {y1: T, y2: base, class: "cross", visibility: "hidden"});
  mc.dot = sv("circle", {r: 4, class: "cdot", visibility: "hidden"});
  mc.x = x; mc.y = y;
  svg.append(mc.cross, mc.dot);
  svg.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect(), i = Math.floor((e.clientX - r.left - L) / bw);
    if (i < 0 || i >= nb) { clearCursor(); return; }
    setCursor(d, row, i, e.clientX, e.clientY);
  });
  svg.addEventListener("pointerleave", clearCursor);
  wrap.append(svg);
}
function setCursor(d, row, i, cx, cy) {   // the crosshair follows the same moment on every GPU
  for (const mc of mcharts) {
    const c = mc.row.cells[i], v = c ? mc.m.v(c) : null;
    mc.cross.setAttribute("x1", mc.x(i)); mc.cross.setAttribute("x2", mc.x(i));
    mc.cross.setAttribute("visibility", "visible");
    if (v == null) mc.dot.setAttribute("visibility", "hidden");
    else { mc.dot.setAttribute("cx", mc.x(i)); mc.dot.setAttribute("cy", mc.y(v)); mc.dot.setAttribute("visibility", "visible"); }
  }
  const c = row.cells[i], title = slotTitle(d, row.index, i);
  if (!c) { tipShow(cx, cy, title, [], d.since + i * d.bin > d.now ? "Not yet" : "No data (logger not running)"); return; }
  const rows = [{value: fmtPct(c.u), label: "utilization"}, {value: fmtGB(c.m), label: "memory"}];
  if (c.p != null) rows.push({value: Math.round(c.p) + " W", label: "power"});
  if (c.t != null) rows.push({value: Math.round(c.t) + " °C", label: "temperature"});
  const users = Object.keys(c.users).sort((a, b) => c.users[b] - c.users[a]);
  tipShow(cx, cy, title, rows, users.length ? (users.length > 1 ? "Shared by " : "Used by ") + users.join(", ") : "Free");
}
function clearCursor() {
  for (const mc of mcharts) { mc.cross.setAttribute("visibility", "hidden"); mc.dot.setAttribute("visibility", "hidden"); }
  tipHide();
}

function renderTable(d) {
  const t = $("#usertable"); t.textContent = "";
  const cols = [["User", ""], ["GPU-hours", "num"], ["Share", ""], ["Avg utilization", "num"],
                ["Held while idle", "num"], ["Shared", "num"], ["Energy", "num"], ["Peak memory", "num"], ["Last active", "num"]];
  t.append(el("thead", null, el("tr", null, ...cols.map(([h, c]) => el("th", c, h)))));
  const body = el("tbody");
  const total = d.kpi.used_h;
  for (const u of d.users) {
    const share = total ? u.hours / total * 100 : 0;
    const fill = el("span"); fill.style.width = share + "%"; fill.style.background = colorOf(keyOf(u.user));
    body.append(el("tr", null,
      el("td", null, swatch(keyOf(u.user)), u.user),
      el("td", "num", fmtH(u.hours)),
      el("td", null, el("span", "sbar", fill), Math.round(share) + "%"),
      el("td", "num", fmtPct(u.avg_util)),
      el("td", "num", fmtH(u.idle_h), el("span", "dim", u.hours ? ` (${Math.round(u.idle_h / u.hours * 100)}%)` : "")),
      el("td", "num", u.shared == null ? "–" : Math.round(u.shared * 100) + "%"),
      el("td", "num", fmtKWh(u.kwh)),
      el("td", "num", fmtGB(u.peak_mem)),
      el("td", "num", ago(u.last_seen, d.now, d.interval))));
  }
  if (!d.users.length) { const td = el("td", "dim", "No GPU use recorded in this period."); td.colSpan = cols.length; body.append(el("tr", null, td)); }
  t.append(body);
  $("#sumcsv").href = "export/summary.csv?range=" + d.range;
  $("#tnote").textContent = "When several people share a GPU, its time is split between them by the GPU memory each holds; " +
    "utilization and energy use each person's own share of the compute (nvidia-smi pmon) when the driver reports it. " +
    `"Held while idle": holding GPU memory while your utilization is below ${d.idle_util}%. ` +
    `"Shared": how much of your GPU time someone else was on the same GPU. Energy counts GPU board power only.`;
}

function renderData(d) {
  const db = d.db, box = $("#dbstats"); box.textContent = "";
  const days = db.first != null ? (db.last - db.first) / 86400 : 0;
  const stat = (lab, val, sub) => el("div", "dstat", el("div", "lab", lab), el("div", "dval", val), sub ? el("div", "sub small", sub) : null);
  box.append(
    stat("Database size", fmtBytes(db.bytes), db.bytes_per_day ? `≈ ${fmtBytes(db.bytes_per_day * 30)} per month` : "Growth estimate after one day"),
    stat("GPU samples", db.gpu_rows.toLocaleString(), `one per GPU every ${d.interval} s`),
    stat("Process samples", db.proc_rows.toLocaleString(), `${db.users} users seen`),
    stat("Logging since", db.first != null ? fmtT(db.first, YEAR) : "–", db.first != null ? `${days < 1 ? "<1" : Math.round(days)} days of data` : null));
  const note = $("#dbnote"); note.textContent = "";
  note.append("Stored in ", el("code", null, db.path), ". Downloads contain everything logged so far; times in CSV files are the node's local time.");
}

// ---- utilization as a share of the theoretical maximum, every GPU x every range
const RANGE_LABEL = {"24h": "Last 24 hours", "7d": "Last 7 days", "30d": "Last 30 days", "90d": "Last 90 days"};
function renderPeriods(d) {
  const p = d.periods, t = $("#periods"); t.textContent = "";
  t.append(el("thead", null, el("tr", null, el("th", null, "GPU"), ...p.ranges.map(r => el("th", "num", RANGE_LABEL[r])))));
  const body = el("tbody");
  const row = (name, g, cls) => {
    const tr = el("tr", cls, el("td", null, name));
    for (const r of p.ranges) {
      const v = g.util[r], logged = g.logged[r];
      const fill = el("span"); fill.style.width = (v == null ? 0 : Math.max(0, Math.min(100, v))) + "%";
      const td = el("td", "num", el("span", "pv", v == null ? "–" : Math.round(v) + "%"), el("span", "pbar", fill),
                    v != null && logged < 0.95 ? el("span", "pl2", `${Math.round(logged * 100)}% of the period logged`) : null);
      td.title = v == null ? "No samples in this period" :
        `${name}, ${RANGE_LABEL[r].toLowerCase()}: ${v.toFixed(1)}% of the theoretical maximum (${Math.round(logged * 100)}% of the period logged)`;
      tr.append(td);
    }
    body.append(tr);
  };
  for (const g of p.gpus) row("GPU " + g.index, g);
  if (p.gpus.length > 1) row("All GPUs", p.all, "all");
  if (!p.gpus.length) { const td = el("td", "dim", "No samples yet."); td.colSpan = p.ranges.length + 1; body.append(el("tr", null, td)); }
  t.append(body);
  $("#pnote").textContent = "Average GPU utilization as reported by nvidia-smi (the share of time a kernel was running, not FLOPs), " +
    "as a percentage of the theoretical maximum: the GPU at 100% for the whole period. Only time when the logger was " +
    "recording counts; a cell says so when that is less than 95% of the period. Periods match the History ranges, which start at the beginning of an hour or day.";
}

// ---- logger uptime: when samples were recorded, and why they were not
const fmtWhen = t => fmtT(t, LONGDAY) + ", " + fmtT(t, HM);
function fmtSpan(a, b, ongoing) {
  const end = ongoing ? "now" : fmtT(a, YEAR) === fmtT(b, YEAR) ? fmtT(b, HM) : fmtWhen(b);
  return `${fmtWhen(a)} – ${end}`;
}
function causeText(o) {
  if (o.error) return (o.ongoing ? "Cannot read the GPUs right now: " : "Logger was running but could not read the GPUs: ") + o.error;
  if (o.reboot) return `Node rebooted at ${fmtWhen(o.reboot)}; logger restarted at ${fmtWhen(o.restart)}`;
  if (o.restart) return `Logger was not running; restarted at ${fmtWhen(o.restart)}`;
  if (o.ongoing) return "Not recording right now";
  return "No samples recorded (logger not running or node down)";
}
function renderUptime(d) {
  const u = d.uptime, box = $("#uptime"), stat = $("#uptime-stat"), leg = $("#uptime-legend"), t = $("#outages");
  box.textContent = ""; stat.textContent = ""; leg.textContent = ""; t.textContent = ""; $("#outnote").textContent = "";
  if (u.pct == null) { box.append(el("div", "empty", "No samples in this period.")); return; }
  const longest = Math.max(0, ...u.down.map(o => o.b - o.a));
  const pct = u.pct >= 0.9995 ? "100%" : (Math.floor(u.pct * 1000) / 10).toFixed(1) + "%";
  stat.append(el("b", null, pct), " uptime", u.start > d.since ? ` since logging started ${fmtWhen(u.start)}` : "", " · ",
              u.down.length ? `${u.down.length} outage${u.down.length > 1 ? "s" : ""}, longest ${fmtAge(longest)}` : "no outages");
  const item = (cls, text) => el("span", "li", el("span", "sw " + cls), text);
  leg.append(item("k-up", "Recording"), item("k-down", "Not recording"));
  if (u.starts.length) leg.append(el("span", "li", el("span", "mk", "▼"), "Logger started"));
  if (u.start > d.since) leg.append(item("k-pre", "Before logging started"));

  const W = Math.max(280, box.clientWidth), L = 52, R = 6, T = 10, SH = 22, H = T + SH + 22;
  const pw = W - L - R, span = d.end - d.since, xs = s => L + (s - d.since) / span * pw;
  const svg = sv("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
                         "aria-label": `Logger recording periods, ${pct} uptime; outages are listed below`});
  svg.append(sv("text", {x: 0, y: T + SH / 2 + 4, class: "rl"}, "Logger"));
  if (u.start > d.since) svg.append(sv("rect", {x: xs(d.since), y: T, width: xs(u.start) - xs(d.since), height: SH, rx: 2, class: "trk"}));
  // recording = thin green band; not recording = full-height red block, at least 3px wide so short gaps stay visible
  for (const [a, b] of u.up) svg.append(sv("rect", {x: xs(a), y: T + 5, width: Math.max(1, xs(b) - xs(a)), height: SH - 10, rx: 2, class: "upseg"}));
  const downs = u.down.map(o => { const w = Math.max(3, xs(o.b) - xs(o.a)); return {o, w, x0: Math.min(xs(o.a), L + pw - w)}; });
  for (const {x0, w} of downs) svg.append(sv("rect", {x: x0, y: T, width: w, height: SH, rx: 1, class: "downseg"}));
  for (const s of u.starts) svg.append(sv("path", {d: `M${xs(s) - 4},${T - 8}h8l-4,6z`, class: "startmk"}));
  drawTimeTicks(svg, d, L, pw, T + SH + 2);
  svg.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect(), px = e.clientX - r.left;
    if (px < L || px > L + pw) { tipHide(); return; }
    const at = d.since + (px - L) / pw * span;
    const hit = downs.find(x => px >= x.x0 - 2 && px <= x.x0 + x.w + 2);
    if (hit) {
      tipShow(e.clientX, e.clientY, "Not recording", [{value: fmtAge(hit.o.b - hit.o.a), label: fmtSpan(hit.o.a, hit.o.b, hit.o.ongoing)}], causeText(hit.o));
    } else if (at < u.start) {
      tipShow(e.clientX, e.clientY, "Before logging started", [], "First sample " + fmtWhen(u.start));
    } else {
      const seg = u.up.find(([a, b]) => at >= a && at <= b);
      if (!seg) { tipHide(); return; }
      const n = u.starts.filter(s => s >= seg[0] && s <= seg[1]).length;
      tipShow(e.clientX, e.clientY, "Recording", [{value: fmtAge(seg[1] - seg[0]), label: fmtSpan(seg[0], seg[1], seg[1] >= u.stop)}],
              n ? `Logger started ${n} time${n > 1 ? "s" : ""} in this stretch` : "");
    }
  });
  svg.addEventListener("pointerleave", tipHide);
  box.append(svg);

  if (!u.down.length) { $("#outnote").textContent = "No outages in this period."; return; }
  const MAX = 8, list = u.down.slice().reverse();   // most recent first
  t.append(el("thead", null, el("tr", null, el("th", null, "Not recording"), el("th", "num", "Duration"), el("th", null, "What happened"))));
  const body = el("tbody");
  for (const o of list.slice(0, MAX))
    body.append(el("tr", null, el("td", null, fmtSpan(o.a, o.b, o.ongoing)), el("td", "num", fmtAge(o.b - o.a)), el("td", "wrap", causeText(o))));
  t.append(body);
  $("#outnote").textContent = (list.length > MAX ? `${list.length - MAX} earlier outage${list.length - MAX > 1 ? "s" : ""} not listed. ` : "") +
    "Missing a single sample does not count as an outage.";
}

function renderHist() {
  const d = hist; TZ = d.tz_offset;
  renderLegends(d); renderKpis(d); renderCols(d); renderTimeline(d); renderMetrics(d); renderTable(d);
  renderPeriods(d); renderUptime(d); renderData(d);
  const off = d.tz_offset / 3600, utc = "UTC" + (off >= 0 ? "+" : "−") + Math.abs(off);
  $("#foot").textContent = `Sampled every ${d.interval} s · times are ${d.host} local time (${utc})`;
}

// ---- loading
async function getJSON(u) {
  const r = await fetch(u, {cache: "no-store"});
  if (!r.ok) throw new Error(r.status + " " + (await r.text()));
  return r.json();
}
async function loadLive() {
  try { live = await getJSON("api/live"); Object.assign(slots, live.slots); renderLive(); }
  catch (e) { const err = $("#err"); err.textContent = "Dashboard server unreachable: " + e.message; err.classList.add("on"); }
}
async function loadHist() {
  document.body.classList.add("loading");
  try { hist = await getJSON("api/history?range=" + range); Object.assign(slots, hist.slots); tipHide(); renderHist(); }
  catch (e) { const err = $("#err"); err.textContent = "Could not load history: " + e.message; err.classList.add("on"); }
  finally { document.body.classList.remove("loading"); }
}
for (const b of document.querySelectorAll(".seg button")) {
  b.setAttribute("aria-pressed", String(b.dataset.r === range));
  b.addEventListener("click", () => {
    range = b.dataset.r;
    history.replaceState(null, "", "#" + range);
    for (const o of document.querySelectorAll(".seg button")) o.setAttribute("aria-pressed", String(o === b));
    loadHist();
  });
}
let rt;
addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (hist) { renderCols(hist); renderTimeline(hist); renderMetrics(hist); renderUptime(hist); } }, 150); });
loadLive(); loadHist();
setInterval(loadLive, 10000);
setInterval(loadHist, 60000);
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
