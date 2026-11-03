"""Tests for gpu_dash.py. They use the fake nvidia-smi in tests/fake_bin, so no GPU is needed.

Run from the repository root:  python3 -m unittest -v
"""
import contextlib
import csv
import fcntl
import io
import json
import os
import pwd
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import zipfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gpu_dash  # noqa: E402

FAKE_BIN = os.path.join(HERE, "fake_bin")
ME = pwd.getpwuid(os.getuid()).pw_name


def gpu(index, util=0.0, mem_used=4.0, power=60.0, temp=35.0, procs=()):
    return {"index": index, "name": "Test GPU", "util": util, "mem_used": mem_used, "mem_total": 81920.0,
            "temp": temp, "power": power, "power_limit": 400.0, "procs": list(procs)}


def proc(user, mem, sm=None, pid=1):
    return {"user": user, "pid": pid, "mem": mem, "sm": sm, "cmd": "python train.py", "age": 10}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"PATH": FAKE_BIN + os.pathsep + os.environ["PATH"]})
        self.env.start()
        for k in ("FAKE_SMI_PROCS", "FAKE_SMI_BASIC", "FAKE_SMI_NO_PMON"):
            os.environ.pop(k, None)
        self.db = os.path.join(self.tmp.name, "usage.db")
        self.store = gpu_dash.Store(self.db)
        self.app = gpu_dash.App(self.store, 60)

    def tearDown(self):
        self.store.con.close()
        self.env.stop()
        self.tmp.cleanup()

    def check_zip(self, path, gpu_rows, proc_rows):
        """The export zip holds the database and one CSV per table, and they agree."""
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            self.assertEqual(len(names), 1 + len(gpu_dash.CSV_EXPORTS))
            db_name = next(n for n in names if n.endswith(".sqlite"))
            z.extract(db_name, self.tmp.name)
            csvs = {n.split("_")[0]: z.read(n).decode() for n in names if n.endswith(".csv")}
        copy = sqlite3.connect(os.path.join(self.tmp.name, db_name))
        self.assertEqual(copy.execute("SELECT COUNT(*) FROM gpu").fetchone()[0], gpu_rows)
        copy.close()
        self.assertEqual(set(csvs), {"gpu-samples", "process-samples", "logger-events"})
        self.assertEqual(len(csvs["gpu-samples"].splitlines()), gpu_rows + 1)
        self.assertEqual(len(csvs["process-samples"].splitlines()), proc_rows + 1)

    def fill(self, minutes, gpus_at):
        """Insert one sample per minute for the last `minutes` minutes."""
        now = int(time.time()) // 60 * 60
        for i in range(minutes):
            ts = now - (minutes - i) * 60
            self.store.insert(ts, 60, gpus_at(i))


class ReadingGpus(Base):
    def test_parse_pmon_handles_old_and_new_drivers(self):
        new = ("# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command\n"
               "# Idx           #    C/G      %      %      %      %      %      %    name\n"
               "    0      12345     C     45     20      -      -      -      -    python\n"
               "    0      12346     C      -      -      -      -      -      -    python\n"
               "    1          -     -      -      -      -      -      -      -    -\n")
        old = ("# gpu        pid  type    sm   mem   enc   dec   command\n"
               "# Idx          #   C/G     %     %     %     %   name\n"
               "    2       777     C    97    61     0     0   python3\n")
        self.assertEqual(gpu_dash.parse_pmon(new), {(0, 12345): 45.0, (0, 12346): 0.0})
        self.assertEqual(gpu_dash.parse_pmon(old), {(2, 777): 97.0})

    def test_snapshot_reads_gpus_processes_power_and_owner(self):
        os.environ["FAKE_SMI_PROCS"] = "0:%d:20480:85" % os.getpid()
        gpus = gpu_dash.Sampler().snapshot()
        self.assertEqual([g["index"] for g in gpus], [0, 1, 2])
        g0 = gpus[0]
        self.assertEqual(g0["name"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual((g0["util"], g0["power_limit"]), (85.0, 400.0))
        self.assertIsNotNone(g0["power"])
        self.assertIsNotNone(g0["temp"])
        p = g0["procs"][0]
        self.assertEqual((p["user"], p["pid"], p["mem"], p["sm"]), (ME, os.getpid(), 20480.0, 85.0))
        self.assertIn("python", p["cmd"])
        self.assertEqual(gpus[1]["procs"], [])

    def test_old_driver_without_power_fields_still_works(self):
        os.environ["FAKE_SMI_BASIC"] = "1"
        sampler = gpu_dash.Sampler()
        with contextlib.redirect_stderr(io.StringIO()):
            gpus = sampler.snapshot()
        self.assertEqual(len(gpus), 3)
        self.assertIsNone(gpus[0]["power"])
        self.assertIs(sampler.fields, gpu_dash.BASIC_GPU_FIELDS)

    def test_without_pmon_per_process_utilization_is_unknown(self):
        os.environ["FAKE_SMI_PROCS"] = "1:%d:1000:50" % os.getpid()
        os.environ["FAKE_SMI_NO_PMON"] = "1"
        gpus = gpu_dash.Sampler().snapshot()
        self.assertIsNone(gpus[1]["procs"][0]["sm"])


class Storage(Base):
    def test_old_database_is_upgraded_in_place(self):
        old = os.path.join(self.tmp.name, "old.db")
        con = sqlite3.connect(old)
        con.executescript("""
            CREATE TABLE gpu (ts INTEGER, dt REAL, gpu INTEGER, util REAL, mem_used REAL, mem_total REAL);
            CREATE TABLE proc (ts INTEGER, gpu INTEGER, user TEXT, pid INTEGER, mem REAL);
            CREATE TABLE users (user TEXT PRIMARY KEY, first_seen INTEGER);
            INSERT INTO gpu VALUES (100, 60, 0, 50, 1000, 81920);
            INSERT INTO proc VALUES (100, 0, 'alice', 1, 1000);""")
        con.commit()
        con.close()
        store = gpu_dash.Store(old)
        cols = {r[1] for r in store.con.execute("PRAGMA table_info(gpu)")}
        self.assertTrue({"power", "temp"} <= cols)
        self.assertIn("sm", {r[1] for r in store.con.execute("PRAGMA table_info(proc)")})
        self.assertEqual(store.con.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
        self.assertEqual(store.con.execute("SELECT util, power FROM gpu").fetchone(), (50.0, None))
        store.insert(200, 60, [gpu(0, util=90, power=300, procs=[proc("bob", 2000, 90)])])
        self.assertEqual(store.con.execute("SELECT COUNT(*) FROM gpu").fetchone()[0], 2)
        self.assertEqual(store.con.execute("PRAGMA user_version").fetchone()[0], gpu_dash.SCHEMA_VERSION)
        store.con.close()

    def test_info_reports_size_and_counts(self):
        self.fill(5, lambda i: [gpu(0, procs=[proc("alice", 100)]), gpu(1)])
        info = self.store.info()
        self.assertEqual((info["gpu_rows"], info["proc_rows"], info["users"]), (10, 5, 1))
        self.assertEqual(info["bytes"], os.path.getsize(self.db))
        self.assertGreater(info["bytes"], 0)


class SharedGpus(Base):
    def test_shared_gpu_is_split_by_memory_and_compute(self):
        # one hour: alice trains (60 GB, 80% SM), bob's idle notebook holds 20 GB on the same GPU
        self.fill(60, lambda i: [gpu(0, util=80, mem_used=80000, power=300, procs=[
            proc("alice", 60000, sm=80, pid=10), proc("bob", 20000, sm=0, pid=11)])])
        d = self.app._compute_history("24h")
        u = {r["user"]: r for r in d["users"]}
        self.assertAlmostEqual(u["alice"]["hours"], 0.75)
        self.assertAlmostEqual(u["bob"]["hours"], 0.25)
        self.assertAlmostEqual(d["kpi"]["used_h"], 1.0)        # never more than the GPU's own time
        self.assertAlmostEqual(u["alice"]["avg_util"], 80)
        self.assertAlmostEqual(u["bob"]["avg_util"], 0)
        self.assertAlmostEqual(u["bob"]["idle_h"], 0.25)       # idle although the GPU is busy
        self.assertAlmostEqual(u["alice"]["idle_h"], 0)
        self.assertAlmostEqual(u["alice"]["kwh"], 0.3)         # 300 W for an hour, all alice's compute
        self.assertAlmostEqual(u["bob"]["kwh"], 0)
        self.assertAlmostEqual(u["alice"]["shared"], 1.0)
        self.assertAlmostEqual(d["kpi"]["shared_h"], 1.0)
        self.assertAlmostEqual(d["kpi"]["energy_kwh"], 0.3)
        cell = next(c for c in d["timeline"][0]["cells"] if c)
        self.assertEqual(set(cell["users"]), {"alice", "bob"})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            gpu_dash.print_report(d)
        self.assertIn("theoretical maximum", out.getvalue())

    def test_even_split_without_memory_or_pmon_data(self):
        self.fill(30, lambda i: [gpu(0, util=50, power=None, procs=[
            proc("alice", None, pid=10), proc("bob", None, pid=11)])])
        d = self.app._compute_history("24h")
        u = {r["user"]: r for r in d["users"]}
        self.assertAlmostEqual(u["alice"]["hours"], 0.25)
        self.assertAlmostEqual(u["bob"]["hours"], 0.25)
        self.assertAlmostEqual(u["bob"]["avg_util"], 50)       # falls back to whole-GPU utilization
        self.assertIsNone(u["bob"]["kwh"])
        self.assertIsNone(d["kpi"]["energy_kwh"])

    def test_one_users_processes_count_once(self):
        self.fill(60, lambda i: [gpu(0, util=90, power=200, procs=[
            proc("alice", 1000, sm=45, pid=1), proc("alice", 1000, sm=45, pid=2)])])
        d = self.app._compute_history("7d")
        self.assertAlmostEqual(d["users"][0]["hours"], 1.0)
        self.assertAlmostEqual(d["users"][0]["shared"], 0.0)
        self.assertAlmostEqual(d["users"][0]["avg_util"], 90)
        self.assertAlmostEqual(sum(sum(b["users"].values()) for b in d["buckets"]), 1.0, places=3)


class UptimeAndUtilization(Base):
    def test_uptime_finds_outages_and_tolerates_one_missed_sample(self):
        t0 = 1_000_000
        samples = [(t0 + 60 * i, 60) for i in range(60) if i != 10 and not 30 <= i < 45]
        u = gpu_dash.compute_uptime(samples, [], t0 - 600, t0 + 3600, t0, 60)
        self.assertEqual(u["start"], t0)                      # nothing counts before logging started
        self.assertEqual([(o["a"], o["b"]) for o in u["down"]], [(t0 + 1800, t0 + 2700)])
        self.assertEqual(u["up"], [[t0, t0 + 1800], [t0 + 2700, t0 + 3600]])
        self.assertAlmostEqual(u["pct"], 0.75)

    def test_outages_are_explained_by_events(self):
        t0 = 1_000_000
        samples = [(t0 + 60 * i, 60) for i in range(100) if not (20 <= i < 40 or 60 <= i < 70 or i >= 90)]
        events = [(t0 + 60 * 39 + 30, "start", json.dumps({"boot": t0 + 60 * 35})),  # node rebooted
                  (t0 + 60 * 60, "error", "nvidia-smi failed: Unknown Error"),
                  (t0 + 60 * 65, "error", "nvidia-smi failed: Unknown Error")]
        u = gpu_dash.compute_uptime(samples, events, t0, t0 + 6000, t0, 60)
        reboot, smi, now = u["down"]
        self.assertEqual((reboot["reboot"], reboot["restart"]), (t0 + 2100, t0 + 2370))
        self.assertEqual((smi["error"], smi["errors"], smi["restart"]), ("nvidia-smi failed: Unknown Error", 2, None))
        self.assertTrue(now["ongoing"])
        self.assertEqual(u["starts"], [t0 + 2370])

    def test_failed_samples_are_logged_at_most_every_ten_minutes(self):
        class Broken:
            fail = True

            def snapshot(self):
                if self.fail:
                    raise RuntimeError("nvidia-smi failed: driver gone")
                return [gpu(0)]
        sampler = Broken()
        app = gpu_dash.App(self.store, 60, sampler=sampler)
        with contextlib.redirect_stderr(io.StringIO()):
            for _ in range(3):
                app.sample_once()
            sampler.fail = False
            app.sample_once()
            sampler.fail = True
            app.sample_once()
        rows = self.store.con.execute("SELECT kind, detail FROM events").fetchall()
        self.assertEqual(rows, [("error", "nvidia-smi failed: driver gone")] * 2)
        self.assertEqual(self.store.con.execute("SELECT COUNT(*) FROM gpu").fetchone()[0], 1)

    def test_four_hour_range_uses_quarter_hour_columns(self):
        self.fill(60, lambda i: [gpu(0, util=50, procs=[proc("alice", 1000, sm=50)])])
        d = self.app._compute_history("4h")
        self.assertEqual((d["bucket_s"], len(d["buckets"])), (900, 16))
        self.assertEqual({b["t"] - a["t"] for a, b in zip(d["buckets"], d["buckets"][1:])}, {900})
        self.assertAlmostEqual(sum(sum(b["users"].values()) for b in d["buckets"]), 1.0, places=3)
        self.assertTrue(all(sum(b["users"].values()) <= 0.25 + 1e-9 for b in d["buckets"]))
        self.assertEqual(len(d["timeline"][0]["cells"]), 120)                # 2-minute slots
        self.assertAlmostEqual(d["periods"]["all"]["util"]["4h"], 50)

    def test_utilization_of_theoretical_maximum_per_gpu_and_range(self):
        # GPU 0 at 50% for an hour; GPU 1 at 100% for half an hour and idle for the other half
        self.fill(60, lambda i: [gpu(0, util=50), gpu(1, util=100 if i < 30 else 0)])
        p = self.app._compute_history("7d")["periods"]
        self.assertEqual(p["ranges"], list(gpu_dash.RANGES))
        for r in p["ranges"]:
            self.assertAlmostEqual(p["gpus"][0]["util"][r], 50)
            self.assertAlmostEqual(p["gpus"][1]["util"][r], 50)
            self.assertAlmostEqual(p["all"]["util"][r], 50)
        self.assertGreater(p["gpus"][0]["logged"]["24h"], p["gpus"][0]["logged"]["90d"])
        self.assertLessEqual(p["gpus"][0]["logged"]["24h"], 1.0)


class Exports(Base):
    def test_csv_pages_through_every_row(self):
        self.fill(125, lambda i: [gpu(0, procs=[proc("alice", 100, 5)])])
        text = b"".join(self.store.iter_csv("gpu", batch=50)).decode()
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(rows[0], gpu_dash.CSV_EXPORTS["gpu"][1])
        self.assertEqual(len(rows), 126)
        proc_rows = list(csv.reader(io.StringIO(b"".join(self.store.iter_csv("proc")).decode())))
        self.assertEqual(proc_rows[1][3:], ["alice", "1", "100.0", "5.0"])

    def test_empty_database_exports_headers(self):
        text = b"".join(self.store.iter_csv("proc")).decode()
        self.assertEqual(text.strip(), ",".join(gpu_dash.CSV_EXPORTS["proc"][1]))

    def test_export_all_writes_one_zip(self):
        self.fill(10, lambda i: [gpu(0, procs=[proc("alice", 100)]), gpu(1)])
        paths = gpu_dash.export_all(self.store, "node", os.path.join(self.tmp.name, "export"))
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith(".zip"))
        self.check_zip(paths[0], gpu_rows=20, proc_rows=10)


class WebAndCli(Base):
    def test_http_endpoints(self):
        os.environ["FAKE_SMI_PROCS"] = "2:%d:30000:60" % os.getpid()
        self.app.sample_once()
        server = gpu_dash.ThreadingHTTPServer(("127.0.0.1", 0), gpu_dash.make_handler(self.app))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = "http://127.0.0.1:%d/" % server.server_address[1]

        def get(path):
            with urllib.request.urlopen(base + path) as r:
                return r.status, r.headers, r.read()
        try:
            status, _, body = get("")
            self.assertIn(b"<title>GPU usage</title>", body)
            live = json.loads(get("api/live")[2])
            self.assertEqual(live["gpus"][2]["procs"][0]["user"], ME)
            for rng in gpu_dash.RANGES:
                hist = json.loads(get("api/history?range=" + rng)[2])
                self.assertEqual(hist["range"], rng)
                self.assertEqual(hist["users"][0]["user"], ME)
                self.assertGreater(hist["db"]["bytes"], 0)
            _, headers, body = get("export/gpu.csv")
            self.assertIn("attachment", headers["Content-Disposition"])
            self.assertEqual(len(body.decode().strip().splitlines()), 4)  # header + 3 GPUs
            self.assertIn(ME.encode(), get("export/proc.csv")[2])
            self.assertTrue(get("export/summary.csv?range=24h")[2].startswith(b"user,gpu_hours"))
            self.assertTrue(get("export/events.csv")[2].startswith(b"time,unix_time,event,detail"))
            _, headers, body = get("export/usage.sqlite")
            self.assertTrue(body.startswith(b"SQLite format 3"))
            self.assertEqual(int(headers["Content-Length"]), len(body))
            _, headers, body = get("export/all.zip")
            self.assertEqual(headers["Content-Type"], "application/zip")
            self.assertIn(".zip", headers["Content-Disposition"])
            zpath = os.path.join(self.tmp.name, "all.zip")
            with open(zpath, "wb") as f:
                f.write(body)
            self.check_zip(zpath, gpu_rows=3, proc_rows=1)
        finally:
            server.shutdown()
            server.server_close()

    def test_cli_report_export_and_data_dir(self):
        data_dir = os.path.join(self.tmp.name, "scratch", "gpu-dash")   # created on demand
        os.environ["FAKE_SMI_PROCS"] = "0:%d:1000:30" % os.getpid()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            gpu_dash.main(["--data-dir", data_dir, "--report", "7d"])
        self.assertTrue(os.path.exists(os.path.join(data_dir, "usage.db")))
        self.assertIn("no GPU use recorded", out.getvalue())
        self.assertIn("database:", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            gpu_dash.main(["--data-dir", data_dir, "--report", "7d", "--csv"])
        self.assertEqual(out.getvalue().strip(), ",".join(gpu_dash.SUMMARY_HEADER))
        with contextlib.redirect_stdout(io.StringIO()):
            gpu_dash.main(["--data-dir", data_dir, "--export", os.path.join(self.tmp.name, "out")])
        exported = os.listdir(os.path.join(self.tmp.name, "out"))
        self.assertEqual(len(exported), 1)
        self.assertTrue(exported[0].startswith("gpu-usage-export_") and exported[0].endswith(".zip"))

    def test_second_copy_exits_when_one_is_running(self):
        data_dir = os.path.join(self.tmp.name, "dd")
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "gpu_dash.lock"), "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                gpu_dash.main(["--data-dir", data_dir, "--port", "0"])
        self.assertIn("already running", out.getvalue())


if __name__ == "__main__":
    unittest.main()
