"""Lightweight, cross-platform CPU and RAM monitoring for evaluation runs.

Collects system CPU%, process CPU%, RAM% and process RSS (bytes) at a fixed
sampling interval in a background thread. Designed to wrap evaluation loops
and return summary stats (avg, max, p95, peak RSS).

Falls back gracefully if psutil is not installed.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


@dataclass
class MonitorSummary:
    cpu_pct_avg: Optional[float]
    cpu_pct_max: Optional[float]
    cpu_proc_pct_avg: Optional[float]
    cpu_proc_pct_max: Optional[float]
    ram_pct_avg: Optional[float]
    ram_pct_max: Optional[float]
    rss_peak_bytes: Optional[int]
    cpu_time_user_s: Optional[float]
    cpu_time_system_s: Optional[float]
    cpu_time_total_s: Optional[float]
    avg_parallelism: Optional[float]
    samples: int
    interval_s: float


class PerfMonitor:
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu: List[float] = []
        self._cpu_proc: List[float] = []
        self._ram_pct: List[float] = []
        self._rss_peak: int = 0
        self._enabled = psutil is not None
        self._proc = psutil.Process(os.getpid()) if self._enabled else None  # type: ignore
        self._t_start: Optional[float] = None
        self._cpu_time_start: Optional[float] = None

    def start(self) -> "PerfMonitor":
        if not self._enabled:
            return self

        # Prime CPU percent meters
        psutil.cpu_percent(interval=None)  # type: ignore
        try:
            assert self._proc is not None
            self._proc.cpu_percent(interval=None)  # type: ignore
        except Exception:
            pass

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="PerfMonitor", daemon=True)
        # record start wall time and CPU time
        self._t_start = time.perf_counter()
        try:
            assert self._proc is not None
            ct = self._proc.cpu_times()
            self._cpu_time_start = float(getattr(ct, "user", 0.0) + getattr(ct, "system", 0.0))
        except Exception:
            self._cpu_time_start = None
        self._thread.start()
        return self

    def stop_and_summarize(self) -> MonitorSummary:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval_s * 4)
            self._thread = None

        def _avg(xs: List[float]) -> Optional[float]:
            return sum(xs) / len(xs) if xs else None

        def _max(xs: List[float]) -> Optional[float]:
            return max(xs) if xs else None

        # end CPU time and wall clock
        cpu_time_user = None
        cpu_time_sys = None
        cpu_time_total = None
        avg_parallelism = None
        wall = None
        try:
            if self._proc is not None:
                ct_end = self._proc.cpu_times()
                cpu_time_user = float(getattr(ct_end, "user", 0.0))
                cpu_time_sys = float(getattr(ct_end, "system", 0.0))
                cpu_time_total = cpu_time_user + cpu_time_sys
                if self._cpu_time_start is not None:
                    cpu_time_total = cpu_time_total - self._cpu_time_start
        except Exception:
            pass
        try:
            if self._t_start is not None:
                wall = time.perf_counter() - self._t_start
        except Exception:
            pass
        if cpu_time_total is not None and wall and wall > 0:
            avg_parallelism = cpu_time_total / wall

        return MonitorSummary(
            cpu_pct_avg=_avg(self._cpu),
            cpu_pct_max=_max(self._cpu),
            cpu_proc_pct_avg=_avg(self._cpu_proc),
            cpu_proc_pct_max=_max(self._cpu_proc),
            ram_pct_avg=_avg(self._ram_pct),
            ram_pct_max=_max(self._ram_pct),
            rss_peak_bytes=self._rss_peak if self._rss_peak > 0 else None,
            cpu_time_user_s=cpu_time_user,
            cpu_time_system_s=cpu_time_sys,
            cpu_time_total_s=cpu_time_total,
            avg_parallelism=avg_parallelism,
            samples=len(self._cpu),
            interval_s=self.interval_s,
        )

    # --- internals ---
    def _run(self) -> None:
        assert psutil is not None and self._proc is not None
        while not self._stop.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                self._cpu.append(float(cpu))

                cpu_proc = self._proc.cpu_percent(interval=None)
                self._cpu_proc.append(float(cpu_proc))

                vm = psutil.virtual_memory()
                self._ram_pct.append(float(vm.percent))

                rss = self._proc.memory_info().rss
                if rss > self._rss_peak:
                    self._rss_peak = rss
            except Exception:
                # Never let monitoring crash the evaluation
                pass
            finally:
                time.sleep(self.interval_s)
