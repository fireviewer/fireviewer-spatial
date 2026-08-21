"""Container resource telemetry for provider-neutral Map Builder runs.

The monitor reads Linux cgroup v2 and procfs counters.  It intentionally has no
AWS dependency: the same code runs in local Docker, EC2 and a future Batch job.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any

GIB = 1024**3
MIB = 1024**2


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[1].lstrip("-").isdigit():
            values[fields[0]] = int(fields[1])
    return values


def _cgroup_root() -> Path | None:
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            relative = fields[2].lstrip("/")
            candidate = Path("/sys/fs/cgroup") / relative
            return candidate if candidate.is_dir() else None
    return None


def _io_totals(path: Path) -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0, 0
    for line in lines:
        for field in line.split()[1:]:
            key, separator, value = field.partition("=")
            if not separator or not value.isdigit():
                continue
            if key == "rbytes":
                read_bytes += int(value)
            elif key == "wbytes":
                write_bytes += int(value)
    return read_bytes, write_bytes


def _system_cpu() -> tuple[int, int] | None:
    try:
        first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    fields = first.split()
    if not fields or fields[0] != "cpu":
        return None
    try:
        counters = [int(value) for value in fields[1:]]
    except ValueError:
        return None
    total = sum(counters)
    iowait = counters[4] if len(counters) > 4 else 0
    return total, iowait


def _system_ram_used() -> int | None:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        number = rest.strip().split()[0]
        if number.isdigit():
            values[key] = int(number) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    return total - available if total is not None and available is not None else None


class ResourceMonitor:
    """Sample cgroup and filesystem counters while the builder is running."""

    def __init__(self, scratch_root: Path, interval_seconds: float = 1.0) -> None:
        self.scratch_root = scratch_root
        self.interval_seconds = interval_seconds
        self.cgroup = _cgroup_root()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._initial_cpu_usec: int | None = None
        self._initial_io = (0, 0)
        self._initial_disk_used = 0
        self._initial_system_cpu: tuple[int, int] | None = None
        self._last_time = 0.0
        self._last_cpu_usec: int | None = None
        self._last_io = (0, 0)
        self._cpu_peak = 0.0
        self._read_peak_mbps = 0.0
        self._write_peak_mbps = 0.0
        self._memory_peak = 0
        self._system_memory_peak = 0
        self._scratch_peak = 0

    def _cpu_usage_usec(self) -> int | None:
        if self.cgroup is None:
            return None
        return _read_key_values(self.cgroup / "cpu.stat").get("usage_usec")

    def _io(self) -> tuple[int, int]:
        return _io_totals(self.cgroup / "io.stat") if self.cgroup else (0, 0)

    def _sample(self) -> None:
        now = time.monotonic()
        cpu_usec = self._cpu_usage_usec()
        read_bytes, write_bytes = self._io()
        elapsed = now - self._last_time
        if elapsed > 0:
            if cpu_usec is not None and self._last_cpu_usec is not None:
                cpu_percent = (cpu_usec - self._last_cpu_usec) / (elapsed * 10_000)
                self._cpu_peak = max(self._cpu_peak, cpu_percent)
            self._read_peak_mbps = max(
                self._read_peak_mbps,
                (read_bytes - self._last_io[0]) / elapsed / MIB,
            )
            self._write_peak_mbps = max(
                self._write_peak_mbps,
                (write_bytes - self._last_io[1]) / elapsed / MIB,
            )
        self._last_time = now
        self._last_cpu_usec = cpu_usec
        self._last_io = (read_bytes, write_bytes)

        if self.cgroup is not None:
            current = _read_int(self.cgroup / "memory.current") or 0
            peak = _read_int(self.cgroup / "memory.peak") or current
            self._memory_peak = max(self._memory_peak, current, peak)
        system_used = _system_ram_used() or 0
        self._system_memory_peak = max(self._system_memory_peak, system_used)
        try:
            disk_used = shutil.disk_usage(self.scratch_root).used
        except OSError:
            disk_used = self._initial_disk_used
        self._scratch_peak = max(
            self._scratch_peak,
            max(0, disk_used - self._initial_disk_used),
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self._started = time.monotonic()
        self._last_time = self._started
        self._initial_cpu_usec = self._cpu_usage_usec()
        self._last_cpu_usec = self._initial_cpu_usec
        self._initial_io = self._io()
        self._last_io = self._initial_io
        self._initial_system_cpu = _system_cpu()
        self._initial_disk_used = shutil.disk_usage(self.scratch_root).used
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._sample()
        runtime = max(time.monotonic() - self._started, 0.000001)
        final_cpu = self._cpu_usage_usec()
        final_read, final_write = self._io()
        cpu_avg: float | None = None
        if final_cpu is not None and self._initial_cpu_usec is not None:
            cpu_avg = (final_cpu - self._initial_cpu_usec) / (runtime * 10_000)

        iowait_percent: float | None = None
        final_system_cpu = _system_cpu()
        if self._initial_system_cpu and final_system_cpu:
            total_delta = final_system_cpu[0] - self._initial_system_cpu[0]
            wait_delta = final_system_cpu[1] - self._initial_system_cpu[1]
            if total_delta > 0:
                iowait_percent = wait_delta * 100.0 / total_delta

        memory_events = (
            _read_key_values(self.cgroup / "memory.events") if self.cgroup else {}
        )
        swap_peak = (
            _read_int(self.cgroup / "memory.swap.peak") if self.cgroup else None
        )
        read_delta = max(0, final_read - self._initial_io[0])
        write_delta = max(0, final_write - self._initial_io[1])
        return {
            "monitor": "linux-cgroup-v2-procfs.v1",
            "runtime_seconds": round(runtime, 3),
            "cpu_avg_percent": round(cpu_avg, 3) if cpu_avg is not None else None,
            "cpu_peak_percent": round(self._cpu_peak, 3),
            "ram_peak_gb": round(self._memory_peak / GIB, 6),
            "system_ram_peak_gb": round(self._system_memory_peak / GIB, 6),
            "swap_peak_gb": round((swap_peak or 0) / GIB, 6),
            "oom_events": memory_events.get("oom", 0),
            "oom_kill_events": memory_events.get("oom_kill", 0),
            "disk_read_gb": round(read_delta / GIB, 6),
            "disk_write_gb": round(write_delta / GIB, 6),
            "disk_read_mbps": round(read_delta / runtime / MIB, 3),
            "disk_write_mbps": round(write_delta / runtime / MIB, 3),
            "disk_read_peak_mbps": round(self._read_peak_mbps, 3),
            "disk_write_peak_mbps": round(self._write_peak_mbps, 3),
            "io_wait_percent": (
                round(iowait_percent, 3) if iowait_percent is not None else None
            ),
            "scratch_peak_gb": round(self._scratch_peak / GIB, 6),
        }
