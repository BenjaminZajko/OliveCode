"""Robust hardware detection.

Design contract: this module MUST NEVER raise. Every individual probe is
wrapped in a try/except that returns a safe default ("unknown") and
records a short reason. The result is a HardwareProfile the rest of the
advisor can use, including when most fields are unknown — recommendation
logic must then fall back to conservative assumptions rather than failing.

We support:
  - Linux, macOS, Windows (and degrade gracefully on anything else)
  - CPU model + logical core count via /proc/cpuinfo (Linux), sysctl (macOS),
    wmic/POWERShell (Windows), with platform.processor() as a final fallback.
  - Total RAM via /proc/meminfo, sysctl, GlobalMemoryStatusEx, psutil.
  - GPU detection on:
        * NVIDIA        -> `nvidia-smi` if installed
        * Apple Silicon -> detected from platform.machine() == "arm64" on macOS
                           (unified memory, no separate VRAM)
        * AMD           -> `rocm-smi` if installed, else attempt WMI on Windows
        * Integrated    -> reported as such on Windows / Linux iGPUs
        * None          -> reported as "no dedicated GPU detected"
  - Disk free space via shutil.disk_usage on the user's home / current dir.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """One detected GPU. `vram_gb` is None for Apple Silicon unified memory
    and unknown/integrated cases where the number isn't separately meaningful
    (we use system RAM as the effective memory pool in those cases)."""
    vendor: str                # "nvidia" | "amd" | "apple" | "intel" | "unknown"
    name: str
    vram_gb: Optional[float] = None
    driver: Optional[str] = None
    notes: str = ""            # e.g. "unified memory" or "integrated"


@dataclass
class HardwareProfile:
    """Everything we managed to find out about the host. Anything that
    couldn't be detected is None and paired with a `notes` entry so the UI
    can tell the user "could not detect X"."""
    os_name: Optional[str] = None            # e.g. "macOS", "Linux", "Windows"
    os_version: Optional[str] = None
    arch: Optional[str] = None               # e.g. "arm64", "x86_64"
    cpu_model: Optional[str] = None
    cpu_cores_physical: Optional[int] = None
    cpu_cores_logical: Optional[int] = None
    ram_total_gb: Optional[float] = None
    gpus: list[GPUInfo] = field(default_factory=list)
    disk_free_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    notes: list[str] = field(default_factory=list)  # human-readable caveats

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    # --- Convenience predicates the recommender uses ---

    @property
    def has_nvidia_gpu(self) -> bool:
        return any(g.vendor == "nvidia" and (g.vram_gb or 0) >= 4 for g in self.gpus)

    @property
    def best_dedicated_vram_gb(self) -> float:
        """VRAM of the best dedicated GPU, or 0.0 if none. Apple Silicon and
        integrated GPUs return 0.0 here — they share system RAM."""
        dedicated = [g for g in self.gpus if g.vendor in ("nvidia", "amd") and g.vram_gb]
        if not dedicated:
            return 0.0
        return max(g.vram_gb for g in dedicated)  # type: ignore[arg-type]

    @property
    def is_apple_silicon(self) -> bool:
        return self.os_name == "macOS" and (self.arch or "").startswith("arm")

    @property
    def effective_memory_gb(self) -> float:
        """Memory pool a model can actually use.

        - Apple Silicon: unified memory, so the full RAM is available to the
          GPU. We still leave a sensible headroom below.
        - Dedicated NVIDIA/AMD GPU: VRAM is the hard ceiling; the model must
          fit there. (CPU offload is theoretically possible with llama.cpp
          but it's a footgun for first-run UX, so we don't recommend it.)
        - Everything else: system RAM.
        """
        if self.is_apple_silicon and self.ram_total_gb:
            return self.ram_total_gb
        if self.best_dedicated_vram_gb > 0:
            return self.best_dedicated_vram_gb
        return self.ram_total_gb or 0.0


# ---------------------------------------------------------------------------
# Tiny safe-exec helper
# ---------------------------------------------------------------------------

def _safe_run(cmd: list[str], timeout: float = 4.0) -> tuple[int, str, str]:
    """Run a subprocess; on any failure return (-1, "", ""). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception:
        return -1, "", ""


# ---------------------------------------------------------------------------
# OS / arch
# ---------------------------------------------------------------------------

def _detect_os(profile: HardwareProfile) -> None:
    try:
        system = platform.system()
        if system == "Darwin":
            profile.os_name = "macOS"
            try:
                # e.g. "14.5" on modern macOS
                ver = platform.mac_ver()[0]
                profile.os_version = ver or None
            except Exception:
                pass
        elif system == "Linux":
            profile.os_name = "Linux"
            # /etc/os-release is the modern standard, present on basically
            # every distro we care about
            try:
                with open("/etc/os-release", "r", encoding="utf-8", errors="replace") as f:
                    data = f.read()
                m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', data, re.MULTILINE)
                if m:
                    profile.os_version = m.group(1).strip()
            except Exception:
                pass
            if not profile.os_version:
                profile.os_version = platform.release() or None
        elif system == "Windows":
            profile.os_name = "Windows"
            try:
                profile.os_version = platform.win32_ver()[0] or None
            except Exception:
                pass
        else:
            profile.os_name = system or "Unknown"
            profile.notes.append(f"Unrecognized OS '{system}'; treating as unknown.")
    except Exception as e:
        profile.os_name = None
        profile.notes.append(f"OS detection failed: {e!r}")

    try:
        profile.arch = platform.machine() or None
    except Exception:
        profile.arch = None


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def _parse_lscpu() -> Optional[tuple[str, int, int]]:
    """Linux: try lscpu, which gives a clean summary on basically every distro."""
    code, out, _ = _safe_run(["lscpu"])
    if code != 0 or not out:
        return None
    model = None
    cores_phys = None
    cores_log = None
    for line in out.splitlines():
        if line.startswith("Model name:"):
            model = line.split(":", 1)[1].strip()
        elif line.startswith("CPU(s):") and "Logical" not in line and "socket" not in line.lower():
            # The plain "CPU(s):" line is the logical count on lscpu
            try:
                cores_log = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Core(s) per socket:"):
            try:
                cores_phys = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Socket(s):"):
            try:
                sockets = int(line.split(":", 1)[1].strip())
                if cores_phys is not None:
                    cores_phys = cores_phys * sockets
            except Exception:
                pass
    if model or cores_log:
        return model, cores_phys or 0, cores_log or 0
    return None


def _parse_proc_cpuinfo() -> Optional[tuple[str, int]]:
    """Fallback Linux: /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
    except Exception:
        return None
    model = None
    m = re.search(r"model name\s*:\s*(.+)", data)
    if m:
        model = m.group(1).strip()
    # Count unique physical ids; on simpler boxes that's 1 and we fall back
    # to counting processors.
    phys_ids = set(re.findall(r"physical id\s*:\s*(\d+)", data))
    if phys_ids:
        cores_per = len(set(re.findall(r"cpu cores\s*:\s*(\d+)", data)))
        try:
            return model, max(1, len(phys_ids) * (cores_per or 1))
        except Exception:
            pass
    # Last resort: count "processor :" lines
    n = len(re.findall(r"^processor\s*:\s*\d+", data, re.MULTILINE))
    return model, (n or 0)


def _detect_cpu_macos(profile: HardwareProfile) -> None:
    # sysctl gives us both model and logical core count.
    _, brand_out, _ = _safe_run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if brand_out.strip():
        profile.cpu_model = brand_out.strip()

    _, nphys_out, _ = _safe_run(["sysctl", "-n", "hw.physicalcpu"])
    _, nlog_out, _ = _safe_run(["sysctl", "-n", "hw.logicalcpu"])
    try:
        profile.cpu_cores_physical = int(nphys_out.strip())
    except Exception:
        pass
    try:
        profile.cpu_cores_logical = int(nlog_out.strip())
    except Exception:
        pass

    # Apple Silicon check: arm64 + Apple brand string typically includes "Apple"
    if (profile.arch or "").startswith("arm") and profile.cpu_model and "Apple" in profile.cpu_model:
        profile.notes.append("Apple Silicon detected: CPU and GPU share unified memory.")


def _detect_cpu_windows(profile: HardwareProfile) -> None:
    # Prefer wmic (still present on Win10 and most Win11), fall back to env.
    code, out, _ = _safe_run(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/value"])
    if code == 0 and out:
        model = None
        cores = None
        logical = None
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Name="):
                model = line.split("=", 1)[1].strip()
            elif line.startswith("NumberOfCores="):
                try:
                    cores = int(line.split("=", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("NumberOfLogicalProcessors="):
                try:
                    logical = int(line.split("=", 1)[1].strip())
                except Exception:
                    pass
        if model:
            profile.cpu_model = model
        if cores is not None:
            profile.cpu_cores_physical = cores
        if logical is not None:
            profile.cpu_cores_logical = logical

    if not profile.cpu_model:
        try:
            profile.cpu_model = platform.processor() or None
        except Exception:
            pass
    if profile.cpu_cores_logical is None:
        try:
            profile.cpu_cores_logical = os.cpu_count()
        except Exception:
            pass


def _detect_cpu(profile: HardwareProfile) -> None:
    try:
        if profile.os_name == "macOS":
            _detect_cpu_macos(profile)
        elif profile.os_name == "Windows":
            _detect_cpu_windows(profile)
        else:
            # Linux or unknown
            res = _parse_lscpu()
            if res:
                model, phys, log = res
                if model:
                    profile.cpu_model = model
                if phys:
                    profile.cpu_cores_physical = phys
                if log:
                    profile.cpu_cores_logical = log
            if not profile.cpu_model or profile.cpu_cores_logical is None:
                res2 = _parse_proc_cpuinfo()
                if res2:
                    model, log = res2
                    if model and not profile.cpu_model:
                        profile.cpu_model = model
                    if log and not profile.cpu_cores_logical:
                        profile.cpu_cores_logical = log
    except Exception as e:
        profile.notes.append(f"CPU detection failed: {e!r}")

    if profile.cpu_cores_logical is None:
        try:
            profile.cpu_cores_logical = os.cpu_count()
        except Exception:
            pass

    if profile.cpu_model is None and profile.cpu_cores_logical is None:
        profile.notes.append("Could not detect CPU model or core count.")


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def _detect_ram_linux(profile: HardwareProfile) -> None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    profile.ram_total_gb = round(kb / 1024 / 1024, 2)
                    return
    except Exception:
        pass
    # sysconf fallback
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # _SC_PHYS_PAGES * _SC_PAGESIZE
        phys_pages = libc.sysconf(80)  # _SC_PHYS_PAGES
        page_size = libc.sysconf(30)   # _SC_PAGESIZE
        if phys_pages > 0 and page_size > 0:
            profile.ram_total_gb = round(phys_pages * page_size / 1024 / 1024 / 1024, 2)
    except Exception:
        pass


def _detect_ram_macos(profile: HardwareProfile) -> None:
    _, out, _ = _safe_run(["sysctl", "-n", "hw.memsize"])
    try:
        if out.strip():
            profile.ram_total_gb = round(int(out.strip()) / 1024 / 1024 / 1024, 2)
    except Exception:
        pass


def _detect_ram_windows(profile: HardwareProfile) -> None:
    # ctypes is the cleanest no-dep path on Windows
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            profile.ram_total_gb = round(stat.ullTotalPhys / 1024 / 1024 / 1024, 2)
            return
    except Exception:
        pass

    # Fallback: wmic
    code, out, _ = _safe_run(["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"])
    if code == 0 and out:
        m = re.search(r"TotalPhysicalMemory=(\d+)", out)
        if m:
            try:
                profile.ram_total_gb = round(int(m.group(1)) / 1024 / 1024 / 1024, 2)
            except Exception:
                pass


def _detect_ram(profile: HardwareProfile) -> None:
    try:
        if profile.os_name == "macOS":
            _detect_ram_macos(profile)
        elif profile.os_name == "Windows":
            _detect_ram_windows(profile)
        else:
            _detect_ram_linux(profile)
    except Exception as e:
        profile.notes.append(f"RAM detection failed: {e!r}")

    # Last-ditch: try psutil if installed
    if profile.ram_total_gb is None:
        try:
            import psutil  # type: ignore
            profile.ram_total_gb = round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2)
        except Exception:
            pass

    if profile.ram_total_gb is None:
        profile.notes.append("Could not detect total RAM.")


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _detect_nvidia_gpus() -> list[GPUInfo]:
    """Use nvidia-smi if present. We deliberately don't require the NVIDIA
    Python libs — nvidia-smi is the universal probe."""
    code, out, _ = _safe_run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if code != 0 or not out.strip():
        return []
    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        vram_mb = None
        driver = None
        try:
            vram_mb = float(parts[1])
        except Exception:
            pass
        if len(parts) >= 3 and parts[2]:
            driver = parts[2]
        gpus.append(GPUInfo(
            vendor="nvidia",
            name=name,
            vram_gb=round(vram_mb / 1024, 2) if vram_mb else None,
            driver=driver,
        ))
    return gpus


def _detect_amd_gpus() -> list[GPUInfo]:
    """Best-effort AMD detection. rocm-smi is the clean path; on Windows we
    try wmic. Many AMD machines have no usable ROCm install, in which case
    we just won't find a dedicated GPU and the recommender will route to
    the CPU/RAM path."""
    code, out, _ = _safe_run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
    if code == 0 and out.strip():
        # rocm-smi output is human-formatted; we just grab a name.
        name = None
        vram = None
        for line in out.splitlines():
            if "Card" in line and ":" in line and "Series" not in line:
                # e.g. "Card series:          Navi 31"
                pass
            if "GPU[" in line and "]" in line:
                # e.g. "GPU[0]          : Navi 31 [Radeon RX 7900 XT]"
                m = re.search(r":\s*(.+)", line)
                if m and not name:
                    name = m.group(1).strip()
            if "vram" in line.lower() and "total" in line.lower():
                m = re.search(r"(\d+)\s*M", line)
                if m and not vram:
                    try:
                        vram = round(int(m.group(1)) / 1024, 2)
                    except Exception:
                        pass
        if name:
            return [GPUInfo(vendor="amd", name=name, vram_gb=vram,
                            notes="rocm-smi" if vram else "rocm-smi (VRAM not parsed)")]
    return []


def _detect_apple_gpu(profile: HardwareProfile) -> None:
    if profile.is_apple_silicon:
        # Add a single representative Apple GPU entry. We deliberately set
        # vram_gb=None and let the recommender use unified memory.
        chip = profile.cpu_model or "Apple Silicon"
        profile.gpus.append(GPUInfo(
            vendor="apple",
            name=chip,
            vram_gb=None,
            notes="Unified memory shared with the system; no separate VRAM.",
        ))


def _detect_intel_or_integrated_linux(profile: HardwareProfile) -> None:
    """Last-resort Intel / iGPU detection on Linux via lspci."""
    code, out, _ = _safe_run(["lspci"])
    if code != 0 or not out:
        return
    for line in out.splitlines():
        if "VGA compatible controller" in line or "3D controller" in line:
            m = re.search(r":\s*(.+)$", line)
            if not m:
                continue
            name = m.group(1).strip()
            vendor = "unknown"
            lname = name.lower()
            if "nvidia" in lname:
                # We already ran nvidia-smi; if it returned nothing, this
                # is probably a Tesla/older card without working drivers.
                continue
            if "amd" in lname or "radeon" in lname or "ati" in lname:
                vendor = "amd"
            elif "intel" in lname:
                vendor = "intel"
            profile.gpus.append(GPUInfo(
                vendor=vendor,  # type: ignore[arg-type]
                name=name,
                vram_gb=None,
                notes="Integrated / no separate VRAM detected.",
            ))


def _detect_gpus(profile: HardwareProfile) -> None:
    # Apple first (it's a different beast)
    _detect_apple_gpu(profile)

    # Try NVIDIA
    nvs = []
    try:
        nvs = _detect_nvidia_gpus()
    except Exception as e:
        profile.notes.append(f"NVIDIA probe failed: {e!r}")
    profile.gpus.extend(nvs)

    # Try AMD
    if not nvs:
        try:
            profile.gpus.extend(_detect_amd_gpus())
        except Exception as e:
            profile.notes.append(f"AMD probe failed: {e!r}")

    # Integrated / fallback
    if not profile.gpus and profile.os_name in ("Linux",):
        try:
            _detect_intel_or_integrated_linux(profile)
        except Exception as e:
            profile.notes.append(f"Integrated-GPU probe failed: {e!r}")

    if not profile.gpus:
        profile.notes.append("No dedicated GPU detected; the model will run on CPU and use system RAM.")


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def _detect_disk(profile: HardwareProfile) -> None:
    target = os.path.expanduser("~")
    try:
        usage = shutil.disk_usage(target)
        profile.disk_free_gb = round(usage.free / 1024 / 1024 / 1024, 2)
        profile.disk_total_gb = round(usage.total / 1024 / 1024 / 1024, 2)
    except Exception as e:
        profile.notes.append(f"Disk-space detection failed: {e!r}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """Top-level detection. NEVER raises. Always returns a HardwareProfile
    (possibly sparse) plus a `notes` list describing what couldn't be
    detected and why."""
    profile = HardwareProfile()
    try:
        _detect_os(profile)
    except Exception as e:
        profile.notes.append(f"OS detection crashed: {e!r}")
    try:
        _detect_cpu(profile)
    except Exception as e:
        profile.notes.append(f"CPU detection crashed: {e!r}")
    try:
        _detect_ram(profile)
    except Exception as e:
        profile.notes.append(f"RAM detection crashed: {e!r}")
    try:
        _detect_gpus(profile)
    except Exception as e:
        profile.notes.append(f"GPU detection crashed: {e!r}")
    try:
        _detect_disk(profile)
    except Exception as e:
        profile.notes.append(f"Disk detection crashed: {e!r}")
    return profile


if __name__ == "__main__":
    # Manual sanity-check when run as a script.
    import json
    p = detect_hardware()
    print(json.dumps(p.to_dict(), indent=2, default=str))
    sys.exit(0)
