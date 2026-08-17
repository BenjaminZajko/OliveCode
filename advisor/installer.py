"""Auto-installation of the recommended local model via Ollama.

This is the ONLY module in the advisor that is allowed to shell out.
Hardware / catalog / reasoning / CLI stay pure logic with no subprocess
calls. Keeping the surface tight here means a failure in this module
can never leak back into the recommendation pipeline.

Public contract:

  - Every function in this module NEVER raises. On any failure it
    returns a structured InstallResult so the CLI can tell the user
    what to do next without crashing or printing a traceback.

  - We do NOT attempt to install Ollama itself. Ollama requires a real
    installer (different per OS) and admin permissions on some systems.
    If Ollama is missing, we print clear, OS-specific instructions and
    stop.

  - The download itself is `ollama pull <model_id>`, run with stdout
    streamed to the terminal so the user sees ollama's own progress
    bar (a multi-GB download shouldn't look like a silent hang). We
    capture stderr separately so we can recognise the specific
    "could not connect to ollama" pattern and tell the user to start
    the Ollama app first.

  - We do NOT try to reimplement resumability. If ollama pull fails
    mid-download, the user can just rerun `python3 -m advisor --install`.

Failure modes we recognise and label clearly:

  - `ollama_missing`           ollama binary not on PATH
  - `already_pulled`           the requested model is already local
  - `pulled`                   the download completed successfully
  - `pull_cancelled`           the user pressed Ctrl+C
  - `daemon_not_running`       ollama is installed but the daemon isn't running
  - `failed`                   anything else (network, disk full, etc.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InstallResult:
    ok: bool
    kind: str                  # one of the *_KINDS values below
    message: str               # human-readable, ready to print
    model_id: Optional[str] = None
    detail: str = ""           # raw stderr or extra context for debugging


OK_KINDS = {"already_pulled", "pulled"}
FAIL_KINDS = {"ollama_missing", "pull_cancelled", "daemon_not_running", "failed"}


# ---------------------------------------------------------------------------
# Ollama detection
# ---------------------------------------------------------------------------

def find_ollama_binary() -> Optional[str]:
    """Locate the `ollama` binary on PATH. Wrapped to never raise.

    On Windows we also check the well-known default install location
    because the .exe installer doesn't always put ollama on PATH for
    the current shell straight away.
    """
    try:
        found = shutil.which("ollama")
        if found:
            return found
        if os.name == "nt":
            # Common install paths. We don't try to be exhaustive.
            for candidate in (
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
            ):
                try:
                    if candidate and os.path.isfile(candidate):
                        return candidate
                except Exception:
                    continue
        return None
    except Exception:
        return None


def ollama_install_instructions(os_name: Optional[str]) -> str:
    """Return the per-OS instructions we'll show when Ollama isn't installed.

    Keep the wording short, specific, and copy-paste-friendly. The whole
    point is to remove the "now go install X manually" friction; the
    command line is the entire job.
    """
    name = (os_name or "").lower()
    if name == "macos":
        return (
            "Ollama isn't installed yet. Install it with one of these:\n"
            "\n"
            "  Option 1 (recommended):\n"
            "    brew install ollama\n"
            "    open -a Ollama\n"
            "\n"
            "  Option 2 (no Homebrew):\n"
            "    Download the .dmg from https://ollama.com/download and install it.\n"
            "    Open the Ollama app once so the daemon starts.\n"
            "\n"
            "Once Ollama is running, come back and run:\n"
            "    python3 -m advisor --install\n"
        )
    if name == "windows":
        return (
            "Ollama isn't installed yet. Install it with one of these:\n"
            "\n"
            "  Option 1 (recommended):\n"
            "    Download OllamaSetup.exe from https://ollama.com/download\n"
            "    and run it. (It puts ollama on your PATH automatically.)\n"
            "\n"
            "  Option 2 (winget):\n"
            "    winget install Ollama.Ollama\n"
            "\n"
            "Once the install finishes, Ollama's server starts automatically.\n"
            "Then come back and run:\n"
            "    python3 -m advisor --install\n"
        )
    if name == "linux":
        return (
            "Ollama isn't installed yet. Install it with one of these:\n"
            "\n"
            "  Option 1 (recommended, official):\n"
            "    curl -fsSL https://ollama.com/install.sh | sh\n"
            "\n"
            "  Option 2 (Debian/Ubuntu via apt):\n"
            "    sudo apt install Ollama\n"
            "\n"
            "After installing, make sure the daemon is running:\n"
            "    sudo systemctl enable --now ollama\n"
            "\n"
            "Then come back and run:\n"
            "    python3 -m advisor --install\n"
        )
    return (
        "Ollama isn't installed yet. Download it from https://ollama.com/download\n"
        "and install it for your operating system, then run:\n"
        "    python3 -m advisor --install\n"
    )


# ---------------------------------------------------------------------------
# Pulled-model check
# ---------------------------------------------------------------------------

def is_model_pulled(model_id: str) -> bool:
    """Return True if `ollama list` reports `model_id` as a local model.

    Ollama's output format is simple: NAME column then ID, then SIZE,
    then MODIFIED. The NAME column sometimes has a ":tag" suffix and
    sometimes not. We match the first whitespace-separated token on each
    non-header line. Local models have a non-empty SIZE; the cloud
    placeholder models Ollama ships with show "-" or empty for SIZE.

    We never raise; on any parse error we return False.
    """
    binary = find_ollama_binary()
    if not binary:
        return False
    try:
        proc = subprocess.run(
            [binary, "list"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError, Exception):
        return False
    if proc.returncode != 0:
        return False
    target = model_id.strip().lower()
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        # NAME is the first whitespace-separated token.
        name = line.split()[0].strip().lower()
        if name == target:
            return True
        # tolerate "model:tag" vs caller passing "model"
        if target in (name, f"{name}:latest"):
            return True
    return False


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

# Substrings in stderr that indicate the daemon isn't running. We use
# small, exact-ish patterns so we don't false-positive on unrelated
# errors.
DAEMON_NOT_RUNNING_PATTERNS = (
    "could not connect",
    "connection refused",
    "connection error",
    "dial tcp",
    "error: ollama",
    "ollama daemon",
    "is ollama running",
    "ollama is not running",
)


def _looks_like_daemon_down(stderr: str) -> bool:
    s = (stderr or "").lower()
    return any(pat in s for pat in DAEMON_NOT_RUNNING_PATTERNS)


def _looks_like_disk_full(stderr: str) -> bool:
    s = (stderr or "").lower()
    return ("no space" in s) or ("disk full" in s) or ("enospc" in s)


def install_model(
    model_id: str,
    *,
    on_message: Optional[Callable[[str], None]] = None,
    printer=print,
) -> InstallResult:
    """Download a model via `ollama pull <model_id>`.

    Behaviour:
      - If Ollama is missing, returns `ollama_missing` immediately.
      - If the model is already pulled, returns `already_pulled` without
        touching the network.
      - Otherwise spawns `ollama pull` and streams its stdout to the
        terminal (via the `printer` callable, default `print`) so the
        user sees ollama's own progress bar.
      - Captures stderr separately so we can pattern-match the
        "daemon not running" failure and tell the user to start the
        Ollama app.
      - Ctrl+C is handled cleanly: the subprocess is terminated and we
        return `pull_cancelled`. We never leave a zombie process.
      - On any other failure (network, disk, bad tag), returns `failed`.

    The function NEVER raises. All failures become structured results.
    """
    # 0. Sanity check the model id. ollama accepts arbitrary tags, but
    # we don't want shell injection from a malicious Recommendation.
    if not model_id or not isinstance(model_id, str):
        return InstallResult(
            ok=False, kind="failed",
            message=f"Refusing to install invalid model id: {model_id!r}",
            model_id=str(model_id) if model_id else None,
        )

    # 1. Ollama must be present.
    binary = find_ollama_binary()
    if not binary:
        return InstallResult(
            ok=False, kind="ollama_missing",
            message="Ollama is not installed on this machine.",
            model_id=model_id,
        )

    # 2. Wrap the rest so a Ctrl+C (SIGINT) at ANY point — the pulled
    # check, spawning, or the wait below — becomes a clean cancel result
    # instead of an uncaught traceback. We terminate the child if one
    # was spawned.
    proc: Optional[subprocess.Popen] = None
    stderr_chunks: list[bytes] = []

    try:
        # 2a. Already pulled?
        try:
            if is_model_pulled(model_id):
                return InstallResult(
                    ok=True, kind="already_pulled",
                    message=f"{model_id} is already installed locally.",
                    model_id=model_id,
                )
        except Exception:
            # Defensive: an unexpected exception here just means we
            # proceed to the pull step.
            pass

        # 3. Announce the pull.
        printer(f"\nDownloading {model_id} via Ollama (this can take a few minutes for large models)...\n")
        printer("$ " + " ".join([binary, "pull", model_id]) + "\n")

        # 4. Spawn the pull. We run it in binary mode and pump BOTH streams
        # from background threads. Critical detail: ollama writes its progress
        # bar to stderr, not stdout. If we only drained stdout, the stderr
        # pipe buffer would fill, ollama would block forever, and a finished
        # download would look like a silent hang. So we stream stderr live to
        # the terminal (the user sees the real progress bar) while also
        # accumulating it for daemon-not-running / disk-full detection.
        try:
            proc = subprocess.Popen(
                [binary, "pull", model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except (OSError, ValueError) as e:
            return InstallResult(
                ok=False, kind="failed",
                message=f"Could not start Ollama: {e}",
                model_id=model_id,
                detail=str(e),
            )
        except Exception as e:
            return InstallResult(
                ok=False, kind="failed",
                message=f"Could not start Ollama: {e!r}",
                model_id=model_id,
                detail=repr(e),
            )

        # 5. Pump both streams to the terminal in background threads. We read
        # raw bytes (os.read) so the output flows as soon as it is written —
        # ollama's progress bar uses \r + ANSI cursor moves rather than \n, so
        # a line-oriented read would buffer everything until the end. If the
        # user cancels with Ctrl+C, the main thread terminates the child and
        # these threads simply see the pipes close.
        def _pump(stream: Optional[object]) -> None:
            """Forward raw bytes from `stream` to the terminal until EOF."""
            try:
                if stream is None:
                    return
                while True:
                    raw = os.read(stream.fileno(), 4096)  # type: ignore[union-attr]
                    if not raw:
                        break
                    printer(raw.decode("utf-8", errors="replace"), end="")
            except Exception:
                # Stream closed, child killed, or fd invalid. Not fatal.
                pass

        def _capture_stderr() -> None:
            try:
                if proc.stderr is None:
                    return
                while True:
                    raw = os.read(proc.stderr.fileno(), 4096)
                    if not raw:
                        break
                    stderr_chunks.append(raw)
                    printer(raw.decode("utf-8", errors="replace"), end="")
            except Exception:
                pass

        out_thread = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        err_thread = threading.Thread(target=_capture_stderr, daemon=True)
        out_thread.start()
        err_thread.start()

        # 6. Wait for the pull to finish.
        while proc.poll() is None:
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue

        # 7. Pull finished. Join the pump threads so any final output gets
        # forwarded, then assemble the captured stderr for error detection.
        out_thread.join(timeout=2)
        err_thread.join(timeout=2)
        try:
            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        except Exception:
            stderr_text = ""

        rc = proc.returncode
        if rc == 0:
            return InstallResult(
                ok=True, kind="pulled",
                message=f"Successfully downloaded {model_id}.",
                model_id=model_id,
            )

        # 8. Non-zero exit. Classify the failure.
        if _looks_like_daemon_down(stderr_text):
            return InstallResult(
                ok=False, kind="daemon_not_running",
                message=(
                    "Ollama is installed but the daemon isn't running. "
                    "Open the Ollama app (macOS/Windows) or run "
                    "`sudo systemctl start ollama` (Linux) and try again."
                ),
                model_id=model_id,
                detail=stderr_text.strip(),
            )
        if _looks_like_disk_full(stderr_text):
            return InstallResult(
                ok=False, kind="failed",
                message=(
                    "Download failed: the disk is full. Free up some space "
                    "and run `python3 -m advisor --install` again to resume."
                ),
                model_id=model_id,
                detail=stderr_text.strip(),
            )

        return InstallResult(
            ok=False, kind="failed",
            message=(
                "Download failed. The most common cause is a network drop — "
                "Ollama will resume automatically on retry. Run "
                "`python3 -m advisor --install` again to continue."
            ),
            model_id=model_id,
            detail=stderr_text.strip(),
        )

    # 9. Ctrl+C anywhere above lands here: terminate the child cleanly
    # (never leave a zombie) and report the cancel.
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            except Exception:
                pass
        try:
            err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        except Exception:
            err = ""
        printer()
        return InstallResult(
            ok=False, kind="pull_cancelled",
            message="Download cancelled.",
            model_id=model_id,
            detail=err.strip(),
        )


__all__ = [
    "InstallResult",
    "find_ollama_binary",
    "ollama_install_instructions",
    "is_model_pulled",
    "install_model",
]
