"""Hide GPUs this build cannot use from CUDA, before torch initializes.

torch cu12.8+/cu13 wheels ship kernels for sm_75+ (Turing) only.  Worse,
cuDNN >= 9.11 refuses to run *at all* in a process that can enumerate an
older GPU -- even when every tensor lives on a newer card: the first
``nn.GRU`` (our encoder) aborts with

    cuDNN version ... is not compatible with devices with SM < 7.5

Boxes that keep an old card for displays (e.g. a GTX 950 at device 1
next to a CMP 50HX at device 0) are therefore broken by default.  This
module filters ``CUDA_VISIBLE_DEVICES`` to sm_75+ GPUs at package import
time, before any CUDA context exists:

* ``CUDA_VISIBLE_DEVICES`` already set in the environment -> respected;
* ``TILECHAT_DISABLE_GPU_FILTER=1`` -> disables the filter;
* no NVIDIA tooling/driver visible -> no-op (torch reports its own state).
"""

from __future__ import annotations

import os
import subprocess

# sm_75 = Turing: oldest architecture supported by the torch/cuDNN build
# this project targets (and by our TileLang kernels).
MIN_COMPUTE_CAP = 75


def _notice_and_keep(good: list[str], hidden: list[str]) -> tuple[str, str]:
    keep = [g.split(" ", 1)[0] for g in good]
    return (
        "[tilechat] hiding CUDA device(s) " + ", ".join(hidden) + ": sub-sm_75 GPUs "
        "are unsupported by this torch/cuDNN build (cuDNN >= 9.11 fails in any "
        "process that can see one), so they are removed from CUDA_VISIBLE_DEVICES.",
        ",".join(keep),
    )


def filter_cuda_devices(min_cap: int = MIN_COMPUTE_CAP) -> str | None:
    """Restrict ``CUDA_VISIBLE_DEVICES`` to GPUs with compute cap >= ``min_cap``.

    Must run before the first CUDA call (``import torch`` is fine, creating
    tensors on CUDA is not).  Returns the printed notice when devices were
    hidden, else ``None``.
    """
    if os.environ.get("TILECHAT_DISABLE_GPU_FILTER") == "1":
        return None
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None  # caller manages visibility themselves
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None  # no driver/tooling: nothing to filter

    good: list[str] = []
    hidden: list[str] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, name, cc = parts
        try:
            major_s, minor_s = cc.split(".")
            cap = int(major_s) * 10 + int(minor_s)
        except ValueError:
            continue
        entry = f"{idx} ({name}, sm_{major_s}{minor_s})"
        (good if cap >= min_cap else hidden).append(entry)

    if not good or not hidden:
        return None  # nothing to hide, or nothing usable anyway

    notice, keep = _notice_and_keep(good, hidden)
    os.environ["CUDA_VISIBLE_DEVICES"] = keep
    print(notice, flush=True)
    return notice
