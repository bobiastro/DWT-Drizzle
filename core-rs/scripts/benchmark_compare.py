"""Benchmark the migrated PWS and Drizzle paths against their Python sources.

The benchmark extracts the self-contained functions from `core/pws.py` so the
measurement does not require importing astropy. The generated workload is
deterministic and matches `core-rs/crates/dwt-backend/examples/benchmark.rs`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from textwrap import dedent

import numpy as np

pws_path = Path(__file__).resolve().parents[2] / "core" / "pws.py"
source = pws_path.read_text(encoding="utf-8")


def extract_function(name: str) -> str:
    marker = f"def {name}("
    start = source.index(marker)
    end = source.find("\ndef ", start + len(marker))
    if end < 0:
        end = len(source)
    return dedent(source[start:end])


constants = {}
for line in source.splitlines():
    if line.startswith("REJ_K = "):
        constants["REJ_K"] = float(line.partition("=")[2].split("#", 1)[0])
    elif line.startswith("REJ_MAX_FRAC = "):
        constants["REJ_MAX_FRAC"] = eval(
            line.partition("=")[2].split("#", 1)[0].strip(),
            {"__builtins__": {}},
        )

namespace: dict[str, object] = {"np": np, "math": math, "os": os, **constants}
exec(extract_function("strip_weights"), namespace)
exec(extract_function("stack_strip"), namespace)
strip_weights = namespace["strip_weights"]
stack_strip = namespace["stack_strip"]


def make_workload(frames: int, width: int, height: int):
    y, x = np.mgrid[0:height, 0:width]
    data = np.empty((frames, height, width), dtype=np.float32)
    for index in range(frames):
        data[index] = (
            ((x * 17 + y * 31 + index * 43) % 257) * 0.01
            + index * 0.02
        )
    metrics = [
        (
            1.0 + (index % 7) * 0.05,
            1.2 + (index % 5) * 0.1,
            1.0 + (index % 3) * 0.1,
        )
        for index in range(frames)
    ]
    region = ((x * 37 + y * 17) % 1000) / 1000.0
    return data, metrics, region.astype(np.float32)


def python_metrics(data, metrics, region, repetitions):
    times = []
    rejection_times = []
    fwhm = np.asarray([metric[1] for metric in metrics])
    noise = np.asarray([metric[0] for metric in metrics])
    c = np.median(fwhm) / fwhm
    s = np.clip(np.median(noise) / noise, 1.0 / 3.0, 3.0)
    for _ in range(repetitions):
        start = time.perf_counter()
        weights = strip_weights(
            c,
            s,
            region,
            A=20.0,
            B=2.0,
            fp=0.0,
        )
        stack_strip(data, weights, region, sky=0.0, rej_k=0.0)
        times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        stack_strip(data, weights, region, sky=0.0)
        rejection_times.append((time.perf_counter() - start) * 1000.0)
    return times, rejection_times


def rust_metrics(repo: Path, frames: int, width: int, height: int, repetitions: int):
    command = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(repo / "core-rs" / "Cargo.toml"),
        "--example",
        "benchmark",
        "--",
        str(frames),
        str(width),
        str(height),
        str(repetitions),
    ]
    output = subprocess.check_output(command, text=True, cwd=tempfile.gettempdir())
    return json.loads(output.splitlines()[-1]), output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    data, metrics, region = make_workload(args.frames, args.width, args.height)
    py_times, py_rejection_times = python_metrics(
        data, metrics, region, args.repetitions
    )
    rust, rust_output = rust_metrics(
        repo, args.frames, args.width, args.height, args.repetitions
    )
    py_median = statistics.median(py_times)
    rust_median = statistics.median(rust["stack_ms"])
    py_rejection_median = statistics.median(py_rejection_times)
    rust_rejection_median = statistics.median(rust["rejection_stack_ms"])
    result = {
        "frames": args.frames,
        "shape": [args.height, args.width],
        "python_pws_ms": py_times,
        "python_pws_median_ms": py_median,
        "rust_pws_ms": rust["stack_ms"],
        "rust_pws_median_ms": rust_median,
        "pws_speedup": py_median / rust_median,
        "python_pws_rejection_ms": py_rejection_times,
        "python_pws_rejection_median_ms": py_rejection_median,
        "rust_pws_rejection_ms": rust["rejection_stack_ms"],
        "rust_pws_rejection_median_ms": rust_rejection_median,
        "pws_rejection_speedup": py_rejection_median / rust_rejection_median,
        "rust_weights_ms": rust["weights_ms"],
        "rust_drizzle_ms": rust["drizzle_ms"],
        "rust_output": rust_output,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
