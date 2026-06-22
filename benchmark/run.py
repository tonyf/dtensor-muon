"""Run the full benchmark suite and write ``benchmark/RESULTS.md``.

Usage::

    uv run python benchmark/run.py            # full run
    uv run python benchmark/run.py --quick    # small shapes, fast smoke
    uv run python benchmark/run.py --no-distributed

Each section degrades gracefully: CUDA-only sections print a "skipped" note on a
CPU-only host, and the distributed section falls back from NCCL/CUDA to gloo/CPU.
"""

import argparse
import sys
from pathlib import Path

# Make ``benchmark`` importable whether invoked as ``python benchmark/run.py`` or
# ``python -m benchmark.run`` (and so spawned children inherit a usable sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark import bench_distributed, bench_kernels, bench_optim  # noqa: E402
from benchmark.harness import capture_env, write_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="small shapes for a fast smoke run"
    )
    parser.add_argument(
        "--no-distributed", action="store_true", help="skip the distributed section"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "benchmark" / "RESULTS.md",
        help="output markdown path",
    )
    args = parser.parse_args()

    env = capture_env()
    sections: list[tuple[str, str]] = []

    print("== kernels ==", file=sys.stderr)
    sections.append(("Triton kernels", bench_kernels.run(args.quick)))

    print("== single-device optimizers ==", file=sys.stderr)
    sections.append(("Single-device optimizers", bench_optim.run(args.quick)))

    if not args.no_distributed:
        print("== distributed ==", file=sys.stderr)
        sections.append(("Distributed orthogonalization", bench_distributed.run(args.quick)))

    write_results(args.out, sections, env)


if __name__ == "__main__":
    main()
