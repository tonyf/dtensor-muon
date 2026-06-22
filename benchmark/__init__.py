"""Benchmark suite for dtensor-muon.

Run everything and regenerate ``benchmark/RESULTS.md``::

    uv run python benchmark/run.py

The individual modules (``bench_kernels``, ``bench_optim``, ``bench_distributed``) each
expose a ``run(quick: bool) -> str`` returning a markdown section.
"""
