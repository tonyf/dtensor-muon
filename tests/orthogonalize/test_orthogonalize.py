"""Correctness tests for the orthogonalization dispatch layer.

Exercises the public ``zeropower`` / ``foreach_zeropower`` entry points on plain
(non-DTensor) tensors, checking that the Triton path (``use_triton=True``)
matches the repository's PyTorch iteration run *uncompiled* (the eager original,
used as a stable, compile-independent ground truth). The distributed DTensor
paths need a process group and are not covered here.
"""

from typing import Any, Callable, cast

import pytest
import torch
from testkit import assert_close, run_example

from muonium.orthogonalize import OrthogonalizationStrategy
from muonium.orthogonalize.newton_schulz import ns_loop, ns_loop_triton
from muonium.orthogonalize.orthogonalize import (
    _get_orthogonalization_fn,
    foreach_zeropower,
    is_fsdp_3d_sharded,
    zeropower,
)
from muonium.orthogonalize.polar_express import pe_loop, pe_loop_triton

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

STRATEGIES: tuple[OrthogonalizationStrategy, ...] = ("newton_schulz", "polar_express")
# zeropower / foreach_zeropower defaults.
STEPS = 5
EPS = 1e-7

RTOL = 1e-2
ATOL = 2e-2
MAX_MISMATCH_PCT = 0.5
PE_FOREACH_MISMATCH_PCT = 5.0


def eager(fn: Callable) -> Callable:
    """The original, uncompiled function behind a ``@torch.compile`` wrapper."""
    return getattr(fn, "__wrapped__", fn)


# Uncompiled iteration matching what zeropower dispatches to internally; the eager
# original is used as a stable, compile-independent ground truth.
_REF = {"newton_schulz": ns_loop, "polar_express": pe_loop}


def test_get_orthogonalization_fn_dispatch_table() -> None:
    # The use_triton=False branch returns the compiled in-repo loop directly (the
    # 2D miscompile that previously forced an eager unwrap has been fixed).
    assert _get_orthogonalization_fn("newton_schulz", True) is ns_loop_triton
    assert _get_orthogonalization_fn("newton_schulz", False) is ns_loop
    assert _get_orthogonalization_fn("polar_express", True) is pe_loop_triton
    assert _get_orthogonalization_fn("polar_express", False) is pe_loop


def test_unknown_strategy_raises() -> None:
    G = torch.randn(16, 8)

    with pytest.raises(ValueError, match="Unknown orthogonalization strategy"):
        _get_orthogonalization_fn(cast(Any, "bogus"), False)
    with pytest.raises(ValueError, match="Unknown orthogonalization strategy"):
        zeropower(G, use_triton=False, strategy=cast(Any, "bogus"))
    with pytest.raises(ValueError, match="Unknown orthogonalization strategy"):
        foreach_zeropower([G], use_triton=False, strategy=cast(Any, "bogus"))


def test_zeropower_polar_express_rejects_more_than_five_steps() -> None:
    G = torch.randn(16, 8)

    with pytest.raises(
        AssertionError,
        match="only supports up to 5 optimization steps",
    ):
        zeropower(G, steps=6, use_triton=False, strategy="polar_express")


@requires_cuda
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("shape", [(256, 128), (128, 256), (512, 512)])
def test_zeropower_triton_matches_torch(
    strategy: OrthogonalizationStrategy, shape: tuple[int, int]
) -> None:
    torch.manual_seed(0)
    G = torch.randn(*shape, device="cuda")
    ref = eager(_REF[strategy])
    run_example(
        lambda g: zeropower(g, use_triton=True, strategy=strategy),
        lambda g: ref(g.bfloat16(), STEPS, eps=EPS),
        (G,),
        kernel_name="triton",
        baseline_name="torch",
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
        benchmark=False,
    )


@requires_cuda
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_zeropower_non_triton_dispatch_matches_triton(
    strategy: OrthogonalizationStrategy,
) -> None:
    torch.manual_seed(4)
    G = torch.randn(128, 64, device="cuda")

    out = zeropower(G, use_triton=False, strategy=strategy)
    triton = zeropower(G, use_triton=True, strategy=strategy)

    assert_close(
        out.float(),
        triton.float(),
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
    )


def test_zeropower_non_triton_polar_express_correct_on_2d() -> None:
    # The non-Triton path dispatches to the compiled pe_loop (the 2D miscompile is
    # fixed), so on a bare 2D matrix it matches the eager reference within bf16
    # tolerance — it is no longer bit-identical, since compile reorders fp ops.
    torch.manual_seed(0)
    G = torch.randn(64, 32)

    out = zeropower(G, use_triton=False, strategy="polar_express")
    ref = eager(pe_loop)(G.bfloat16(), STEPS, eps=EPS)

    assert_close(
        out.float(), ref.float(), rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_foreach_zeropower_returns_same_length_and_order(
    strategy: OrthogonalizationStrategy,
) -> None:
    torch.manual_seed(1)
    Gs = [torch.randn(32, 16) + i for i in range(3)]

    outs = foreach_zeropower(Gs, use_triton=False, strategy=strategy)

    assert isinstance(outs, list)
    assert len(outs) == len(Gs)
    # Each output corresponds (in order) to its own input. foreach stacks into a
    # batched (3D) compiled graph while single zeropower uses a 2D graph, so the
    # match is within bf16 tolerance, not bit-exact.
    for out, g in zip(outs, Gs, strict=True):
        ref = zeropower(g, use_triton=False, strategy=strategy)
        assert_close(
            out.float(),
            ref.float(),
            rtol=RTOL,
            atol=ATOL,
            max_mismatch_pct=(
                PE_FOREACH_MISMATCH_PCT
                if strategy == "polar_express"
                else MAX_MISMATCH_PCT
            ),
        )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_foreach_zeropower_single_element_matches_zeropower(
    strategy: OrthogonalizationStrategy,
) -> None:
    torch.manual_seed(2)
    G = torch.randn(32, 16)

    out = foreach_zeropower([G], use_triton=False, strategy=strategy)[0]
    ref = zeropower(G, use_triton=False, strategy=strategy)

    # Batched (3D) vs single (2D) compiled graphs differ at the bf16 level.
    assert_close(
        out.float(),
        ref.float(),
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=(
            PE_FOREACH_MISMATCH_PCT if strategy == "polar_express" else MAX_MISMATCH_PCT
        ),
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_zeropower_returns_bfloat16(strategy: OrthogonalizationStrategy) -> None:
    G = torch.randn(32, 16, dtype=torch.float32)

    out = zeropower(G, use_triton=False, strategy=strategy)

    assert out.dtype is torch.bfloat16


def test_foreach_zeropower_mixed_shapes_raise() -> None:
    with pytest.raises(RuntimeError, match="stack"):
        foreach_zeropower(
            [torch.randn(32, 16), torch.randn(16, 32)],
            use_triton=False,
        )


def test_foreach_zeropower_empty_list_raises() -> None:
    with pytest.raises(IndexError):
        foreach_zeropower([], use_triton=False)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_foreach_zeropower_plain_3d_inputs(strategy: OrthogonalizationStrategy) -> None:
    torch.manual_seed(3)
    Gs = [torch.randn(2, 32, 16) + i for i in range(3)]

    outs = foreach_zeropower(Gs, use_triton=False, strategy=strategy)

    assert len(outs) == len(Gs)
    for out, g in zip(outs, Gs, strict=True):
        ref = zeropower(g, use_triton=False, strategy=strategy)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_is_fsdp_3d_sharded_rejects_plain_tensors() -> None:
    plain = torch.randn(2, 32, 16)

    assert not is_fsdp_3d_sharded([plain])


@requires_cuda
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_foreach_zeropower_triton_matches_torch(strategy: OrthogonalizationStrategy) -> None:
    torch.manual_seed(0)
    Gs = [torch.randn(256, 128, device="cuda") for _ in range(4)]
    ref = eager(_REF[strategy])

    def baseline(gs: list[torch.Tensor]) -> list[torch.Tensor]:
        stacked = torch.stack(gs).bfloat16()
        return list(ref(stacked, steps=STEPS, eps=EPS).unbind(0))

    run_example(
        lambda gs: cast(
            list[torch.Tensor],
            foreach_zeropower(gs, use_triton=True, strategy=strategy),
        ),
        baseline,
        (Gs,),
        kernel_name="triton",
        baseline_name="torch",
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
        benchmark=False,
    )
