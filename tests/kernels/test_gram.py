import os
from typing import Annotated

import pytest
import torch
from testkit import run_example
from torch import Tensor

from muonium.kernels.gram import gram, gram_

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 CUDA devices"
)


@torch.compile(fullgraph=True)
def gram_ref(
    x: Annotated[Tensor, "... M K"],
) -> Annotated[Tensor, "... M M"]:
    return x @ x.mT


def gram_inplace_wrapper(
    x: Annotated[Tensor, "... M K"],
) -> Annotated[Tensor, "... M M"]:
    M = x.size(-2)
    out = torch.empty(*x.shape[:-2], M, M, device=x.device, dtype=x.dtype)
    gram_(x, out)
    return out


# Plain (uncompiled) reference for the correctness checks below.
def _gram_eager(x: Tensor) -> Tensor:
    return x @ x.mT


# (rtol, atol, max_mismatch_pct). The kernel's tl.dot uses TF32 for fp32 inputs
# (matmul precision is "high" above), so fp32 needs the same ballpark tolerance
# as bf16, with a tiny mismatch budget for near-zero entries.
_TOL = {
    torch.float32: (2e-2, 2e-2, 0.1),
    torch.bfloat16: (1e-2, 1e-2, 0.1),
}


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (128, 64),  # 2D, no batch
        (32, 128, 64),  # 3D batched
        (2, 32, 128, 64),  # 4D batched
        (32, 128, 256),  # wide (K > M)
        (32, 256, 64),  # tall (M > K)
        (4, 300, 70),  # multiple M-blocks, non-multiple M/K
        (3, 130, 70),  # mask-heavy partial M/K blocks
        (1, 1, 8),  # M=1 degenerate batch
        (4, 1, 1),  # M=K=1 degenerate batch
        (4, 64, 1024),  # many K-loop blocks
    ],
)
def test_gram_matches_reference(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    rtol, atol, mm_pct = _TOL[dtype]
    if dtype is torch.float32 and shape[-1] >= 1024:
        mm_pct = 0.3
    x = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    run_example(
        gram,
        _gram_eager,
        (x,),
        kernel_name="gram_triton",
        baseline_name="gram_ref",
        rtol=rtol,
        atol=atol,
        max_mismatch_pct=mm_pct,
        benchmark=False,
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gram_output_is_exactly_symmetric(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    x = torch.randn(4, 130, 70, device="cuda", dtype=dtype)

    y = gram(x)

    torch.testing.assert_close(y, y.mT, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(128, 64), (32, 128, 64), (2, 32, 128, 64)])
def test_gram_inplace_matches_reference(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    rtol, atol, mm_pct = _TOL[dtype]
    x = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    run_example(
        gram_inplace_wrapper,
        _gram_eager,
        (x,),
        kernel_name="gram_inplace_triton",
        baseline_name="gram_ref",
        rtol=rtol,
        atol=atol,
        max_mismatch_pct=mm_pct,
        benchmark=False,
    )


@requires_cuda
def test_gram_inplace_accepts_non_contiguous_input() -> None:
    base = torch.randn(4, 70, 130, device="cuda", dtype=torch.float32)
    x = base.transpose(-2, -1)
    assert not x.is_contiguous()
    out = torch.empty(4, 130, 130, device="cuda", dtype=x.dtype)

    gram_(x, out)

    torch.testing.assert_close(out, _gram_eager(x), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_gram_inplace_rejects_non_contiguous_output() -> None:
    x = torch.randn(32, 16, device="cuda", dtype=torch.float32)
    out = torch.empty(32, 32, device="cuda", dtype=x.dtype).T
    assert not out.is_contiguous()

    with pytest.raises(AssertionError, match="d_out must be contiguous"):
        gram_(x, out)


@requires_cuda
def test_gram_inplace_rejects_contract_violations() -> None:
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    out = torch.empty(4, 8, 8, device="cuda", dtype=torch.float32)

    with pytest.raises(AssertionError):
        gram_(x.cpu(), out.cpu())
    with pytest.raises(AssertionError):
        gram_(x, out.cpu())
    with pytest.raises(AssertionError):
        gram_(x, out.to(torch.bfloat16))
    with pytest.raises(AssertionError):
        gram_(x, torch.empty(3, 8, 8, device="cuda", dtype=x.dtype))
    with pytest.raises(AssertionError):
        gram_(x, torch.empty(4, 8, 7, device="cuda", dtype=x.dtype))
    with pytest.raises(AssertionError):
        gram_(torch.randn(8, device="cuda"), torch.empty(8, 8, device="cuda"))


@requires_2_gpus
def test_gram_inplace_rejects_different_cuda_devices() -> None:
    x = torch.randn(4, 8, 16, device="cuda:0", dtype=torch.float32)
    out = torch.empty(4, 8, 8, device="cuda:1", dtype=torch.float32)

    with pytest.raises(AssertionError):
        gram_(x, out)


@requires_cuda
def test_gram_is_deterministic_for_same_input() -> None:
    torch.manual_seed(42)
    x = torch.randn(4, 130, 70, device="cuda", dtype=torch.float32)

    first = gram(x)
    second = gram(x)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@requires_cuda
def test_gram_zero_rows_produce_zero_rows_and_columns() -> None:
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    x[2].zero_()
    x[:, 3].zero_()

    y = gram(x)

    torch.testing.assert_close(y[2], torch.zeros_like(y[2]), rtol=0, atol=0)
    torch.testing.assert_close(y[:, 3], torch.zeros_like(y[:, 3]), rtol=0, atol=0)
    torch.testing.assert_close(y[:, :, 3], torch.zeros_like(y[:, :, 3]), rtol=0, atol=0)


if __name__ == "__main__":
    from typer import Typer  # ty: ignore[unresolved-import]

    cli = Typer(pretty_exceptions_show_locals=False)

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    @cli.command()
    def test(
        batch_size: int = 32,
        m: int = 128,
        k: int = 64,
        dtype: str = "bfloat16",
        no_cache: bool = False,
    ) -> None:
        from helion._testing import DEVICE, run_example  # ty: ignore[unresolved-import]

        torch.manual_seed(42)
        torch.set_float32_matmul_precision("high")
        device = DEVICE

        if no_cache:
            os.environ["HELION_SKIP_CACHE"] = "1"

        assert dtype in ["bfloat16", "float32"], "dtype must be bfloat16 or float32"
        if dtype == "bfloat16":
            _dtype = torch.bfloat16
        else:
            _dtype = torch.float32

        # --- Test gram (allocating) ---
        print(f"Testing gram with shape ({batch_size}, {m}, {k}), dtype={dtype}")
        x = torch.randn(batch_size, m, k, device=device, dtype=_dtype).contiguous()
        run_example(
            gram,
            gram_ref,
            (x,),
            kernel_name="gram_triton",
            baseline_name="gram_ref",
            rtol=1e-2,
            atol=1e-2,
            bwd=False,
        )
        print("  gram: PASSED")

        # --- Test gram_ (in-place) ---
        print(f"Testing gram_ with shape ({batch_size}, {m}, {k}), dtype={dtype}")
        x = torch.randn(batch_size, m, k, device=device, dtype=_dtype).contiguous()
        run_example(
            gram_inplace_wrapper,
            gram_ref,
            (x,),
            kernel_name="gram_inplace_triton",
            baseline_name="gram_ref",
            rtol=1e-2,
            atol=1e-2,
            bwd=False,
        )
        print("  gram_: PASSED")

        # --- Test higher-dimensional batch ---
        print(f"Testing gram with shape (2, {batch_size}, {m}, {k}), dtype={dtype}")
        x = torch.randn(2, batch_size, m, k, device=device, dtype=_dtype).contiguous()
        run_example(
            gram,
            gram_ref,
            (x,),
            kernel_name="gram_triton_4d",
            baseline_name="gram_ref_4d",
            rtol=1e-2,
            atol=1e-2,
            bwd=False,
        )
        print("  gram (4D): PASSED")

        # --- Test 2D (no batch) ---
        print(f"Testing gram with shape ({m}, {k}), dtype={dtype}")
        x = torch.randn(m, k, device=device, dtype=_dtype).contiguous()
        run_example(
            gram,
            gram_ref,
            (x,),
            kernel_name="gram_triton_2d",
            baseline_name="gram_ref_2d",
            rtol=1e-2,
            atol=1e-2,
            bwd=False,
        )
        print("  gram (2D): PASSED")

        # --- Test non-square (M != K) with larger K ---
        print(f"Testing gram with shape ({batch_size}, {m}, {k * 4}), dtype={dtype}")
        x = torch.randn(batch_size, m, k * 4, device=device, dtype=_dtype).contiguous()
        run_example(
            gram,
            gram_ref,
            (x,),
            kernel_name="gram_triton_wide",
            baseline_name="gram_ref_wide",
            rtol=1e-2,
            atol=1e-2,
            bwd=False,
        )
        print("  gram (wide): PASSED")

        print("\nAll tests passed!")

    cli()
