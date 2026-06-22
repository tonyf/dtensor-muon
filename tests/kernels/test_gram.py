from __future__ import annotations

import os
from typing import Annotated

import torch
from dtensor_muon.kernels.gram import gram, gram_
from torch import Tensor

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")


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


if __name__ == "__main__":
    from typer import Typer

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
        from helion._testing import DEVICE, run_example

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
