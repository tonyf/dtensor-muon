import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": blk_m, "BLOCK_SIZE_K": blk_k, "GROUP_SIZE_M": grp_sz},
            num_stages=n_stages,
            num_warps=n_warps,
        )
        for blk_m in [32, 64, 128]
        for blk_k in [32, 64]
        for grp_sz in [8]
        for n_stages in [3, 4, 5]
        for n_warps in [4, 8]
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "K"],
)
@triton.jit
def gram_nd(
    x,
    y,
    M,
    K,
    stride_xb,
    stride_xm,
    stride_xk,
    stride_yb,
    stride_ym,
    stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Computes (per batch): y[b] = x[b] @ x[b].T
    where x has shape (B, M, K) and y has shape (B, M, M).

    We only compute the upper triangle blocks and mirror them into the lower triangle.
    """
    # 2D launch grid:
    pid = tl.program_id(axis=0)  # tiles over (M, M)
    pid_b = tl.program_id(axis=1)  # batch index (flattened leading dims)

    # Move base pointers to this batch
    x = x + pid_b * stride_xb
    y = y + pid_b * stride_yb

    # --- same block mapping logic as your original kernel ---
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Only compute upper triangle (including diagonal)
    if pid_m > pid_n:
        return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    # K loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # mask only on K (M handled via %M wrap + store mask)
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)

        # b^T via permute
        acc = tl.dot(a, tl.permute(b, (1, 0)), acc)

        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk

    c = acc.to(x.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    # transpose+copy to lower triangle block
    if pid_m < pid_n:
        ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def gram_(d_in: torch.Tensor, d_out: torch.Tensor) -> None:
    """
    d_in:  (..., M, K)
    d_out: (..., M, M)
    """
    assert d_in.is_cuda and d_out.is_cuda
    assert d_in.device == d_out.device
    assert d_in.dtype == d_out.dtype
    assert d_in.ndim >= 2 and d_out.ndim >= 2
    assert d_in.shape[:-2] == d_out.shape[:-2]

    M, K = d_in.shape[-2], d_in.shape[-1]
    assert d_out.shape[-2:] == (M, M)

    # We need a view as (B, M, K) and (B, M, M). Output must be viewable (no copies).
    # Input we can safely make contiguous (if it was transposed etc).
    x = d_in.contiguous().view(-1, M, K)

    # For correctness, don't allow reshape() to silently copy the output.
    assert d_out.is_contiguous(), (
        "d_out must be contiguous (or you must write into a contiguous temp and copy_ back)"
    )
    y = d_out.view(-1, M, M)

    B = x.shape[0]

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_M"]),
        B,
    )

    with torch.cuda.device(d_in.device.index):
        gram_nd[grid](
            x,
            y,
            M,
            K,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
        )


def gram(d_in: torch.Tensor) -> torch.Tensor:
    """
    Returns y = d_in @ d_in^T over the last 2 dims.
    d_in: (..., M, K) -> y: (..., M, M)
    """
    assert d_in.ndim >= 2
    M = d_in.size(-2)
    out_shape = (*d_in.shape[:-2], M, M)
    d_out = torch.empty(out_shape, device=d_in.device, dtype=d_in.dtype)
    gram_(d_in, d_out)
    return d_out
