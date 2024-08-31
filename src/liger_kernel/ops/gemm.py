import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous

@triton.jit
def _gemm_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    alpha,
    beta,
    out_ptr,
    BLOCK_DIM: tl.constexpr = 16,
):
    """
    A(m, k) x B(k, n) + C(m, n) -> (m, n)
    """

    program_id = tl.program_id(0).to(tl.int64)

    block_row = program_id // tl.cdiv(n, BLOCK_DIM)
    block_col = program_id % tl.cdiv(n, BLOCK_DIM)

    offs_m = tl.arange(0, BLOCK_DIM)[:, None]
    offs_n = tl.arange(0, BLOCK_DIM)[None, :]
    offs_k = tl.arange(0, BLOCK_DIM)

    accumulator = tl.zeros((BLOCK_DIM, BLOCK_DIM), dtype=tl.float32)
    for kk in range(0, k, BLOCK_DIM):
        a_ptrs = a_ptr + (block_row * BLOCK_DIM + offs_m) * k + (kk + offs_k)
        b_ptrs = b_ptr + (offs_m + kk) * n + (offs_k + block_col * BLOCK_DIM)

        a = tl.load(a_ptrs, mask=offs_m + block_row * BLOCK_DIM < m, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_n + block_col * BLOCK_DIM < n, other=0.0).to(tl.float32)

        accumulator = tl.dot(a, b, accumulator)

    c_ptrs = (
        c_ptr + (block_row * BLOCK_DIM + offs_m) * n + (block_col * BLOCK_DIM + offs_n)
    )
    o_ptrs = (
        out_ptr
        + (block_row * BLOCK_DIM + offs_m) * n
        + (block_col * BLOCK_DIM + offs_n)
    )

    mask_m = (block_row * BLOCK_DIM + offs_m) < m
    mask_n = (block_col * BLOCK_DIM + offs_n) < n
    mask = mask_m & mask_n

    if beta != 0.0:
        c = tl.load(c_ptrs, mask=mask, other=0.0)
        o = accumulator * alpha + c * beta
    else:
        o = accumulator * alpha

    tl.store(o_ptrs, o, mask=mask)


def _gemm_forward(
    a: tl.tensor,
    b: tl.tensor,
    c: tl.tensor,
    alpha: float,
    beta: float,
    out: tl.tensor,
    BLOCK_DIM: int = 16,
):
    m, k = a.shape
    k_, n = b.shape
    assert k == k_

    grid = (triton.cdiv(m, BLOCK_DIM) * triton.cdiv(n, BLOCK_DIM),)

    _gemm_forward_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        alpha,
        beta,
        out,
        BLOCK_DIM,
    )


@triton.jit
def _mul_with_scalar_kernel(
    a_ptr,
    scalar,
    out_ptr,
    m,
    n,
    BLOCK_DIM: tl.constexpr = 8,  # cache line size
):
    program_id = tl.program_id(0).to(tl.int64)

    block_row = program_id // tl.cdiv(n, BLOCK_DIM)
    block_col = program_id % tl.cdiv(n, BLOCK_DIM)

    offs_k = tl.arange(0, BLOCK_DIM)

    a_ptrs = a_ptr + block_row * n + block_col * BLOCK_DIM + offs_k
    a = tl.load(a_ptrs, mask=offs_k < n, other=0.0)

    o_ptrs = out_ptr + block_row * n + block_col * BLOCK_DIM + offs_k
    tl.store(o_ptrs, a * scalar, mask=offs_k < n)


def _mul_with_scalar(
    a: tl.tensor,
    scalar: float,
    out: tl.tensor,
    BLOCK_DIM: int = 8,
):
    m, n = a.shape

    grid = (m * triton.cdiv(n, BLOCK_DIM),)

    _mul_with_scalar_kernel[grid](
        a,
        scalar,
        out,
        m,
        n,
        BLOCK_DIM,
    )


def _gemm_backward(
    a: tl.tensor,
    b: tl.tensor,
    c: tl.tensor,
    output_grad: tl.tensor,
    alpha: float,
    beta: float,
    a_requires_grad: bool,
    b_requires_grad: bool,
    c_requires_grad: bool,
):
    m, k = a.shape
    k_, n = b.shape
    assert k == k_

    a_grad: tl.tensor | None = None
    b_grad: tl.tensor | None = None
    c_grad: tl.tensor | None = None

    zeros = torch.zeros_like(c)

    if a_requires_grad:
        a_grad = torch.zeros_like(a)
        _gemm_forward(
            output_grad,
            b.T.contiguous(),
            zeros,
            alpha,
            0.0,
            a_grad,
        )

    if b_requires_grad:
        b_grad = torch.zeros_like(b)
        _gemm_forward(
            a.T.contiguous(),
            output_grad,
            zeros,
            alpha,
            0.0,
            b_grad,
        )

    if c_requires_grad:
        c_grad = torch.zeros_like(c)
        _mul_with_scalar(
            output_grad,
            beta,
            c_grad,
        )

    return a_grad, b_grad, c_grad


class AddmmFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None = None,
        alpha=1.0,
        beta=1.0,
    ):
        y = torch.zeros(x.shape[0], w.shape[1], device=x.device)

        _gemm_forward(
            x,
            w,
            b,
            alpha,
            beta,
            y,
        )
        ctx.save_for_backward(x, w, b)
        ctx.alpha = alpha
        ctx.beta = beta
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        x, w, b = ctx.saved_tensors
        a_grad, b_grad, c_grad = _gemm_backward(
            x,
            w,
            b,
            dY,
            ctx.alpha,
            ctx.beta,
            x.requires_grad,
            w.requires_grad,
            b.requires_grad,
        )
        return a_grad, b_grad, c_grad, None, None
