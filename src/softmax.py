import torch
import triton
import triton.language as tl


@triton.jit
def softmax_fwd_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < (pid + 1) * n_cols

    x = tl.load(x_ptr + row_offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    x /= tl.sum(x, axis=0)

    tl.store(pointer=out_ptr + row_offsets, value=x, mask=mask)


@triton.jit
def softmax_bwd_kernel(
    x_ptr,  # softmax outputs
    dy_ptr,  # upstream gradients
    dx_ptr,  # logits gradients
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < (pid + 1) * n_cols

    dy = tl.load(dy_ptr + row_offsets, mask=mask, other=0.0)
    y = tl.load(x_ptr + row_offsets, mask=mask, other=0.0)

    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)

    tl.store(pointer=dx_ptr + row_offsets, value=dx, mask=mask)


def softmax_fwd(ctx, x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    x_rows, x_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(x_cols)
    grid = lambda meta: (x_rows,)

    softmax_fwd_kernel[grid](x, output, x_cols, BLOCK_SIZE=BLOCK_SIZE)

    return output


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        output = torch.empty_like(x)
        x_rows, x_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(x_cols)
        grid = lambda meta: (x_rows,)

        softmax_fwd_kernel[grid](x, output, x_cols, BLOCK_SIZE=BLOCK_SIZE)

        ctx.save_for_backward(output)  # cache to use in backward
        return output

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x = ctx.saved_tensors[0]  # read softmax outputs
        dx = torch.empty_like(x)
        x_rows, x_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(x_cols)
        grid = lambda meta: (x_rows,)

        softmax_bwd_kernel[grid](x, dy, dx, x_cols, BLOCK_SIZE=BLOCK_SIZE)

        return dx
