import torch
import triton
import triton.language as tl


@triton.jit
def naive_matmul_kernel(
    x_ptr,
    y_t_ptr,  # the kernel gets Y.T
    out_ptr,
    x_cols,
    y_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # --- row and col id ---
    x_row_id = tl.program_id(axis=0)
    y_col_id = tl.program_id(axis=1)

    # --- start id, offsets and masks ---
    x_row_start = x_row_id * x_cols
    x_offsets = x_row_start + tl.arange(0, BLOCK_SIZE)
    x_mask = x_offsets < x_row_start + x_cols

    y_t_row_start = y_col_id * x_cols
    y_t_offsets = y_t_row_start + tl.arange(0, BLOCK_SIZE)
    y_mask = y_t_offsets < y_t_row_start + x_cols

    # --- load data ---
    x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
    y = tl.load(y_t_ptr + y_t_offsets, mask=y_mask, other=0.0)

    # --- compute output and offsets ---
    out = tl.sum(x * y, axis=0)
    out_offset = (x_row_id * y_cols) + y_col_id

    tl.store(pointer=out_ptr + out_offset, value=out)


def naive_matmul(X, Y):
    x_rows, x_cols = X.shape
    y_rows, y_cols = Y.shape

    out = torch.empty((x_rows, y_cols), device="cuda")
    BLOCK_SIZE = triton.next_power_of_2(x_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    Y = Y.T.contiguous()

    grid = lambda meta: (x_rows, y_cols)
    naive_matmul_kernel[grid](
        X,
        Y,
        out,
        x_cols,
        y_cols,
        BLOCK_SIZE=1024,
        num_warps=num_warps,
    )

    return out
