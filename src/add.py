import torch
import triton
import triton.language as tl


@triton.jit
def naive_vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(pointer=output_ptr + offsets, value=output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.zeros_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    naive_vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=8192)

    return output


if __name__ == "__main__":
    x, y = torch.randn((2, 204_800), device="cuda")
    print(add(x, y))
    print(f"Max absolute difference: {torch.max(torch.abs((x + y) - add(x, y)))}")
