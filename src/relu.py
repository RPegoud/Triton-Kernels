import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input + offsets, mask=mask)
    y = tl.maximum(x, tl.zeros_like(x))

    tl.store(pointer=output + offsets, value=y, mask=mask)


def relu(x: torch.Tensor):
    output = torch.zeros_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=512)

    return output


if __name__ == "__main__":
    x = torch.tensor([-1.0, 2.0, 3.0, 0.0], device="cuda")
    print(f"Input: {x}")
    print(f"Output: {relu(x)}")
