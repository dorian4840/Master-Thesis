
import torch

class ClippedSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        # Save the input for use in backward pass
        ctx.save_for_backward(input)

        # Clip the input and forward the result
        output = input.clamp(min_val, max_val)
        print(min_val, max_val)

        return output


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors

        # Create a mask where the gradient should be passed through
        pass_through_mask = (input >= 0.0) & (input <= 0.1)

        # Only pass gradient where the input was within the clipping range
        grad_input = grad_output * pass_through_mask.float()

        return grad_input, None, None


# Example usage
if __name__ == "__main__":
    min_val = -1.0
    max_val = 1.0
    input_tensor = torch.tensor([1.5, -0.5, 0.5, -1.5], requires_grad=True)

    output = ClippedSTE.apply(input_tensor, min_val, max_val)
    output.sum().backward()

    print()
    print("Input Tensor:", input_tensor)
    print("Clipped Output:", output)
    print("Input Tensor Gradient:", input_tensor.grad)
