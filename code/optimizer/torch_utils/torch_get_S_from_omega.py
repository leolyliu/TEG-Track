import torch


class GetSFromOmega(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        S = torch.tensor([[0, -input[2], input[1]], [input[2], 0, -input[0]], [-input[1], input[0], 0]], dtype=torch.float32, device=input.device)
        return S

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = torch.zeros(input.shape, dtype=torch.float32, device=input.device)
        grad_input[0] += - grad_output[1, 2] + grad_output[2, 1]
        grad_input[1] += grad_output[0, 2] - grad_output[2, 0]
        grad_input[2] += - grad_output[0, 1] + grad_output[1, 0]
        return grad_input
