import torch


class Euler2Mat(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        si, sj, sk = torch.sin(input)
        ci, cj, ck = torch.cos(input)
        cc, cs = ci*ck, ci*sk
        sc, ss = si*ck, si*sk
        M = torch.tensor([[cj*ck,sj*sc-cs,sj*cc+ss],[cj*sk,sj*ss+cc,sj*cs-sc],[-sj,cj*si,cj*ci]], dtype=torch.float32, device=input.device)
        return M
    
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        si, sj, sk = torch.sin(input)
        ci, cj, ck = torch.cos(input)
        cc, cs = ci*ck, ci*sk
        sc, ss = si*ck, si*sk

        grad_input = torch.zeros(input.shape, dtype=torch.float32, device=input.device)
        grad_input[1] += grad_output[0, 0] * ck * (-sj)
        grad_input[2] += grad_output[0, 0] * cj * (-sk)
        grad_input[0] += grad_output[0, 1] * (sj * ck * ci - sk * (-si))
        grad_input[1] += grad_output[0, 1] * si * ck * cj
        grad_input[2] += grad_output[0, 1] * (si * sj * (-sk) - ci * ck)
        grad_input[0] += grad_output[0, 2] * (sj * ck * (-si) + sk * ci)
        grad_input[1] += grad_output[0, 2] * ci * ck * cj
        grad_input[2] += grad_output[0, 2] * (ci * sj * (-sk) + si * ck)
        grad_input[1] += grad_output[1, 0] * sk * (-sj)
        grad_input[2] += grad_output[1, 0] * cj * ck
        grad_input[0] += grad_output[1, 1] * (sj * sk * ci + ck * (-si))
        grad_input[1] += grad_output[1, 1] * si * sk * cj
        grad_input[2] += grad_output[1, 1] * (si * sj * ck + ci * (-sk))
        grad_input[0] += grad_output[1, 2] * (sj * sk * (-si) - ck * ci)
        grad_input[1] += grad_output[1, 2] * ci * sk * cj
        grad_input[2] += grad_output[1, 2] * (ci * sj * ck - si * (-sk))
        grad_input[1] += grad_output[2, 0] * (-cj)
        grad_input[0] += grad_output[2, 1] * cj * ci
        grad_input[1] += grad_output[2, 1] * si * (-sj)
        grad_input[0] += grad_output[2, 2] * cj * (-si)
        grad_input[1] += grad_output[2, 2] * ci * (-sj)

        return grad_input
