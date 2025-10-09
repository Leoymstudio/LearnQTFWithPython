import torch

class LatticeField:
    def __init__(self, L, device='cpu'):
        self.L = L
        self.field = torch.randn(L,L, device=device)  # 可选为 nn.Parameter
    def action(self):
        # periodic boundary 可用 roll 实现
        kinetic = 0.5 * torch.sum((self.field - torch.roll(self.field, shifts=1, dims=0))**2) \
                + 0.5 * torch.sum((self.field - torch.roll(self.field, shifts=1, dims=1))**2)
        mass = 0.5 * 0.1 * torch.sum(self.field**2)
        interaction = 0.01 * torch.sum(self.field**4)
        return kinetic + mass + interaction
    def metropolis_step(self, eps=0.1):
        old = self.field.clone()
        old_action = self.action()
        proposal = old + eps * torch.randn_like(old)
        self.field = proposal
        new_action = self.action()
        accept = torch.rand(1, device=self.field.device) < torch.exp(old_action - new_action)
        if not accept:
            self.field = old
