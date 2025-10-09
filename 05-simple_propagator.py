import torch
import torch.nn as nn

class SimplePropagator(nn.Module):
    def __init__(self, latent_dim, num_particles):
        super().__init__()
        self.propagators = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(num_particles)])
        self.vertices = nn.ModuleList([nn.Sequential(nn.Linear(3*latent_dim, latent_dim), nn.ReLU()) for _ in range(num_particles)])
    def forward(self, z):
        # z: [B, P, D]
        P = z.shape[1]
        prop = [self.propagators[i](z[:,i,:]) for i in range(P)]  # list of [B,D]
        out = []
        for i in range(P):
            neighbors = []
            for j in (i-1, i+1):
                if 0 <= j < P:
                    neighbors.append(prop[j])
            inputs = [prop[i]] + neighbors
            if len(inputs) < 3:
                # pad with zeros
                inputs += [torch.zeros_like(prop[i])] * (3 - len(inputs))
            cat = torch.cat(inputs[:3], dim=-1)
            out.append(self.vertices[i](cat))
        return torch.stack(out, dim=1)
