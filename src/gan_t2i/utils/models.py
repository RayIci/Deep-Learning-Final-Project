import torch
import torch.nn as nn
import torch.optim as optim

class NormalSampler(nn.Module):
    def __init__(self, init_mean=0.0, init_std=1.0, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(NormalSampler, self).__init__()
        self.mean = nn.Parameter(torch.tensor(init_mean)).to(device)
        self.std = nn.Parameter(torch.tensor(init_std)).to(device)
    
    def forward(self, sample_shape):
        normal_dist = torch.distributions.Normal(self.mean, self.std)
        return normal_dist.sample(sample_shape)