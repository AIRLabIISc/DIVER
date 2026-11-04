import torch
import torch.nn as nn
from losses import adaptive_huber


class VeilNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.veil_conv = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv = nn.Conv2d(1, 3, 1, bias=False)
        nn.init.uniform_(self.veil_conv.weight, 0, 5)
        nn.init.uniform_(self.residual_conv.weight, 0, 5)
        self.alp_b_coef = nn.Parameter(torch.rand(6, 1, 1))
        self.alp_d_coef = nn.Parameter(torch.rand(6, 1, 1))
        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.J_prime = nn.Parameter(torch.rand(3, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()

    def forward(self, image, depth):
        alp_b_conv = self.relu(self.veil_conv(depth))
        alp_d_conv = self.relu(self.residual_conv(depth))
        Bc1 = self.B_inf * (1 - torch.nn.functional.softplus(-alp_b_conv))
        Bc2 = self.J_prime * torch.nn.functional.softplus(-alp_d_conv)
        Bc = Bc1 + Bc2
        veil = self.tanh(Bc)
              
        veil_masked = veil * (depth > 0.).repeat(1, 3, 1, 1)
        direct = image - veil_masked
        return direct, veil


class VeilLoss(nn.Module):
    def __init__(self, cost_ratio=1.0, delta=0.5):
        super(VeilLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio
        self.delta = delta


    def forward(self, veil):
               
        penalty = adaptive_huber(veil, torch.zeros_like(veil), self.delta)
        huber_mean = penalty.mean()

        print(f"\n[Veil Loss]")
        print(f"Huber loss (backscatter magnitude): {huber_mean.item():.6f}")
        
        

        veil_loss = self.cost_ratio * huber_mean
        print(f"Total veilnet Loss: {veil_loss.item():.6f}")
        stats = {
            "Huber loss (backscatter magnitude)": huber_mean.item(),
            "Total veilnet Loss": veil_loss.item(),
        }
        return veil_loss,stats
