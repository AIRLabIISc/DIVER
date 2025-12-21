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
        
        # print("\n[VeilNet Forward Pass]")
        # print("alp_b_conv.mean():", alp_b_conv.mean().item())
        # print("alp_d_conv.mean():", alp_d_conv.mean().item())
        # print("Bc.mean():", Bc.mean().item())
        
        # stats = {
        #     "alp_b_conv_mean": alp_b_conv.mean().item(),
        #     "alp_d_conv_mean": alp_d_conv.mean().item(),
        #     "Bc_mean": Bc.mean().item(),
        #     "J_prime_values": self.J_prime.detach().cpu().numpy().tolist(),
        #     "B_inf_values": self.B_inf.detach().cpu().numpy().tolist()
        # }
        
        veil_masked = veil * (depth > 0.).repeat(1, 3, 1, 1)
        direct = image - veil_masked
        return direct, veil


class VeilLoss(nn.Module):
    def __init__(self, cost_ratio=1000.0, delta=0.5):
        super(VeilLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio
        self.delta = delta


    def forward(self, direct):
        # Positive loss (ReLU applied to direct)
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        
        # Negative loss (ReLU applied to -direct using Adaptive Huber Loss)
        neg = adaptive_huber(self.relu(-direct), torch.zeros_like(direct), self.delta)
        neg_mean = torch.mean(neg)

        print(f"\n[Veil Loss]")
        print(f"L1 loss (positive values): {pos.item():.6f}")
        print(f"SmoothL1 loss (negative values): {neg.mean().item():.6f}")
        
        
        # veil loss combining positive and negative parts
        veil_loss = self.cost_ratio * neg_mean + pos
        print(f"Total veilnet Loss: {veil_loss.item():.6f}")
        stats = {
            "L1 loss (positive values)":pos.item(),
            "SmoothL1 loss (negative values)": neg.mean().item(),
            "Total veilnet Loss": veil_loss.item(),
            # Add other losses as needed
        }
        return veil_loss,stats