import torch
import torch.nn as nn
import cv2 
import numpy as np
from losses import color_constancy_loss, sobel_edge_loss

class AttenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 5)
        self.attenuation_coef = nn.Parameter(torch.rand(6, 1, 1))
        self.relu = nn.ReLU()
        self.wb = nn.Parameter(torch.rand(1, 1, 1))
        nn.init.constant_(self.wb, 1)
        self.output_act = nn.Sigmoid()
        self.tanh=nn.Tanh()

    def forward(self, direct, depth):
        attn_conv = torch.nn.functional.softplus(-self.relu(self.attenuation_conv(depth)))
        # print("attn_1:", attn_conv.mean().item())
        #   OLD ONE----------
        alp_d = torch.stack(tuple(
            torch.sum(attn_conv[:, i:i + 2, :, :] * self.relu(self.attenuation_coef[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1) 

        #NEW ONE-----------
        alpha_1_terms = []
        for i in range(0, 6, 2):
            alpha_1 = torch.sum(
                attn_conv[:, i:i + 2, :, :] * self.relu(self.attenuation_coef[i:i + 2]).view(2, 1, 1),
                dim=1
            )
            alpha_1_terms.append(alpha_1)
        # print("alp_d (mean):", alp_d.mean().item())
        alpha_1_sum = sum(alpha_1_terms)  # Adds alp_0 + alp_1 + alp_2

        f = torch.nn.functional.softplus(
            torch.clamp(alpha_1_sum * depth, 0, float(torch.log(torch.tensor([3.]))))
        )
    
        #f = torch.exp(torch.clamp(alp_d * depth, 0, float(torch.log(torch.tensor([3.])))))
        f = torch.nn.functional.softplus(torch.clamp(alp_d * depth, 0, float(torch.log(torch.tensor([3.])))))
        # f_masked = f * ((depth == 0.) / f + (depth > 0.))
        f_masked = f 
        J = f_masked * direct * self.wb 
        # print("AttenNet:")
        # print("alp_0_mean:", alp_terms[0].mean().item())
        # print("alp_1_mean": ,alp_terms[1].mean().item())
        # print("alp_2_mean": ,alp_terms[2].mean().item())
        
        # stats = {
        #     "attn_1_mean": attn_conv.mean().item(),
        #     "alp_0_mean": alpha_1_terms[0].mean().item(),
        #     "alp_1_mean": alpha_1_terms[1].mean().item(),
        #     "alp_2_mean": alpha_1_terms[2].mean().item(),
        #     "attenuation_coef": self.attenuation_coef.detach().cpu().numpy().tolist()
        # }
        
        nanmask = torch.isnan(J)
        if torch.any(nanmask): 
            print("Warning! NaN values in J")
            J[nanmask] = 0
        return f_masked, J
    

class AttenLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_luminance = 0.5
  
    def forward(self, direct, J):
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        lum_loss = (channel_intensities - self.target_luminance).square().mean()

        # Compute additional losses
        Y_pred = J.cpu().detach()
        Y_true = direct.cpu().detach()
        sobel_loss_value = sobel_edge_loss(Y_pred, Y_true)
        cc_loss = color_constancy_loss(J)
        if torch.any(torch.isnan(lum_loss)):
            print("NaN luminous loss!")

        
        print(f"\n[Deattenuate Loss]")
        print(f"luminous Loss: {lum_loss.item():.6f}")
        print(f"Color Constancy Loss: {cc_loss.item():.6f}")
        print(f"Sobel Loss: {sobel_loss_value:.6f}")
 
        loss= lum_loss +cc_loss+0.5*sobel_loss_value   
        print(f"Total attennet Loss: {loss.item():.6f}")
        stats = {
            "lum_loss": lum_loss.item(),
            "Color Constancy": cc_loss.item(),
            "Sobel LOss": sobel_loss_value.item(),
            "total_attennet_loss": loss.item()
        }
        # + sobel_loss_value
        return loss,stats