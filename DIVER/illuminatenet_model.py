import torch
import torch.nn as nn

class IlluminateNetModel(nn.Module):
    """_
    Input: Low_lit raw image+ transmission map
    Uses Image formation model to enhance the underwater image
    
    """
    def __init__(self):
        super().__init__()
        # Define separate layers for each channel
        self.R_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.G_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.B_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.map = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize weights
        nn.init.uniform_(self.R_layer.weight, 0, 5)
        nn.init.uniform_(self.G_layer.weight, 0, 5)
        nn.init.uniform_(self.B_layer.weight, 0, 5)
        
        nn.init.uniform_(self.map.weight, 0, 5)
    
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.mish = nn.Mish()

    def forward(self, image, depth):
        # Separate the R, G, B channels
        R = image[:, 0:1, :, :]
        G = image[:, 1:2, :, :]
        B = image[:, 2:3, :, :]

        # Enhance each channel separately
        R_enhanced = self.relu(self.R_layer(R))
        G_enhanced = self.relu(self.G_layer(G))
        B_enhanced = self.relu(self.B_layer(B))
        
        # Combine the enhanced channels back into a single image
        S1 = torch.cat([R_enhanced, G_enhanced, B_enhanced], dim=1)
        S = self.tanh(S1)

        D = image - S
        
        # Dividing by depth
        T = D / (depth + 1e-6) 
        T = self.relu(self.map(T))
       
        # Final enhanced image - using image formation model
        I = (S+T)
       
        return S1,S,I,D,T


class IlluminateNetLoss(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def forward(self,S1,I):
        channel_intensities = torch.mean(I, dim=[2, 3], keepdim=True)
        lum_loss = (channel_intensities - self.target_intensity).square().mean()
        mean_color = S1.mean(dim=(2, 3), keepdim=True)  # Average across spatial dimensions
        gray_level = mean_color.mean(dim=1, keepdim=True)  # Average across RGB channels
        GW_loss = (mean_color - gray_level).abs().mean()
        return  lum_loss + GW_loss
