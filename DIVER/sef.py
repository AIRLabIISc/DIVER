import numpy as np
import cv2 as cv

def spectral_equalization_filter(img, alpha=0.3, eps=1e-6):
    """
    Multiplicative Gain Spectral Equalization Filter (SEF).
    
    Parameters:
        img (np.ndarray): Input BGR image (uint8).
        alpha (float): Correction strength in (0, 1]. 
                       1.0 gives full mean equalization; 
                       The current default is 0.3 for gentler channel correction.
        eps (float): Numerical stability term to avoid division by zero.
        
    Returns:
        np.ndarray: Equalized BGR image (uint8).
    """
    # 1. Convert to float32 for computation
    img_float = img.astype(np.float32)
    B, G, R = cv.split(img_float)
    
    # 2. Compute channel means mu^c
    mu_B = np.mean(B)
    mu_G = np.mean(G)
    mu_R = np.mean(R)
    
    # 3. Dominant channel mean C_sup = max(mu^R, mu^G, mu^B)
    C_sup = max(mu_R, mu_G, mu_B)
    
    # 4. Relative gain g_c = C_sup / (mu^c + eps)
    g_B = C_sup / (mu_B + eps)
    g_G = C_sup / (mu_G + eps)
    g_R = C_sup / (mu_R + eps)
    
    # 5. Multiplicative normalization: U_S^c = U^c * (g_c)^alpha
    B_eq = np.clip(B * (g_B ** alpha), 0, 255).astype(np.uint8)
    G_eq = np.clip(G * (g_G ** alpha), 0, 255).astype(np.uint8)
    R_eq = np.clip(R * (g_R ** alpha), 0, 255).astype(np.uint8)
    
    # 6. Recombine BGR channels
    return cv.merge([B_eq, G_eq, R_eq])
