import os
import argparse
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from hydroopticnet_main import DatasetLoad, VeilNet, AttenNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
def load_checkpoint(checkpoint_path, veil_model, atten_net_model, veil_optimizer, atten_net_optimizer):
    # Load directly without gzip
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    
    key_map = {
        'bs_model_state_dict': 'veil_model_state_dict',
        'bs_optimizer_state_dict': 'veil_optimizer_state_dict',
        'da_model_state_dict': 'atten_net_model_state_dict',          
    }

    for old_key, new_key in key_map.items():
        if old_key in checkpoint and new_key not in checkpoint:
            checkpoint[new_key] = checkpoint.pop(old_key)


    veil_sd = checkpoint['veil_model_state_dict']
    rename_map = {
        'beta_b_coef': 'alp_b_coef',
        'beta_d_coef': 'alp_d_coef',
        'backscatter_conv.weight': 'veil_conv.weight'
    }

    for old_key, new_key in rename_map.items():
        if old_key in veil_sd:
            veil_sd[new_key] = veil_sd.pop(old_key)
            
    veil_model.load_state_dict(veil_sd, strict=True)       
    
    atten_sd_key = (
        'atten_net_model_state_dict'
        if 'atten_net_model_state_dict' in checkpoint
        else 'da_model_state_dict'
    )

    atten_net_model.load_state_dict(
        checkpoint[atten_sd_key],
        strict=True
    )

    veil_optimizer.load_state_dict(checkpoint['veil_optimizer_state_dict'])
    
    atten_opt_key = (
        'atten_net_optimizer_state_dict'
        if 'atten_net_optimizer_state_dict' in checkpoint
        else 'da_optimizer_state_dict'
    )

    atten_net_optimizer.load_state_dict(
        checkpoint[atten_opt_key]
    )
    
    return veil_model, atten_net_model, veil_optimizer, atten_net_optimizer

def main(args):
    # Load models
    veil_model = VeilNet().to(device)
    atten_net_model = AttenNet().to(device)
    veil_optimizer = torch.optim.Adam(veil_model.parameters(), lr=args.init_lr)
    atten_net_optimizer = torch.optim.Adam(atten_net_model.parameters(), lr=args.init_lr)
    save_dir = args.output
    
    os.makedirs(save_dir, exist_ok=True)
    # Load the latest checkpoint
    checkpoint_path = sorted(os.listdir(args.checkpoints))[-1]
    checkpoint_path = os.path.join(args.checkpoints, checkpoint_path)
    veil_model, atten_net_model, veil_optimizer, atten_net_optimizer = load_checkpoint(checkpoint_path, veil_model, atten_net_model, veil_optimizer, atten_net_optimizer)

    # Prepare the dataset and dataloader for inference
    inference_dataset = DatasetLoad(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height, args.width, args.device)
    dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    
    veil_model.eval()
    atten_net_model.eval()
    
    with torch.no_grad():
        for i, (left, depth, fnames) in enumerate(dataloader):
            image_batch = left.to(device)
            depth = depth.to(device)

            # Perform inference
            direct, veil = veil_model(image_batch, depth)
            f, J = atten_net_model(direct, depth)
            
            # Process and save results
            direct_img = torch.clamp(direct, 0., 1.).cpu()
            veil_img = torch.clamp(veil, 0., 1.).cpu()
            f_img = f.detach().cpu()
            f_img = f_img / f_img.max()
            J_img = torch.clamp(J, 0., 1.).cpu()
            
            for n in range(image_batch.size(0)):
                fname = fnames[0][n]
                # save_image(direct_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-direct.png'))
                # save_image(veil_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-backscatter.png'))
                # save_image(f_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-f.png'))
                save_image(J_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-corrected.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--images', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--depth', type=str, required=True, help='Path to the depth folder')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, required=True, help='Path to the checkpoint folder')
    parser.add_argument('--height', type=int, default=640, help='Height of the image and depth files')
    #parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=640, help='Width of the image and depth files')
    parser.add_argument('--depth_16u', action='store_true', help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true', help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)