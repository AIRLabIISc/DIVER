import os
import argparse
import torch
from torchvision.utils import save_image
import torch
import gzip
from torch.utils.data import DataLoader

# Import the dataset and model classes from the training script
from illuminatenet_main import DatasetLoad, IlluminateNetModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
def load_checkpoint(checkpoint_path, rs_model, rs_optimizer):
    # Open the checkpoint file with gzip for decompression
    with gzip.open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f)
    
    
    key_map = {
        'bs_model_state_dict': 'rs_model_state_dict',
        'bs_optimizer_state_dict': 'rs_optimizer_state_dict',
    }

    for old_key, new_key in key_map.items():
        if old_key in checkpoint and new_key not in checkpoint:
            checkpoint[new_key] = checkpoint.pop(old_key)

    # Load the state dictionaries into the models and optimizers
    rs_model.load_state_dict(checkpoint['rs_model_state_dict'])
    rs_optimizer.load_state_dict(checkpoint['rs_optimizer_state_dict'])
    
    return rs_model, rs_optimizer

def main(args):
    # Load models
    rs_model = IlluminateNetModel().to(device)
    os.makedirs(args.output, exist_ok=True)
    rs_optimizer = torch.optim.Adam(rs_model.parameters(), lr=args.init_lr)
    
    # Load the latest checkpoint
    checkpoint_path = sorted(os.listdir(args.checkpoints))[-1]
    checkpoint_path = os.path.join(args.checkpoints, checkpoint_path)
    rs_model, rs_optimizer = load_checkpoint(checkpoint_path, rs_model, rs_optimizer)

    # Prepare the dataset and dataloader for inference
    inference_dataset = DatasetLoad(
        args.images, 
        args.depth, 
        args.depth_16u, 
        args.mask_max_depth, 
        args.height, 
        args.width, 
        device=device
    )

    dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    
    rs_model.eval()
     
    with torch.no_grad():
        for i, (left, depth, fnames) in enumerate(dataloader):
            image_batch = left.to(device)
            depth = depth.to(device)

            # Perform inference
            S1, S,I, D, S = rs_model(image_batch, depth)
            
            # Process and save results
            D_img = torch.clamp(D, 0., 1.).cpu()
            S_img = torch.clamp(S, 0., 1.).cpu()
            I_img = torch.clamp(I, 0., 1.).cpu()
            
            for n in range(image_batch.size(0)):
                fname = fnames[0][n]
                # save_image(D_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-direct.png'))
                # save_image(S_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-backscatter.png'))
                save_image(I_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-corrected.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--depth', type=str, required=True, help='Path to the depth folder')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, required=True, help='Path to the checkpoint folder')
    parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1952, help='Width of the image and depth files')
    parser.add_argument('--depth_16u', action='store_true', help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true', help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')

    args = parser.parse_args()
    main(args)