import os
import argparse
import torch
from time import time
from torchvision.utils import save_image
from PIL import Image
try:
    from tqdm import trange , tqdm
except:
    trange = range
import numpy as np
import pandas as pd
from PIL import Image
from scores import getUCIQE, getUIQM, get_avg_rgb_hsv
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from veilnet_model import VeilNet, VeilLoss
from attennet_model import AttenNet, AttenLoss
from torch.utils.data import DataLoader
from data_load_and_logger import DatasetLoad, LoggerToFile

# Device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def main(args):

    uciqe_values = []
    uqims_values=[]
    uqims_values=[]
    uicm_values=[]
    uism_values=[]
    uicomn_values=[]
    output_names = []
    avg_r_values = []
    avg_g_values = []
    avg_b_values = []
    avg_h_values = []
    avg_s_values = []
    avg_v_values = []
    veil_loss_values = []
    atten_net_loss_values = []
    iteration_indices = []
    
    logger = LoggerToFile(os.path.join(args.output, 'training_log.txt'))
    seed = int(torch.randint(9223372036854775807, (1,))[0]) if args.seed is None else args.seed
    
    if args.seed is None:
        print('Seed:', seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    train_dataset = DatasetLoad(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height,
                                             args.width, args.device)
    save_dir = args.output
    check_dir = args.checkpoints
    os.makedirs(save_dir, exist_ok=True)
    target_batch_size = args.batch_size
    os.makedirs(check_dir, exist_ok=True)
    dataloader = DataLoader(train_dataset, batch_size=target_batch_size, shuffle=False)
    veil_model = VeilNet().to(device)
    atten_net_model = AttenNet().to(device)
    veil_criterion = VeilLoss().to(device)
    atten_net_criterion = AttenLoss().to(device)
    veil_optimizer = torch.optim.Adam(veil_model.parameters(), lr=args.init_lr)
    atten_net_optimizer = torch.optim.Adam(atten_net_model.parameters(), lr=args.init_lr)
    skip_right = True
    total_veil_eval_time = 0.
    total_veil_evals = 0
    total_at_eval_time = 0.
    total_at_evals = 0
    for j, (left, depth, fnames) in enumerate(dataloader):
        print("training")
        image_batch = left
        batch_size = image_batch.shape[0]
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            direct, veil= veil_model(image_batch, depth)
            # logger.log(f"alp_b_conv.mean(): {stats['alp_b_conv_mean']}")
            # logger.log(f"alp_d_conv.mean(): {stats['alp_d_conv_mean']}")
            # logger.log(f"Bc.mean(): {stats['Bc_mean']}")
            
            veil_loss,veil_stats = veil_criterion(direct)
            logger.log(f"veilLoss stats: {veil_stats}")

            veil_optimizer.zero_grad()
            veil_loss.backward()
            veil_optimizer.step()
            total_veil_eval_time += time() - start
            total_veil_evals += batch_size
            iteration_indices.append(f"veilnet_{j}_{iter}")
            veil_loss_values.append(veil_loss.item())
            atten_net_loss_values.append(None)
        direct_mean = direct.mean(dim=[2, 3], keepdim=True)
        direct_std = direct.std(dim=[2, 3], keepdim=True)
        direct_z = (direct - direct_mean) / direct_std
        clamped_z = torch.clamp(direct_z, -5, 5)
        direct_no_grad = torch.clamp(
            (clamped_z * direct_std) + torch.maximum(direct_mean, torch.Tensor([1. / 255]).to(device)), 0, 1).detach()
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            f, J = atten_net_model(direct_no_grad, depth)
            # logger.log(f"AttenNet attn_1 mean: {stats['attn_1_mean']}")
            # logger.log(f"alpha_1_term1_mean {stats['alp_0_mean']}")
            # logger.log(f"alpha_1_term2_mean:{stats['alp_1_mean']}")
            # logger.log(f"alpha_1_term3_mean:{stats['alp_2_mean']}")
            # logger.log(f"AttenNet attenuation_coef: {stats['attenuation_coef']}")
            
            atten_net_loss,atten_net_stats= atten_net_criterion(direct_no_grad, J)
            logger.log(f"DeattenuationLoss stats: {atten_net_stats}")
            atten_net_optimizer.zero_grad()
            atten_net_loss.backward()
            atten_net_optimizer.step()
            total_at_eval_time += time() - start
            total_at_evals += batch_size

            iteration_indices.append(f"attennet_{j}_{iter}")
            veil_loss_values.append(None)
            atten_net_loss_values.append(atten_net_loss.item()) 
                      
        print("Losses: %.9f %.9f" % (veil_loss.item(), atten_net_loss.item()))
        log_file_path = os.path.join(args.output, "loss_log.txt")
        
        # Save the loss
        with open(log_file_path, "a") as text:
            text.write("veilnet Loss: %.9f | attennet Loss: %.9f\n" % (veil_loss.item(), atten_net_loss.item()))

        avg_veil_time = total_veil_eval_time / total_veil_evals * 1000
        avg_at_time = total_at_eval_time / total_at_evals * 1000
        avg_time = avg_veil_time + avg_at_time
        print("Avg time per eval: %f ms (%f ms veil, %f ms at)" % (avg_time, avg_veil_time, avg_at_time))
        img = image_batch.cpu()
        direct_img = torch.clamp(direct_no_grad, 0., 1.).cpu()
        veil_img = torch.clamp(veil, 0., 1.).detach().cpu()
        # BC1 = torch.clamp(bc1, 0., 1.).detach().cpu()
        # BC2 = torch.clamp(bc2, 0., 1.).detach().cpu()
        # A1_img=Image.fromarray((torch.clamp(A1[0], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8)) 
        # A2_img=Image.fromarray((torch.clamp(A2[0], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8)) 
        # A3_img=Image.fromarray((torch.clamp(A3[0], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8)) 
        # J_prime_img=Image.fromarray((torch.clamp(J_prime[0], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8))
        # B_inf_img=Image.fromarray((torch.clamp(B_inf[0], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8))  
        f_img = f.detach().cpu()
        f_img = f_img / f_img.max()
        J_img = torch.clamp(J, 0., 1.).cpu()
        for side in range(1 if skip_right else 2):
            side_name = 'left' if side == 0 else 'right'
            names = fnames[side]
            for n in range(batch_size):
                i = n + target_batch_size * side
                if args.save_intermediates:
                    save_image(direct_img[i], "%s/%s-direct.png" % (save_dir, names[n].rstrip('.png')))
                    save_image(veil_img[i], "%s/%s-veil.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(BC1[i], "%s/%s-BC1.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(BC2[i], "%s/%s-BC2.png" % (save_dir, names[n].rstrip('.png')))
                    # A1_img.save("%s/%s-A1.png" % (save_dir, names[n].rstrip('.png')))
                    # A2_img.save("%s/%s-A2.png" % (save_dir, names[n].rstrip('.png')))
                    # A3_img.save("%s/%s-A3.png" % (save_dir, names[n].rstrip('.png')))
                    # B_inf_img.save("%s/%s-B_inf.png" % (save_dir, names[n].rstrip('.png')))
                    # J_prime_img.save("%s/%s-J_prime.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(A2_img, "%s/%s-A2.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(A3_img, "%s/%s-A3.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(B_inf_img, "%s/%s-B_inf.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(J_prime_img, "%s/%s-J_prime.png" % (save_dir, names[n].rstrip('.png'))
                    
                    save_image(f_img[i], "%s/%s-f.png" % (save_dir, names[n].rstrip('.png')))
                save_image(J_img[i], "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png')))
                output_image_path = "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png'))
                output_image =Image.open(output_image_path)
                output_image = output_image.resize((256, 256))
                image = output_image.convert('RGB')
                avg_rgb, avg_hsv = get_avg_rgb_hsv(image)
                avg_r_values.append(avg_rgb[0])
                avg_g_values.append(avg_rgb[1])
                avg_b_values.append(avg_rgb[2])
                avg_h_values.append(avg_hsv[0])
                avg_s_values.append(avg_hsv[1])
                avg_v_values.append(avg_hsv[2])
                image_array = np.array(image)
                #output_image_np = np.array(output_image)
                #output_image_np = output_image_np.astype(np.float32)
                uciqe_value = getUCIQE(output_image_path)
                print('UCIQE:',uciqe_value)
                uqims_value,uicm_value,uism_value,uicomn_value =getUIQM(image_array)
                print('UQIMS:',uqims_value)
                uciqe_values.append(uciqe_value)
                uqims_values.append(uqims_value)
                uicm_values.append(uicm_value)
                uism_values.append(uism_value)
                uicomn_values.append(uicomn_value)
                output_names.append(names[n])


        torch.save({
            'veil_model_state_dict': veil_model.state_dict(),
            'atten_net_model_state_dict': atten_net_model.state_dict(),
            'veil_optimizer_state_dict': veil_optimizer.state_dict(),
            'atten_net_optimizer_state_dict': atten_net_optimizer.state_dict(),
        }, os.path.join(check_dir, f'model_checkpoint_{j}.pth'))


            # Save to Excel
    df = pd.DataFrame({
        'Output Image Name': output_names,
        'uciqe': uciqe_values,
        'uqims': uqims_values,
        'uicm':uicm_values,
        'uism':uism_values,
        'uicomn':uicomn_values,
        'Avg R': avg_r_values,
        'Avg G': avg_g_values,
        'Avg B': avg_b_values,
        'Avg H': avg_h_values,
        'Avg S': avg_s_values,
        'Avg V': avg_v_values,
        })
    
    excel_path = os.path.join(save_dir, 'evaluation_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Evaluation metrics saved to {excel_path}")

    loss_df = pd.DataFrame({
        'Iteration': iteration_indices,
        'veilnet_Loss': veil_loss_values,
        'attennet_Loss': atten_net_loss_values
    })

    loss_csv_path = os.path.join(save_dir, 'training_losses.csv')
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"Loss values saved to {loss_csv_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--images', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--depth', type=str, required=True, help='Path to the depth folder')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, required=True, help='Path to the checkpoint folder')
    parser.add_argument('--height', type=int, default=720, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=720, help='Width of the image and depth')
    parser.add_argument('--depth_16u', action='store_true',
                        help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true',
                        help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--seed', type=int, default=None, help='Seed to initialize network weights (use 1337 to replicate paper results)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--save_intermediates', action='store_true', default=False, help='Set to True to save intermediate files (veil, attenuation, and direct images)')
    parser.add_argument('--init_iters', type=int, default=500, help='How many iterations to refine the first image batch (should be >= iters)')
    parser.add_argument('--iters', type=int, default=50, help='How many iterations to refine each image batch')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)