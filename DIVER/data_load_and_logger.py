from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import torch
import kornia.morphology as morph

class DatasetLoad(Dataset):
    def __init__(self, image_path, depth_path, openni_depth, mask_max_depth, image_height, image_width, device):
        self.raw_image_folder = image_path
        self.depth_dir = depth_path
        self.raw_image_files = sorted(os.listdir(image_path))
        self.depth_files = sorted(os.listdir(depth_path))
        self.device = device
        self.openni_depth = openni_depth
        self.mask_max_depth = mask_max_depth
        self.crop = (0, 0, image_height, image_width)
        self.depth_perc = 0.0001
        self.kernel = torch.ones(3, 3).to(device=device)
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.crop[2], self.crop[3]), transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.PILToTensor(),
        ])
        assert len(self.raw_image_files) == len(self.depth_files)

    def __len__(self):
        return len(self.raw_image_files)

    def __getitem__(self, index):
        fname = self.raw_image_files[index]
        image = Image.open(os.path.join(self.raw_image_folder, fname))
        depth_fname = self.depth_files[index]
        depth = Image.open(os.path.join(self.depth_dir, depth_fname))
        if depth.mode != 'L':
            depth = depth.convert('L')
        depth_transformed: torch.Tensor = self.image_transforms(depth).float().to(device=self.device)
        if self.openni_depth:
            depth_transformed = depth_transformed / 1000.
        if self.mask_max_depth:
            depth_transformed[depth_transformed == 0.] = depth_transformed.max()
        low, high = torch.nanquantile(depth_transformed, self.depth_perc), torch.nanquantile(depth_transformed,
                                                                                             1. - self.depth_perc)
        
        depth_transformed[(depth_transformed < low) | (depth_transformed > high)] = 0.
        depth_transformed = torch.squeeze(morph.closing(torch.unsqueeze(depth_transformed, dim=0), self.kernel), dim=0)
        left_transformed: torch.Tensor = self.image_transforms(image).to(device=self.device) / 255.
        return left_transformed, depth_transformed, [fname]

class LoggerToFile:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        # Create the parent directory if it does not exist:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Then safely open/clear the file
        with open(self.log_file_path, 'w') as file:
            pass

    def log(self, msg):
        with open(self.log_file_path, 'a') as file:
            file.write(msg + '\n')
            