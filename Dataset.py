import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class SRDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, patch_size=None, split='train', test_ratio=0.1):
        self.lr_images = sorted(os.listdir(lr_folder))
        self.hr_images = sorted(os.listdir(hr_folder))
        assert len(self.lr_images) == len(self.hr_images), "LR and HR folders must match"
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.patch_size = patch_size
        split_index = int(len(self.lr_images) * (1 - test_ratio))

        if split == 'train':
            self.lr_images = self.lr_images[:split_index]
            self.hr_images = self.hr_images[:split_index]
        else:  # test
            self.lr_images = self.lr_images[split_index:]
            self.hr_images = self.hr_images[split_index:]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        # Load LR and HR images
        
        lr_path = os.path.join(self.lr_folder, self.lr_images[idx])
        hr_path = os.path.join(self.hr_folder, self.hr_images[idx])
        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")
        
        
        # Optional patch cropping (same region in both LR and HR)
        if self.patch_size is not None:
            scale = 4
            lr_patch_size = self.patch_size // scale
            left = random.randint(0, lr.width - lr_patch_size)
            top = random.randint(0, lr.height - lr_patch_size)

            
            #crop lr
            lr = lr.crop((left, top, left + lr_patch_size, top + lr_patch_size))

            #fixed scale
            

            hr = hr.crop((
                left * scale,
                top * scale,
               (left + lr_patch_size) * scale,
               (top + lr_patch_size) * scale
            ))
           
        # Convert to tensors
        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)
        
        
        return lr, hr


# Example usage
if __name__ == "__main__":
    dataset = SRDataset("D:\\for proj\\LR_new", "D:\\for proj\\hr", patch_size=128) # set destination for low and high resolution images
    lr, hr = dataset[0]
    print("LR shape:", lr.shape)
    print("HR shape:", hr.shape)
    print(hr.shape[1] / lr.shape[1])