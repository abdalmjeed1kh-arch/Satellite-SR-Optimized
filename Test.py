import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import random
import os
from Model import SRresnet
from Dataset import SRDataset

def psnr(sr, hr):
    # Clamp to ensure valid pixel range [0, 1]
    sr = torch.clamp(sr, 0, 1)
    hr = torch.clamp(hr, 0, 1)
    
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)

def evaluate_model(model, test_loader, device):
    """Function called by Train.py during training"""
    model.eval()
    total_psnr = 0
    
    with torch.no_grad():
        for lr, hr in test_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            # --- Sync Cropping Logic ---
            h_sr, w_sr = sr.shape[2], sr.shape[3]
            h_hr, w_hr = hr.shape[2], hr.shape[3]
            top = (h_hr - h_sr) // 2
            left = (w_hr - w_sr) // 2
            hr_cropped = hr[:, :, top:top+h_sr, left:left+w_sr]
            
            total_psnr += psnr(sr, hr_cropped).item()
            
    return total_psnr / len(test_loader) if len(test_loader) > 0 else 0

def visualize_samples(model, test_dataset, device, num_samples=5):
    """Generates side-by-side comparison images"""
    model.eval()
    for i in range(num_samples):
        idx = random.randint(0, len(test_dataset) - 1)
        lr, hr = test_dataset[idx]
        
        lr_tensor = lr.unsqueeze(0).to(device)
        hr_tensor = hr.unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(lr_tensor)
            
            h_sr, w_sr = sr_tensor.shape[2], sr_tensor.shape[3]
            h_hr, w_hr = hr_tensor.shape[2], hr_tensor.shape[3]
            top = (h_hr - h_sr) // 2
            left = (w_hr - w_sr) // 2
            
            hr_tensor_cropped = hr_tensor[:, :, top:top+h_sr, left:left+w_sr]
            sr_tensor = torch.clamp(sr_tensor, 0, 1)

        current_psnr = psnr(sr_tensor, hr_tensor_cropped)

        # Convert to numpy
        lr_img = lr.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        sr_img = sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy().clip(0, 1)
        hr_img = hr_tensor_cropped.squeeze().permute(1, 2, 0).cpu().numpy().clip(0, 1)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.title("LR Input"); plt.imshow(lr_img)
        plt.subplot(1, 3, 2); plt.title(f"SR Output ({current_psnr:.2f}dB)"); plt.imshow(sr_img)
        plt.subplot(1, 3, 3); plt.title("HR Ground Truth"); plt.imshow(hr_img)
        
        plt.tight_layout()
        plt.savefig(f"result_sample_{i}.png")
        plt.close() # Close to save memory during long evals

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = SRDataset(
        lr_folder="E:\\Datasets\\LR_new",
        hr_folder="E:\\Datasets\\Hr",
        patch_size=None,
        split='test',
        test_ratio=0.1
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = SRresnet(scale_factor=4).to(device)
    
    # Load the PSNR-optimized model we saved in Train.py
    model_path = "best_model_psnr.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded {model_path}")
    else:
        print("Warning: best_model_psnr.pth not found. Testing with random weights.")

    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0
    total_ssim = 0

    print(f"Starting Evaluation on {len(test_dataset)} images...")

    with torch.no_grad():
        for batch_idx, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            h_sr, w_sr = sr.shape[2], sr.shape[3]
            h_hr, w_hr = hr.shape[2], hr.shape[3]
            top, left = (h_hr - h_sr) // 2, (w_hr - w_sr) // 2
            hr_cropped = hr[:, :, top:top+h_sr, left:left+w_sr]
            
            sr = torch.clamp(sr, 0, 1)

            batch_psnr = psnr(sr, hr_cropped)
            batch_ssim = ssim_metric(sr, hr_cropped)

            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()

            if batch_idx % 20 == 0:
                print(f"[{batch_idx}/{len(test_dataset)}] PSNR: {batch_psnr:.2f} | SSIM: {batch_ssim:.4f}")

    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    print(f"\nFinal Average PSNR: {avg_psnr:.2f} dB")
    print(f"Final Average SSIM: {avg_ssim:.4f}")

    visualize_samples(model, test_dataset, device)

if __name__ == "__main__":
    main()