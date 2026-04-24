import torch
from torch.utils.data import DataLoader
from datetime import datetime
from Model import SRresnet
from Dataset import SRDataset
from Test import evaluate_model
import math 

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Datasets (Train and Test)
    train_dataset = SRDataset(
        lr_folder="E:\\Datasets\\LR_new",
        hr_folder="E:\\Datasets\\Hr",
        patch_size=None, 
        split='train',
        test_ratio=0.1
    )

    test_dataset = SRDataset(
        lr_folder="E:\\Datasets\\LR_new",
        hr_folder="E:\\Datasets\\Hr",
        patch_size=None, 
        split='test',
        test_ratio=0.1
    )

    # 2. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16, 
        shuffle=True,
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True 
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model & Config
    model = SRresnet(scale_factor=4).to(device)
    model.load_state_dict(torch.load("best_model_psnr.pth")) 
    print(" Weights loaded!")
    num_epochs = 10
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"Train size: {len(train_dataset)} | Batches: {len(train_loader)}")
    
    best_psnr = 0.0 

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            # Generate SR
            sr = model(lr)

            # Dynamic Cropping
            h_sr, w_sr = sr.size(2), sr.size(3)
            h_hr, w_hr = hr.size(2), hr.size(3)
            top = (h_hr - h_sr) // 2
            left = (w_hr - w_sr) // 2
            hr_cropped = hr[:, :, top:top+h_sr, left:left+w_sr]

            # Loss & Backprop
            loss = criterion(sr, hr_cropped)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Quick Pulse Check (Every 50 batches) ---
            if batch_idx % 50 == 0:
                current_clock = datetime.now().strftime("%H:%M:%S")
                # Calculate simple PSNR on the fly
                with torch.no_grad():
                    mse = torch.mean((sr - hr_cropped) ** 2)
                    quick_psnr = 10 * math.log10(1 / mse.item()) if mse.item() > 0 else 100

                print(f"[{current_clock}] Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | Quick PSNR: {quick_psnr:.2f}dB")
            
            epoch_loss += loss.item()

        # End of Epoch Math
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 3. Validation (Runs once per epoch)
        current_psnr = evaluate_model(model, test_loader, device)

        # 4. Final Summary for Epoch
        print(f"\n" + "="*40)
        print(f"EPOCH [{epoch+1}/{num_epochs}] COMPLETE")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"Validation PSNR: {current_psnr:.2f} dB")

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            torch.save(model.state_dict(), "best_model_psnr.pth")
            print(f" NEW BEST! Saved model with {best_psnr:.2f} dB")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()