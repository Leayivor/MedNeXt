import os
import torch
import torch.nn as nn
import argparse
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from tqdm import tqdm
from dataloader import get_brats2021_dataloaders
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt

def train_model(model, train_loader, val_loader, device, args):
    # Loss and optimizer
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler()
    
    # Metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Resumed from checkpoint: {args.resume}")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            images = batch["image"].to(device)  # [B, 4, H, W, D]
            labels = batch["label"].to(device)  # [B, 1, H, W, D]
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)  # [B, num_classes, H, W, D]
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                
                dice_metric(y_pred=outputs, y=labels)
        
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.output_path)
            print(f"Saved best model to {args.output_path}")

def evaluate_model(model, val_loader, device):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            
            dice_metric(y_pred=outputs, y=labels)
    
    final_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Final Validation Dice: {final_dice:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train MedNeXt on BraTS 2021")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/BraTS2021_Training_Data",
                        help="Path to BraTS 2021 dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_channels", type=int, default=32, help="Base number of channels in MedNeXt")
    parser.add_argument("--exp_r", type=int, default=2, help="Expansion ratio in MedNeXt")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--train_val_split", type=float, default=0.8,
                        help="Fraction of data for training")
    parser.add_argument("--output_path", type=str, default="best_mednext_brats.pth",
                        help="Path to save best model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--evaluate_only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loaders
    train_loader, val_loader = get_brats2021_dataloaders(args)
    
    # Model
    model = MedNeXt(
        in_channels=4,  # 4 MRI modalities
        n_channels=args.n_channels,
        n_classes=4,    # 4 classes: background, necrotic, edema, enhancing
        exp_r=args.exp_r,
        kernel_size=3,
        deep_supervision=False
    ).to(device)
    
    if args.evaluate_only:
        if not args.resume:
            raise ValueError("Must provide --resume checkpoint for evaluation")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        evaluate_model(model, val_loader, device)
    else:
        train_model(model, train_loader, val_loader, device, args)

if __name__ == "__main__":
    main()