import torch
import argparse
import nibabel as nib
from monai.transforms import LoadImage, AddChannel, ScaleIntensity, ToTensor
from models.mednext import MedNeXt

def infer_single_subject(model, image_paths, device):
    # Load and preprocess
    transforms = [
        LoadImage(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        ToTensor()
    ]
    images = []
    for path in image_paths:
        img = path
        for t in transforms:
            img = t(img)
        images.append(img)
    images = torch.stack(images).to(device)  # [4, H, W, D]
    
    # Infer
    model.eval()
    with torch.no_grad():
        output = model(images.unsqueeze(0))  # [1, 4, H, W, D]
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    
    return output.cpu()

def main():
    parser = argparse.ArgumentParser(description="Infer MedNeXt on BraTS 2021 subject")
    parser.add_argument("--model_path", type=str, default="best_mednext_brats.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/BraTS2021_Training_Data",
                        help="Path to BraTS 2021 dataset")
    parser.add_argument("--subject_id", type=str, default="BraTS2021_00000",
                        help="Subject ID (e.g., BraTS2021_00000)")
    parser.add_argument("--output_nifti", type=str, default="prediction.nii.gz",
                        help="Path to save predicted segmentation (NIfTI)")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedNeXt(in_channels=4, n_channels=32, n_classes=4).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Image paths
    image_paths = [
        f"{args.data_dir}/{args.subject_id}/{args.subject_id}_flair.nii.gz",
        f"{args.data_dir}/{args.subject_id}/{args.subject_id}_t1.nii.gz",
        f"{args.data_dir}/{args.subject_id}/{args.subject_id}_t1ce.nii.gz",
        f"{args.data_dir}/{args.subject_id}/{args.subject_id}_t2.nii.gz"
    ]
    
    # Run inference
    output = infer_single_subject(model, image_paths, device)
    print("Segmentation shape:", output.shape)
    
    # Save as NIfTI
    # Load one input image to get affine and header
    ref_nii = nib.load(image_paths[0])
    # Convert output to numpy [H, W, D, 4]
    output_np = output.squeeze(0).permute(1, 2, 3, 0).numpy()
    # Argmax to get class indices [H, W, D]
    output_np = output_np.argmax(axis=-1).astype(np.uint8)
    nii_img = nib.Nifti1Image(output_np, affine=ref_nii.affine, header=ref_nii.header)
    nib.save(nii_img, args.output_nifti)
    print(f"Saved prediction to {args.output_nifti}")

if __name__ == "__main__":
    main()