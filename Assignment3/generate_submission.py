from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset import (
    MedicalImageDataset, 
    LoadImaged, 
    ToTensord, 
    ResizeImaged
)
from src.model import SimpleCNNUNet
from src.model_p2 import TransUNet
from src.utils import dict_to_device, set_random_seed

# ============================================================
# ⭐ NEW: ComposeDict (same as train.py)
# ============================================================
class ComposeDict:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--public_dir', type=str, required=True)
    parser.add_argument('--private_dir', type=str, required=True)
    return parser.parse_args()


def rle(image: np.ndarray) -> str:
    indices = np.where(image.flatten(order='F'))[0] + 1
    
    if len(indices) == 0:
        return "-1"

    prev, code = -1, []
    for ind in indices:
        if ind > prev + 1:
            code.append(ind)
            code.append(0)
        code[-1] += 1
        prev = ind

    return ' '.join(map(str, code))


def infer(dataset, model, device):
    pred_dict = {}
    model.eval()

    print(f"Processing {len(dataset)} images...")

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        filename = data['filename']

        img_path = os.path.join(dataset.data_dir, 'image', filename)
        with Image.open(img_path) as original_img:
            orig_w, orig_h = original_img.size

        data = dict_to_device(data, device)

        with torch.inference_mode():
            input_tensor = data["image"].unsqueeze(0)
            output_logits = model(input_tensor)

            # resize back to original size
            output_logits = F.interpolate(
                output_logits,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )

            pred = (torch.sigmoid(output_logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            pred_dict[filename] = rle(pred)

    return pred_dict


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))

    # ============================================================
    # ⭐ Replace torchvision.Compose → ComposeDict
    # ============================================================
    transform = ComposeDict([
        LoadImaged(keys=["image"]),
        ToTensord(keys=["image"]),
        ResizeImaged(keys=["image"], **config.transform.resized_imaged),
    ])

    public_dataset = MedicalImageDataset(data_dir=args.public_dir, has_label=False, transform=transform)
    private_dataset = MedicalImageDataset(data_dir=args.private_dir, has_label=False, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.model.name

    print(f"Loading model: {model_name} from {config.checkpoint.path}")

    if model_name == "SimpleCNNUNet":
        model = SimpleCNNUNet(**config.model.args)
    elif model_name == "TransUNet":
        model = TransUNet(**config.model.args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    checkpoint_path = config.checkpoint.path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        exit(1)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print("Inferencing Public Dataset...")
    public_pred = infer(public_dataset, model, device)

    print("Inferencing Private Dataset...")
    private_pred = infer(private_dataset, model, device)

    all_pred = {**public_pred, **private_pred}

    df = pd.DataFrame.from_dict(all_pred, orient="index", columns=["label"])
    df.index.name = "row ID"

    output_path = "submission.csv"
    df.reset_index().to_csv(output_path, index=False, encoding='utf-8')

    print(f"Done! Saved to {output_path}")
    print(f"Total Rows: {len(df)}")

    with open(output_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        print(f"Header Check: {header}")
