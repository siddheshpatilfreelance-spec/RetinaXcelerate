from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 384

DISEASE_COLS = [
    "Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM", "PRH",
    "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA", "VS", "BRAO",
    "PLQ", "HPED", "CL",
]

DISEASE_NAMES = {
    "Disease_Risk": "Any Disease Present",
    "DR": "Diabetic Retinopathy",
    "ARMD": "Age-Related Macular Degeneration",
    "MH": "Media Haze",
    "DN": "Drusen",
    "MYA": "Myopia",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tessellation",
    "ERM": "Epiretinal Membrane",
    "LS": "Laser Scars",
    "MS": "Macular Scar",
    "CSR": "Central Serous Retinopathy",
    "ODC": "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tortuous Vessels",
    "AH": "Asteroid Hyalosis",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
    "ST": "Shunt",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT": "Parafoveal Telangiectasia",
    "RT": "Retinal Traction",
    "RS": "Retinitis",
    "CRS": "Chorioretinitis",
    "EDN": "Exudation",
    "RPEC": "Retinal Pigment Epithelium Changes",
    "MHL": "Macular Hole",
    "RP": "Retinitis Pigmentosa",
    "CWS": "Cotton Wool Spots",
    "CB": "Coloboma",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "MNF": "Myelinated Nerve Fibers",
    "HR": "Hemorrhagic Retinopathy",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD": "Tilted Disc",
    "CME": "Cystoid Macular Edema",
    "PTCR": "Post-Traumatic Choroidal Rupture",
    "CF": "Choroidal Folds",
    "VH": "Vitreous Hemorrhage",
    "MCA": "Macroaneurysm",
    "VS": "Vasculitis",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ": "Plaque",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "CL": "Collateral",
}

SPLITS = {
    "train": (
        "Training_Set/Training_Set/Training",
        "Training_Set/Training_Set/RFMiD_Training_Labels.csv",
    ),
    "val": (
        "Evaluation_Set/Evaluation_Set/Validation",
        "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv",
    ),
    "test": (
        "Test_Set/Test_Set/Test",
        "Test_Set/Test_Set/RFMiD_Testing_Labels.csv",
    ),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.15, 0.15, 0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RFMiDDataset(Dataset):
    def __init__(self, data_root: str, split: str):
        img_dir, csv_path = SPLITS[split]
        self.img_dir = Path(data_root) / img_dir
        self.df = pd.read_csv(Path(data_root) / csv_path)
        self.tfm = build_transforms(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{int(row['ID'])}.png"
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor([float(row[c]) for c in DISEASE_COLS], dtype=torch.float32)
        return x, y


def compute_pos_weight(data_root: str) -> torch.Tensor:
    df = pd.read_csv(Path(data_root) / SPLITS["train"][1])
    pos = df[DISEASE_COLS].sum().values
    neg = len(df) - pos
    weight = neg / (pos + 1.0)
    return torch.tensor(weight, dtype=torch.float32)


def label_distribution(data_root: str, split: str = "train") -> pd.DataFrame:
    df = pd.read_csv(Path(data_root) / SPLITS[split][1])
    counts = df[DISEASE_COLS].sum().astype(int)
    return pd.DataFrame({
        "code": counts.index,
        "name": [DISEASE_NAMES[c] for c in counts.index],
        "count": counts.values,
    }).sort_values("count", ascending=False).reset_index(drop=True)
