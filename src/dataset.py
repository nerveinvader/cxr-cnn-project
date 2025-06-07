### src/dataset.py
### Dataset and Dataloader script
### Load data into PyTorch in usable batches

from pathlib import Path
import io
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

### Image Labels for Dataset
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion',
    'Infiltration','Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax','Consolidation', 'Edema', 'Emphysema',
    'Fibrosis','Pleural_Thickening', 'Hernia'
]

### Private Helper Method - Read CSV without Null Bytes
def safe_read_csv(path: str):
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    return pd.read_csv(
        io.BytesIO(raw), engine='python', encoding='latin1',
        on_bad_lines='skip', delimiter=',',
        quoting=1 # Quote All
    )

### ChestXRay14 Class
### Accepts Dataset as parameter
class ChestXRay14(Dataset):
    def __init__(self, img_dir: str, csv_file:str):
        #super().__init__()
        self.img_dir = Path(img_dir)
        raw_df = safe_read_csv(csv_file)
        # Frontal PA View Only
        df = raw_df[raw_df['View Position'] == 'PA'].reset_index(drop=True)
        self.paths = df['Image Index'].values
        # Split Finding Labels into multi-hot Tensor
        self.targets = []
        for labels in df['Finding Labels']:
            vec = torch.zeros(len(LABELS), dtype=torch.float32)
            if labels != 'No Finding':
                for lbl in labels.split('|'):
                    idx = LABELS.index(lbl)
                    vec[idx] = 1.
            self.targets.append(vec)
        self.targets = torch.stack(self.targets)
        ### Transformations
        ### Happening on the train.py file
        # self.tf = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225])
        # ])

    def __len__(self):
        return len(self.paths)
    ### Get Item (individual image)
    def __getitem__(self, idx):
        img_path = self.img_dir / self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.tf(img), self.targets[idx]
