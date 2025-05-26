### src/playground.py
### Playground Testing Script
### Test the Dataset and Dataloader

# %%
from torch.utils.data import DataLoader
from dataset import ChestXRay14
from torch.multiprocessing import freeze_support

#import os
#print("UNDER data/:", os.listdir("data/"))
#print("UNDER data/images:", os.listdir("data/images"))

# %%
if __name__ == "__main__":
    freeze_support()  # For Windows compatibility
    ds = ChestXRay14(img_dir='data/images', csv_file='cxr_csv/Data_Entry_2017.csv')
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

    x, y = next(iter(dl))

    print(x.shape)   # torch.Size([16, 3, 224, 224])
    print(y.shape)  # torch.Size([16, 14])
