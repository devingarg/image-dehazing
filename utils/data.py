import glob
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from pprint import pprint

class OHazeDataset(Dataset):
    def __init__(self, parent):
        
        self.fpaths = glob.glob(os.path.join(parent, "*.jpg"))
        self.gtpaths = [f.replace("hazy", "GT") for f in self.fpaths]
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop(400),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         
                                       ])

        self.target_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.CenterCrop(400),
                                                ])
        

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.fpaths[idx]))/255
        target = np.array(Image.open(self.gtpaths[idx]))/255
        
        # apply the transformations
        image = self.transform(image.astype("float32"))
        target = self.target_transform(target.astype("float32"))
        
        return image, target
    
class DHazyDataset(Dataset):
    def __init__(self, parent, tmap_gt = False):
        
        
        if "Middlebury" in parent:
            self.fpaths = glob.glob(os.path.join(parent, "*.bmp"))
            self.gtpaths = []
            for f in self.fpaths:
                h, t = os.path.split(f)
                self.gtpaths.append(os.path.join(h.replace("hazy", "GT"), t.replace("hazy.bmp", "im0.png")))

        elif "NYU" in parent:
            self.fpaths = glob.glob(os.path.join(parent, "*.bmp"))

            self.gtpaths = []
            for f in self.fpaths:
                h, t = os.path.split(f)
                if tmap_gt:
                    self.gtpaths.append(os.path.join(h.replace("hazy", "GT"), t.replace("hazy", "Depth_")))
                else:
                    self.gtpaths.append(os.path.join(h.replace("hazy", "GT"), t.replace("hazy", "Image_")))
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(400, antialias=True),
                                        transforms.CenterCrop(400),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])          
                                    ])

        self.target_transform = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize(400, antialias=True),
                                               transforms.CenterCrop(400),
                                            ])
        

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.fpaths[idx]))/255
        target = np.array(Image.open(self.gtpaths[idx]))/255
        
        # apply the transformations
        image = self.transform(image.astype("float32"))
        target = self.target_transform(target.astype("float32"))
        
        return image, target

def get_dataloaders(dataset, train_split, batch_size, tmap_gt=False):
    
    data_dir = f"../datasets/{dataset}/hazy"

    if dataset == "ohaze":    
        dataset = OHazeDataset(data_dir)
    
    elif dataset.startswith("dh/"):
        if tmap_gt:
            assert dataset == "dh/NYU", "For tmap predictor, the only dataset possible is dh/NYU"
        dataset = DHazyDataset(data_dir, tmap_gt)
        
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_indices = indices[:int(train_split*len(dataset))]
    test_indices = indices[int(train_split*len(dataset)):]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False,
                              num_workers=4, drop_last=False, persistent_workers=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False,
                             drop_last=False)

    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch in train set: {len(train_loader)}")
    print(f"Number of batches in test set: {len(test_loader)}")

    return train_loader, test_loader