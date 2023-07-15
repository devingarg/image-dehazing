import glob
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms


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
    def __init__(self, parent):
        
        self.fpaths = glob.glob(os.path.join(parent, "*.jpg"))
        self.gtpaths = [f.replace("hazy", "GT") for f in self.fpaths]
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(400),
                                        transforms.CenterCrop(400),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])          
                                    ])

        self.target_transform = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize(400),
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

def get_dataloaders(dataset):
    if dataset == "ohaze":
        
        data_dir = f"datasets/{dataset}/hazy"
        dataset = OHazeDataset(data_dir)

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        SPLIT = 0.8
        BATCH_SIZE = 8
        train_indices = indices[:int(SPLIT*len(dataset))]
        test_indices = indices[int(SPLIT*len(dataset)):]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False,
                                  num_workers=4, drop_last=False, persistent_workers=True)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, shuffle=False,
                                 drop_last=False)

#         print(f"Batch size: {BATCH_SIZE}")
#         print(f"Number of batches per epoch in train set: {len(train_loader)}")
#         print(f"Number of batches in test set: {len(test_loader)}")

    elif dataset.startswith("dh/"):
        
        data_dir = f"datasets/{dataset}/hazy"
        dataset = DHazyDataset(data_dir)

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        SPLIT = 0.8
        BATCH_SIZE = 8
        train_indices = indices[:int(SPLIT*len(dataset))]
        test_indices = indices[int(SPLIT*len(dataset)):]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False,
                                  num_workers=4, drop_last=False, persistent_workers=True)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, shuffle=False,
                                 drop_last=False)
    
    return train_loader, test_loader