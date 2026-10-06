import os                                   
import numpy as np                          
import torch                                 
from PIL import Image                        
from torch.utils.data import Dataset       

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DamageDataset(Dataset):
    def __init__(self, df, images_dir, targets_dir):
        self.base_names = df["base_name"].tolist()  
        self.images_dir = images_dir                 
        self.targets_dir = targets_dir          

    def __len__(self):
        return len(self.base_names)                

    def _load_image(self, path):
        image = np.array(Image.open(path))           
        image = image.astype(np.float32) / 255.0    
        image = (image - IMAGENET_MEAN) / IMAGENET_STD   
        image = np.transpose(image, (2, 0, 1))      
        return image

    def __getitem__(self, idx):
        base_name = self.base_names[idx]          

        pre = self._load_image(os.path.join(self.images_dir, f"{base_name}_pre_disaster.png"))
        post = self._load_image(os.path.join(self.images_dir, f"{base_name}_post_disaster.png"))
        mask = np.array(Image.open(os.path.join(self.targets_dir, f"{base_name}_post_disaster_target.png")))

        pre_tensor = torch.from_numpy(pre)         
        post_tensor = torch.from_numpy(post)
        mask_tensor = torch.from_numpy(mask).long() 

        return pre_tensor, post_tensor, mask_tensor