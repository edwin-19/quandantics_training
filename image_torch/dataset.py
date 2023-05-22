from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_dataset, labels, transforms=None) -> None:
        super(ImageDataset, self).__init__()
        self.image_dataset = image_dataset
        self.labels = labels
        self.transforms = transforms
        
        self.label_map = {
            'broken': 0,
            'defect': 1,
            'good': 2
        }
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index):
        image = self.image_dataset[index]
        label = self.labels[index]
        
        img = Image.open(image).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
            
        return {
            'img': img,
            'label': self.label_map[label]
        }