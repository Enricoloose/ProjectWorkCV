from torchvision import datasets, transforms
from torch.utils.data import Dataset
import glob
from PIL import Image
import os
from config import *

class RGB_to_BGR(object):
    def __call__(self, tensor):
        return tensor[[2, 1, 0], ...]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    RGB_to_BGR()
])

def getTrainDatasets(root_dir, labels, transform):
    
    dataset_list = []
        
    for label in labels:
        train_directory = os.path.join(root_dir, label, 'train')
    
        if os.path.exists(train_directory):
            dataset = datasets.ImageFolder(root=train_directory, transform=transform)
            dataset_list.append((label,dataset))
    
    return dataset_list


#Describable Textures Dataset (DTD)
#Dataset utilizzato per prendere le texture di rumore
texture_dataset = datasets.ImageFolder(root=DTD_DIR, transform=transform)


