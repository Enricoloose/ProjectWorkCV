from torchvision import datasets, transforms
import os
from config import *

#Funzione RGB_to_BGR perché l'encoder originale è stato addestrato 
#usando immagini in formato RGB
#utilizzando datasets.ImageFolder le immagini vengono prese in formato RGB
class RGB_to_BGR(object):
    def __call__(self, tensor):
        return tensor[[2, 1, 0], ...]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    RGB_to_BGR()
])

#Funzione per ottenere la lista di dataset di training
def getDatasets(root_dir, labels, transform):
    
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