import torch.nn as nn
import torch
import os
from config import PARENT_DIR, checkPath

#formula rivisitata del paper
def weightInit(layer):
    #Si trova nel layer convoluzionale
    if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias.data, 0.0)


WEIGHT_DIR = os.path.join(PARENT_DIR, 'weights')

#AUTOENCODER
AE_DIR = os.path.join(WEIGHT_DIR,'paper_weights')
AE_HEADER = 'DRAEM_seg_large_ae_large_0.0001_800_bs8_'
#UNET
UNET_DIR = os.path.join(WEIGHT_DIR,'unet_weights')
UNET_HEADER = 'unet_weights_'


def loadWeight(model, label_name, device):
    if(model.getName() == 'ReconstructiveSubNetwork'):
        ae_file_name = AE_HEADER + label_name + '_.pckl'
        ae_load_path = os.path.join(AE_DIR,ae_file_name)
        if os.path.exists(ae_load_path):
            model.load_state_dict(torch.load(ae_load_path, map_location=device))
            print(f"AE: Pesi per la categoria '{label_name.upper()}' caricati correttamente!\n - {ae_load_path}")
        else:
            print(f"Errore: File dei pesi non trovati.\n{ae_load_path}\n")
    
    if(model.getName() == 'DiscriminativeSubNetwork'):
        unet_file_name = UNET_HEADER + label_name + '.pth'
        unet_load_path = os.path.join(UNET_DIR,unet_file_name)
        if os.path.exists(unet_load_path):
            model.load_state_dict(torch.load(unet_load_path, map_location=device))
            print(f"UNET: Pesi per la categoria '{label_name.upper()}' caricati correttamente!\n - {unet_load_path}")
        else:
            print(f"UNET: Errore: File dei pesi non trovati. Verifica i percorsi:\n - {unet_load_path}\n")