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
FPN_DIR = os.path.join(WEIGHT_DIR,'fpn_weights')
FPN_HEADER = 'fpn_weights_'


def loadWeight(model, label_name, device):
    if(model.getName() == 'ReconstructiveSubNetwork'):
        ae_file_name = AE_HEADER + label_name + '_.pckl'
        ae_load_path = os.path.join(AE_DIR,ae_file_name)
        if os.path.exists(ae_load_path):
            model.load_state_dict(torch.load(ae_load_path, map_location=device))
            print(f"AE: Pesi per la categoria '{label_name.upper()}' caricati correttamente!\n - {ae_load_path}")
        else:
            print(f"Errore: File dei pesi non trovati.\n{ae_load_path}\n")
    
    if(model.getName() == 'FPNNetwork'):
        fpn_file_name = FPN_HEADER + label_name + '.pth'
        fpn_load_path = os.path.join(FPN_DIR,fpn_file_name)
        if os.path.exists(fpn_load_path):
            model.load_state_dict(torch.load(fpn_load_path, map_location=device))
            print(f"FPN: Pesi per la categoria '{label_name.upper()}' caricati correttamente!\n - {fpn_load_path}")
        else:
            print(f"FPN: Errore: File dei pesi non trovati. Verifica i percorsi:\n - {fpn_load_path}\n")