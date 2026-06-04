from weight import loadWeight
from model_ae import ReconstructiveSubNetwork
from model_unet import DiscriminativeSubNetwork
import torch
import random
from dataset import *
from config import *
from perlin import *
from plot import showImg

def init(device, label_name):

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=128)
    model.to(device)
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2, base_channels=64)
    model_seg.to(device)

    loadWeight(model, label_name, device)
    loadWeight(model_seg, label_name, device)

    model.eval()
    model_seg.eval()

    return model, model_seg

def main():
    if torch.xpu.is_available():
        device = 'xpu'
    else:
        device = 'cpu'

    print("using: ",device)

    dataset_list = getTrainDatasets(MVTEC_DIR, LABELS, transform)
    label_index = input("Seleziona la categoria: \n0 - Leather\n1 - Pill\n2 - Tile\n3 - Transistor\n4 - Wood\n")

    label_name, train_dataset = dataset_list[int(label_index)]
    model, model_seg = init(device, label_name)

    response = 'y'
    while(response=='y'):
        img_idx = random.randint(0, len(train_dataset)-1)
        text_idx = random.randint(0, len(texture_dataset)-1)

        original_img, _ = train_dataset[img_idx]
        texture_image, _ = texture_dataset[text_idx]

        original_img = original_img.unsqueeze(0).to(device)
        texture_image = texture_image.unsqueeze(0).to(device)

        #applicazione rumore 
        noisy_img, gt_mask = applyPerlinNoise(original_img, texture_image, label_name)
        noisy_img = noisy_img.to(device)
        gt_mask = gt_mask.to(device)

        rec_img = model(noisy_img)
        concat_img = torch.cat((rec_img, noisy_img), dim=1)
        out_mask = model_seg(concat_img)
        out_mask = torch.softmax(out_mask, dim=1)

        showImg(original_img, noisy_img, rec_img, gt_mask, out_mask, img_idx)

        response = input('Continuare?')

if __name__ == "__main__":
    main()