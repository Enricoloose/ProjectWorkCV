import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

#Formula conversione per matplotlib
def tensorToImg(tensor, is_mask=False):
    img = tensor.cpu().clone().detach().float().numpy()

    
    if not is_mask:
        #Conversione in RGB solo per il plot
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = img[:, :, ::-1] 
        return img

    elif img.shape[0] == 2:
        return img[1, :, :]

    else:
        return img.squeeze()



def showImg(original_img, noisy_img, rec_img, gt_mask, out_mask, img_index):
    idx = 0

    plt.figure(figsize=(20, 4))
    plt.title(f"Immagine n° {img_index}")
    plt.axis('off')

    plt.subplot(1, 5, 1)
    plt.imshow(tensorToImg(original_img[idx]))
    plt.title("Originale")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(tensorToImg(noisy_img[idx]))
    plt.title("Difettosa")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(tensorToImg(rec_img[idx]))
    plt.title("Ricostruita (AE)")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(tensorToImg(gt_mask[idx], is_mask=True), cmap='jet', vmin=0, vmax=1)
    plt.title("Maschera Reale")
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(tensorToImg(out_mask[idx], is_mask=True), cmap='jet', vmin=0, vmax=1)
    plt.title("Predizione (U-Net)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()