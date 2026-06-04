import os


def checkPath(path):
    if(os.path.exists(path)):
        print(path)
    else:
        print(f'Percorso: {path} non trovato')

CURRENT_DIR = os.getcwd()

#INIZIALIZZAZIONE DIRECTORY PER I DATASET
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(PARENT_DIR,'data')

data_name = 'MVTec1'
MVTEC_DIR = os.path.join(DATASET_DIR, data_name)
#checkPath(MVTEC_DIR)
dtd_name = 'dtd'
DTD_DIR = os.path.join(DATASET_DIR,dtd_name,'images')
#checkPath(DTD_DIR)

#Categorie da codificare
LABELS = ['leather', 'pill', 'tile', 'transistor', 'wood']


#Object mi serve per capire se l'immagine contiene un oggetto,
#e di conseguenza applicare il rumore di perlin
L_OBJECT = ['transistor']


