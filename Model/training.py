from sys import argv
from pathlib import Path
from torch import device, cuda
from torch.utils.data import DataLoader
from dataset_tools import Img_Dataset

from Model.models.SRGAN import srgan as model

IMAGE_PATH = Path(__file__).parents[1] / 'Images'
MODEL_PATH = Path(__file__).parent / "models"
# Model parameter
B, H, W = 10, 128, 128
SUBSIZE = 0   # divide gen_input(survey img) size by 2^n
NB = 1024*2

DATASET_ON_GPU = True
if DATASET_ON_GPU:
    dev = device("cuda" if (cuda.is_available()) else "cpu")
    pin_mem = False
else:
    dev = device("cpu")
    pin_mem = True


def training(nb=NB, batch=B, h=H, w=W, subsize=SUBSIZE):
    training_data = Img_Dataset(n=nb, height=h, width=w, subsize=SUBSIZE, data_path=IMAGE_PATH, device=dev)
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, pin_memory=pin_mem)
    g_loss, d_loss = model.train(train_dataloader, load=False, save=True, sub=SUBSIZE, show_example=True)

def test_sample_gen():
    pass

def time_benchmarking():
    pass


if __name__ == '__main__':
    # Rework that stuff
    if len(argv) > 1:
        if argv[0].lower() in [f.name.lower() for f in MODEL_PATH.iterdir()]:
            training(*argv)
        else:
            print("No model found check in the 'models' folder")
    else:
        print('Param are in order: number_of_img batch_size height width subsize')



