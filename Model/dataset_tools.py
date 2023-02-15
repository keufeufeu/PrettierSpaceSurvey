import json
import numpy as np
from pathlib import Path
from torch import float32, device, cuda
from torchvision.io import read_image
from torchvision.transforms.functional import crop
from torchvision.transforms import Resize, InterpolationMode
from torch.utils.data import Dataset

IMAGE_PATH = Path(__file__).parents[1] / 'Images'
METADATA = json.load(open(IMAGE_PATH / 'meta_data.json'))
device = device("cuda" if (cuda.is_available()) else "cpu")

class Img_Dataset(Dataset):
    def __init__(self, n, height=256, width=256, subsize=0, data_path=IMAGE_PATH, device=device):
        self.n = n
        self.data_path = Path(data_path)
        self.metadata = json.load(open(data_path / 'meta_data.json'))
        self.listid = self.random_ids()
        self.height = height
        self.width = width
        self.subsize = subsize
        self.device = device

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        im_survey, im_obs = get_croped( id=self.listid[index], 
                                        height=self.height, width=self.width, subsize=self.subsize, 
                                        data_path=self.data_path, metadata=self.metadata, device=self.device)
        return (im_survey, im_obs)

    def random_ids(self):
        listid = np.random.choice(list(self.metadata.keys()), self.n)
        return listid


def get_croped(id, height=256, width=256, subsize=0, data_path=IMAGE_PATH, metadata=METADATA, device=device):
    img_path = data_path / id
    # Select small or big image (all screen image have width = 1280)
    if height > metadata[id]['Size']['Screen_Height'] or width > 1280:
        im_height, im_width = (metadata[id]['Size']['Height'], metadata[id]['Size']['Width'])
        listimg = list(img_path.glob('*large.jpg'))
        if len(listimg) == 0:
            listimg = list(img_path.glob('*publication.jpg'))
    else:
        im_height, im_width = (metadata[id]['Size']['Screen_Height'], metadata[id]['Size']['Screen_Width'])
        listimg = list(img_path.glob('*screen.jpg'))

    if len(listimg) < 2:
        raise FileNotFoundError(f'Some files are missing in {id} folder')
    # Random choose dss or sdss, and select obs file
    survey_file = np.random.choice([file for file in listimg if file.name.find('dss') > 0])
    obs_file = [file for file in listimg if file.name.find('obs') > 0][0]

    # Find a random size that fit for croping
    if height > width:
        crop_height = np.random.randint(im_height // 2, im_height)
        crop_width = int(crop_height * (width / height))
        # Backup if not fit (when img_size ~= size_wanted)
        if crop_width > im_width:
            crop_height = np.random.randint(min((im_height, im_width)) // 2, min((im_height, im_width)))
            crop_width = int(crop_height * (width / height))
    else:
        crop_width = np.random.randint(im_width // 2, im_width)
        crop_height = int(crop_width * (height / width))
        if crop_height > im_height:
            crop_width = np.random.randint(min((im_height,im_width)) // 2, min((im_height,im_width)))
            crop_height = int(crop_width * (height / width))

    # Generate a top left position of the futur box to crop
    if abs(im_height - crop_height) == 0:
        top = 0
    else:
        top = np.random.randint(0, abs(im_height - crop_height))
    if abs(im_width - crop_width) == 0:
        left = 0
    else:
        left = np.random.randint(0, abs(im_width - crop_width))

    # Open the two files and apply same crop between them
    im_obs = read_image(str(obs_file)).to(device)
    im_obs = crop(im_obs, top, left, crop_height, crop_width)
    im_obs = Resize(size=(height, width), interpolation=InterpolationMode.BICUBIC, antialias=True)(im_obs)

    im_survey = read_image(str(survey_file)).to(device)
    im_survey = crop(im_survey, top, left, crop_height, crop_width)
    if subsize > 0:
        height = height // (2**subsize)
        width = width // (2**subsize)
    im_survey = Resize(size=(height, width), interpolation=InterpolationMode.BICUBIC, antialias=True)(im_survey)

    return im_survey.to(dtype=float32) / 255., im_obs.to(dtype=float32) / 255.
