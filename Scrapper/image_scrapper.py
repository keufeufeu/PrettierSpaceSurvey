import json
import numpy as np
from sys import argv
from pathlib import Path
from PIL import Image
from img_get_tools import image_from_url, get_metadata, get_size, get_survey, save_image, remove_folder
from img_get_tools import is_white_img, is_gray_img, is_black_img, if_obs_black_zone, transpose_black_zones, adding_mask

# Launch the script with the terminal (it take some hours to run)
# As input the script will use .json file in the Scrapper folder, those files can be generate with metadata_scrapper.py
# It will download image from hubble/webb, and the same from survey like dss and sdss, with some filter.
# As output it will generate Images folder with:
#   meta_data.json that compile json input information
#   autoblacklist.json that contain id of observation unusable for training (survey or hubble/webb too bright, too dark, gray ...)
#   A lot of folder name by id, each folder should contain, one small and one big image respectively for hubble/webb and at least one survey.
# Only Hubble/Webb Outreach, DSS and SDSS color are suported.
# Futur upgrade: - Adding filter to survey image that are full of one color
#                - Adding support for black box part in hubble/webb => to add them on the survey image
#                - Save where DSS and SDSS unusable to not retry if rerun

# Option are:
# no flag => run normaly
# -o flag => run with force to re-download every image
# -m flag => download only missing folder



FOLDER_PATH = Path(__file__).parent
IMAGE_PATH = FOLDER_PATH.parent / 'Images'
MAXSIZE = 20000
DATA = get_metadata()
# remove pillow size limit for image size
Image.MAX_IMAGE_PIXELS = None


def run(override=False, reset_blacklist=False):
    IMAGE_PATH.mkdir(exist_ok=True)
    (IMAGE_PATH / 'autoblacklist.json').touch(exist_ok=True)
    if reset_blacklist or (IMAGE_PATH / 'autoblacklist.json').stat().st_size == 0:
        write_autoblacklist([])
    blacklist = set(json.load(open(IMAGE_PATH / 'autoblacklist.json')))
    nstart = len(list(IMAGE_PATH.glob('**/*.jpg')))
    listid = set(DATA.keys()) - blacklist
    n = len(listid)
    c = 0
    for i in listid:
        c += 1
        print(f'Getting {i} | Progress: {c} / {n}')
        aborted = dl_img(i, override)
        # If connection problem persist, aborted will be False, folder will be remove, but not added to the blacklist.
        if aborted:
            remove_folder(i, IMAGE_PATH)
            blacklist.add(i)
            write_autoblacklist(list(blacklist))
    update_metadata()
    nfinal = len(list(IMAGE_PATH.glob('**/*.jpg')))
    print(f'Img added: {nfinal - nstart} | Img: {nfinal}')
    verification_database(retry_missing=True)


def dl_img(id, override=False):
    publication = False
    folder = IMAGE_PATH / id
    width_large, height_large = DATA[id]['Size']['Width'], DATA[id]['Size']['Height']
    # Image that are too large for being process can switch to a lower format called "publication"
    if width_large > MAXSIZE or height_large > MAXSIZE:
        publication = True
    # if abort folder is delete after
    folder.mkdir(exist_ok=True)
    # Saving the first hubble/webb image check if it's gray image then blacklist the id
    abort, screen_size, screen_mask  = obs_screen(id, override=override)
    if abort:
        return True
    if screen_size is None:
        return False
    # DSS is a survey with low resolution and some low fov can be so bright the image is white and so unuable for training
    cancel_dss = survey_screen(id, 'dss', size=screen_size, mask=screen_mask, override=override)
    # SDSS is a limited survey some observation will be unusable, not downloading them
    cancel_sdss = survey_screen(id, 'sdss', size=screen_size, mask=screen_mask, override=override)
    # abort the block because nor dss or sdss are usable, no training possible
    if cancel_dss and cancel_sdss:
        return True
    # Download the big sized images
    size_big, big_mask = obs_big(id, publication, screen_mask, override)
    if size_big is None:
        return False
    if not cancel_dss:
        survey_big(id, 'dss', publication, size=size_big, mask=big_mask, override=override)
    if not cancel_sdss:
        survey_big(id, 'sdss', publication, size=size_big, mask=big_mask, override=override)
    return False


def obs_screen(id, override=False):
    path_file = IMAGE_PATH / id / (id + '_obs_screen.jpg')
    if (not path_file.exists()) or override:
        img = image_from_url(DATA[id]['Img_url']['Screensize'])
        size = get_size(img)
        # Filter gray image
        if is_gray_img(img):
            return True, size, None
        else:
            save_image(img, path_file)
    else:
        img = Image.open(path_file)
        size = get_size(img)
    if if_obs_black_zone(img):
        mask = transpose_black_zones(img)
    else:
        mask = None
    return False, size, mask


def obs_big(id, publication, screen_mask=None, override=False):
    if publication:
        format = 'Publication'
    else:
        format = 'Large'
    path_file = IMAGE_PATH / id / (f"{id}_obs_{format.lower()}.jpg")
    if (not path_file.exists()) or override:
        img = image_from_url(DATA[id]['Img_url'][format])
        size = get_size(img)
        save_image(img, path_file)
    else:
        img = Image.open(path_file)
        size = get_size(img)
    if screen_mask is not None:
        mask = transpose_black_zones(img)
    else:
        mask = None
    return size, mask


def survey_screen(id, survey, size, mask=None, override=False):
    path_file = IMAGE_PATH / id / (f"{id}_{survey}_screen.jpg")
    if (not path_file.exists()) or override:
        info = DATA[id]
        img = get_survey(
            ra_d=info['Position']['RA_d'], 
            dec_d=info['Position']['Dec_d'],
            width=size[0],
            height=size[1],
            fov_w=info['FOV']['Width'],
            fov_h=info['FOV']['Height'],
            orientation=info['Orientation'],
            survey=survey)
        # Filter full black image (sdss out of bound)
        if survey == 'sdss':
            if is_black_img(img):
                return True
        # Filter too bright image
        if is_white_img(img):
            return True
        if mask is not None:
            img = adding_mask(img, mask)
        save_image(img, path_file)
    return False


def survey_big(id, survey, publication, size, mask=None, override=False):
    if publication:
        format = 'Publication'
    else:
        format = 'Large'
    path_file = IMAGE_PATH / id / (f"{id}_{survey}_{format.lower()}.jpg")
    if (not path_file.exists()) or override:
        info = DATA[id]
        img = get_survey(
            ra_d=info['Position']['RA_d'], 
            dec_d=info['Position']['Dec_d'],
            width=size[0],
            height=size[1],
            fov_w=info['FOV']['Width'],
            fov_h=info['FOV']['Height'],
            orientation=info['Orientation'],
            survey=survey)
        if mask is not None:
            img = adding_mask(img, mask)
        save_image(img, path_file)


def update_metadata():
    # Check id and folder, change metadata size if switch from large img to publication img, and adding screensize size in metadata
    to_pop = []
    for id in DATA.keys():
        folder = IMAGE_PATH / id
        if folder.exists():
            if is_valid_folder(folder):
                w_screen, h_screen = get_size(Image.open(folder / (id + '_obs_screen.jpg')))
                f = folder / (id + '_obs_publication.jpg')
                if f.exists():
                    w_big, h_big = get_size(Image.open(f))
                else:
                    f = folder / (id + '_obs_large.jpg')
                    w_big, h_big = get_size(Image.open(f))
                size_dic = {
                    'Width': w_big,
                    'Height': h_big,
                    'Screen_Width': w_screen,
                    'Screen_Height': h_screen
                    }
                DATA[id]['Size'].update(size_dic)
            else:
                # if folder not valid, remove it for retry, not poping the id
                remove_folder(id, IMAGE_PATH)
        else:
            to_pop.append(id)
    for id in to_pop:
        DATA.pop(id)
    # Saving updated metadata in one file in Image folder
    with open(IMAGE_PATH / 'meta_data.json', 'w') as f:
        json.dump(DATA, f)
    gen_credit()
    # Removing folders not in meta_data
    for f in IMAGE_PATH.iterdir():
        if f.is_dir() and (f.name not in DATA.keys()):
            remove_folder(f.name, IMAGE_PATH)


def is_valid_folder(folder):
    # A valid folder, should have at least one survey and one observatory set of two image one screen and one large or publication
    dss_screen = (folder / f"{folder.name}_dss_screen.jpg").exists()
    dss_large = (folder / f"{folder.name}_dss_large.jpg").exists() or (folder / f"{folder.name}_dss_publication.jpg").exists()
    sdss_screen = (folder / f"{folder.name}_sdss_screen.jpg").exists()
    sdss_large = (folder / f"{folder.name}_sdss_large.jpg").exists() or (folder / f"{folder.name}_sdss_publication.jpg").exists()
    obs_screen = (folder / f"{folder.name}_obs_screen.jpg").exists()
    obs_large = (folder / f"{folder.name}_obs_large.jpg").exists() or (folder / f"{folder.name}_obs_publication.jpg").exists()
    error = 0
    # Check for partner image missing
    if (dss_screen ^ dss_large) or (sdss_screen ^ sdss_large) or (obs_screen ^ obs_large):
        return False
    # Check for at least one survey and observatory 
    if (dss_screen or sdss_screen) and obs_screen:
        return True
    else:
        return False


def write_autoblacklist(blacklist):
    with open(IMAGE_PATH / 'autoblacklist.json', 'w') as f:
        json.dump(blacklist, f)


def verification_database(retry_missing=False):
    # Since update_metadata() will check in the folder and remove it if some image missing then it just need to check by id and folder
    print('Verification of the database:')
    folder_id = [p.name for p in Path(IMAGE_PATH).iterdir() if p.is_dir()]
    json_id = list(json.load(open(IMAGE_PATH / 'meta_data.json')).keys())
    n_missing = abs(len(json_id) - len(folder_id))
    print(f'Folders founds: {len(folder_id)} | Id: {len(json_id)} | Missing: {n_missing}')
    if n_missing > 0 and retry_missing:
        missing_set = set(json_id) - set(folder_id)
        for id in missing_set:
            print('Getting missing:', id)
            dl_img(id, False)
        new_n_missing = abs(len(json_id) - len(folder_id))
        print('Still missing:', new_n_missing)
        missing_set = set(json_id) - set(folder_id)
        if new_n_missing != 0:
            print(list(missing_set))
    else:
        print('No missing Found')


def gen_credit():
    credit_file = FOLDER_PATH.parent / 'CreditESA.txt'
    credit_file.touch(exist_ok=True)
    last_data = json.load(open(IMAGE_PATH / 'meta_data.json'))
    sorted_id = sorted(list(last_data.keys()))
    list_cred = []
    for i in sorted_id:
        cred = last_data[i]['Credit']
        cred = cred.replace('Acknowledgement', ' Acknowledgement').replace('  ', ' ')
        cred = f"{i} : {cred}"
        list_cred.append(cred)
    with open(credit_file, 'w') as f:
        for item in list_cred:
            f.write(item + "\n")


if __name__ == '__main__':
    if len(argv) > 1:
        for arg in argv[1:]:
            if arg in ['override', '-o']:
                run(override=True)
            if arg in ['missing', '-m']:
                update_metadata()
                verification_database(True)
    else:    
        run()