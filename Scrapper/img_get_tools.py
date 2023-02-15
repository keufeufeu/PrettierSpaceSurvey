import requests
import json
import numpy as np
from time import sleep
from io import BytesIO
from PIL import Image
from PIL.ImageOps import invert
from pathlib import Path
from math import sin, cos, radians
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.morphology import dilation
from skimage.transform import probabilistic_hough_line, rotate, resize, rescale


FOLDER_PATH = Path(__file__).parent
# Set if image transformation use Pillow or numpy / skimage (numpy slower and more conversion loss)
ARRAY = False


def get_metadata():
    # Create dictionnary from metadata files
    jsonfiles = list(FOLDER_PATH.glob('*_galaxy.json'))
    dic_ids = {}
    for f in jsonfiles:
        dic_ids = dic_ids | json.load(open(f))
    return dic_ids


def image_from_url(url, array=ARRAY):
    # Checking good connection for downloading image
    Image.MAX_IMAGE_PIXELS = None
    response = None
    timeout_sec = 300
    max_retry = 3
    for retry in range(1, max_retry + 1):
        try:
            response = requests.get(url, timeout=timeout_sec)
        except:
            if retry == 1:
                print(url)
            print(f'Retry connection: {retry} / {max_retry}')
            sleep(60)
            continue
        break
    if response is not None:
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if array:
                return np.array(img) / 255
            return img
    return None


def enlarge_your_galaxy(w, h, angle):
    # Return bigger Width and Height to make a rotation 
    if angle != 0:
        W = int(max(w,h) * (abs(sin(radians(angle))) + abs(cos(radians(angle))))) + 1
        return W, W
    else:
        return w, h


def get_size(img):
    if img is not None:
        if type(img) == np.ndarray:
            h, w, c = img.shape
        else:
            w, h = img.size
        return (w, h)
    else:
        return None


def delarge(img, new_w, new_h):
    old_w, old_h = get_size(img)
    # Crop image by the center, if new dim are smaller that image
    if new_h <= old_h and new_w <= old_w:
        left = int((old_w - new_w)/2)
        right = int((old_w - new_w)/2) + new_w
        up = int((old_h - new_h)/2)
        down = int((old_h - new_h)/2) + new_h
        if type(img) == np.ndarray:
            return img[up : down, left : right]
        else:
            return img.crop((left, up, right, down))
    else:
        return img


def outofbound(w, h):
    # Limit size if the futur rotated image is to big for alasky api
    max_pixel = 5000
    if w > max_pixel or h > max_pixel:
        ratio = h / w
        if w == h:
            w = max_pixel
            h = max_pixel
        elif ratio < 1:
            w = max_pixel
            h = int(max_pixel * ratio)
        else:
            h = max_pixel
            w = int(max_pixel / ratio)
    return w, h


def get_survey(ra_d=148.8882194, dec_d=+69.0652951,
               width=512,height=512,
               fov_w=25,fov_h=25,
               orientation=0, survey='dss'):
    # Predict bigger size for rotated image
    width_rot, height_rot = enlarge_your_galaxy(width, height, orientation)
    # Limit size if larger than API capacity
    width_limit, height_limit = outofbound(width_rot, height_rot)
    # FOV correction with the bigger axe
    if fov_w >= fov_h:
        fov = width_rot * (fov_w / 60) / width
    else:
        fov = height_rot * (fov_h / 60) / height
    # Other survey available can be found here: https://aladin.u-strasbg.fr/hips/list
    if survey == 'sdss':
        survey_hips = 'CDS/P/SDSS9/color'
    else: 
        survey_hips = 'CDS/P/DSS2/color'
    baseurl = 'https://alasky.u-strasbg.fr/hips-thumbnails/thumbnail?ra={}&dec={}&fov={}&width={}&height={}&hips_kw={}'
    imgurl = baseurl.format(ra_d, dec_d, fov, width_limit, height_limit, survey_hips)
    # Getting the image
    img = image_from_url(imgurl)
    if img is None:
        return None
    if orientation != 0:
        if type(img) == np.ndarray:
            img = rotate(img, orientation)
        else:
            img = img.rotate(orientation)
    # Resize to the larger size wanted if API limit reach
    if width_limit == 5000 or height_limit == 5000:
        if type(img) == np.ndarray:
            img = resize(img, [height_rot, width_rot], anti_aliasing=False)
        else:
            img = img.resize((width_rot, height_rot))
    # Cropping to the final size wanted
    img = delarge(img, width, height)
    return img


def wallp_resize(img, w, h):
    # Resize wallpaper type image to size wanted
    if type(img) == np.ndarray:
        scaling = max(h / img.shape[0], w / img.shape[1])
        img = rescale(img, scaling, channel_axis=2)
    else:
        scaling = max(h / img.height, w / img.width)
        img.reduce(scaling)
    return delarge(img, w, h)


def remove_folder(id, folder_path):
    folder = folder_path / id
    if folder.exists():
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()


def save_image(img, path, quality=90):
    if img is not None:
        if type(img) == np.ndarray:
            if img.dtype == 'float':
                img = (img * 255).astype('uint8')
            img = Image.fromarray(img)
        img.save(path, quality=quality, subsampling=0)


def is_gray_img(img):
    if img is not None:
        if type(img) != np.ndarray:
            img = np.array(img)
        # Found if all color are the same
        img = np.swapaxes(img, 0, 2)
        if np.array_equiv(img[0],[img[1],img[2]]):
            return True
        else:
            return False
    else:
        return False


def is_black_img(img):
    if img is not None:
        if type(img) != np.ndarray:
            img = np.array(img)
        # Efficient way to find if the image is full black
        if np.any(img) == False:
            return True
        # For image that have some black cut in it:
        # Create mask with full black pixel only and find the ratio of black pixel
        avg = np.mean(np.all(img != [0, 0, 0], axis=-1))
        # 2% of black pixel is the limit for non cutted image in sdss
        # Some cut can still be found but small enought to be acceptable
        if avg < 0.98:
            return True
        else:
            return False
    else:
        return False


def is_white_img(img):
    if img is not None:
        if type(img) != np.ndarray:
            img = np.array(img)
        if img.dtype == np.uint8:
            maximum = 255
        else:
            maximum = 1
        # Brightness too high is unusable 
        bright = img.mean() / maximum
        # Mean of standard deviation for each color near 0 show that the image is compose by the same color so unusable for training
        mean_std = np.mean([img[:,:,0].std(), img[:,:,1].std(), img[:,:,2].std()]) / maximum
        # Low mean_std can be acceptable if brightness is low
        cond = mean_std < 0.04 and bright > 0.2
        if bright > 0.90 or cond:
            return True
        else:
            return False
    else:
        return False


def if_obs_black_zone(img):
    if img is not None:
        if type(img) != np.ndarray:
            img = np.array(img)
        # Black zone in hubble image are due to collage or rotated image, so there are mostly with pixel very dark, 
        # and with some straights lines patterns, but all angle are possible
        mask = rgb2gray(img)
        maxh, maxw = mask.shape
        # Take the lowest pixels of the corners
        min_found = np.min((mask[0, 0], mask[maxh - 1, 0], mask[0, maxw - 1], mask[maxh - 1, maxw -1]))
        if min_found > 0.03:
            return False
        mask = mask > min_found
        # Delete holes in the mask
        mask = binary_fill_holes(mask)
        # Find the contour
        mask = canny(mask, 15)
        # Make the contour larger for having more lines detected
        for i in range(1):
            mask = dilation(mask)
        # Find some long line in the contour, if enough it's an image with black zone
        lines = probabilistic_hough_line(mask, threshold=100, line_length=150, line_gap=1)
        if len(lines) < 5:
            return False
        else:
            return True
    else:
        return False


def transpose_black_zones(img):
    if img is not None:
        if type(img) != np.ndarray:
            img = np.array(img)
        mask = rgb2gray(img)
        maxh, maxw = mask.shape
        # All image have a corner black, so find the lowest value in it
        min_found = np.min((mask[0, 0], mask[maxh - 1, 0], mask[0, maxw - 1], mask[maxh - 1, maxw -1]))
        mask = mask > min_found
        # Delete holes in the mask
        mask_temp = binary_fill_holes(mask)
        # Same with inverted black and white, and return to normal mask
        mask = binary_fill_holes(mask_temp == 0) == 0
        # Safe switch if the black zone completely surround the mask
        if np.any(mask) == False:
            mask = mask_temp
        # Return to 3 channel image for fusion with other image
        mask = np.stack((mask,)*3, axis=-1).astype('float')
        return mask
    else:
        return None


def adding_mask(img, mask):
    if img is not None and mask is not None:
        if type(img) != np.ndarray:
            mask = Image.fromarray((mask * 255).astype('uint8')).convert('1')
            img.paste(mask, mask = invert(mask))
        else:
            img = img * mask
    return img