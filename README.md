# Prettier Space survey 🌌
This repo is a student project design to train an AI to upgrade image from survey like DSS or SDSS, into image like Hubble or Webb outreach.
The absolute goal is to point somewhere in the sky, and produce an halucination of what hubble/webb could see.

Main purpose: Learn deep learning stuff, and test some models

## Scrapping part:
Colored image of Hubble or Webb are actualy artist work, NASA provide multiples images for each wavelenth on one observation, and artist combine them with an association of color. Those images are call Outreach images. 

The way i use to get the images and data associated is to scrap website esahubble.org and esawebb.org
Second to this part is to get the associated image of survey, i use Alasky API from University of Strasbourg with the metadata previously scrapped.

First scrap metadata of hubble and webb observation:
```
python3 Scrapper/metadata_scrapper.py hubble webb
```

Then download actual image of previous observation and survey counterpart (actualy dss and sdss)
```
python3 Scrapper/image_scrapper.py
```
This can be run multiple times without redownloading every image, but it redownloaded the previously rejected image, still long.

Flag:
- -o overide, force redownload correct image
- -m missing, execute only the validation script with retry download for missing

## Model part:
I wrote the first version with tensorflow, this is the second version with pytorch.
The first model, was custom version of SRGAN train with a 6Gb 1060 laptop for few days so keep you're expectation low. 

### Status
- SRGAN pytorch: (WIP) Try to reproduce the first SRGAN like model in pytorch, not learning, imbalance generator/discriminator
- Resolution Diffusion: Later
- Use of pretrainmodel: TBA

Cf training.ipynb

-------------------------
## Copyright:
All image use by this project is subject to some copyright (mostly CC BY 4.0), the detail of this can be found in the Credit.md file.

-------------------------
## Futur upgrade possible:
Integration of more telescope in both survey and observatory side can be made.

Some improvement on the scrapper / filter / selection of training data could be made:
- Fix rare misalign between survey and obs (wrong coordinate are present in ESA site) at least blacklist them.
- Better filter for obs with black box (actual version kinda works with some weird case that are manualy blacklisted and some edge case i choose to keep cause good enough).
- Better solution for one color type survey image, actual filter too permissive.

On the deep learning part:
- Better data augmentation for training (with this dataset i suppose random crop / resize is all it's needed but maybe it can be better)
- Better batch normalisation (Using high memory GPU if nothing works)
- Correction of padding artefact 
- Do math to understand whats wrong
- Try new models
    - Resolution diffusion model
    - Transfert learning with pre-train model
    - etc

