# Prettier Space survey
This repo is a student project design to train an AI to upgrade image from survey like DSS or SDSS, into image like Hubble or Webb outreach.
The absolute goal is to point somewhere in the sky, and produce an halucination of what hubble or webb could see.

This project was born by doing some galaxy classification with galaxy zoo, and being frustrated by the lack of detail we have in most of the galaxy present in the sky.

Main utility i found, is to illustrate news with an artistic view. And just for the pleasure to see the incredible thing happening in space.
Since it's goal is to halucinate, maybe it can be usefull to find new target for future observation.

## Scrapping part
Colored image of Hubble or Webb are actualy artist work, NASA provide multiples images for each wavelenth on one observation, and artist combine them with an association of color. Those image are call Outreach image. 
A other approach i do not take, is to get source image from MAST, and combine them automaticly with color recommanded by NASA, but a whole set of problem could append for filtering and cleaning the dataset.

The best way i found to get the images and data associated is to scrap website like esahubble.org and esawebb.org

Second to this part is to get the associated image of survey, i use Alasky API from University of Strasbourg with the metadata previously scrapped.


## Training part:
I wrote the first version with tensorflow, this is the second version with pytorch.




## Copyright
All image use by this project is subject to some copyright, the detail of this can be found in the Credit.md file.

#### For Hubble / Webb:
Nasa base images are on public domain, but this project use colorized and combined version of them in provenance of ESA/Hubble and ESA/Webb website, their policy is "CC BY 4.0" therefore all credit should be available for each image. It should be available in the Credit.md file

Source: https://esahubble.org/copyright/   https://esawebb.org/copyright/

#### For HiPS:
CNRS/Unistra are providing the API use to point somewhere in the sky and collect the survey image corresponding in each observation of Hubble and Webb

#### For DSS:
Source: https://archive.stsci.edu/dss/copyright.html

#### For SDSS 9:
Source: https://skyserver.sdss.org/dr9/en/credits/



## Futur upgrade possible:
A lot of new source of data can be added, i supposed a lot more improvement can be added with the use of astropy, went i look (start of the project) i don't find proper way to extract what i need, but i know it's possible.

Some improvement on the scrapper / filter / selection of training data could be made:
    - Fix slight misalign between survey and obs (wrong coordinate are present in ESA site) or blacklist them
    - Better filter for obs with black box (actual version kinda works with some weird case that are manualy blacklisted and some edge case i choose to keep)
    - Better solution for one color type survey image, actual filter too permissive

On the deep learning part:
    - Better data augmentation for training (with this dataset i suppose random crop and resize is all it's needed but maybe it can be better)
    - Try with diffusion model (i'm working on it)
    - Better loss / hybrid system (like SRGAN that having VGGxx for eucledian distance loss in the feature map)
    - Better batch normalisation (First version was train with 6gb gpu and f64 precision in tensorflow so batch size = 1...)
    - Correction of padding artefact 
    

