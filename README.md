# **MAPS** (**Ma**rtian **P**it **S**hadow Extractor)
## Introduction
MAPS is an automated Python framework which employs K-Means clustering to extract the shadow from a cropped red-band Mars Reconnaissance Orbiter HiRISE image of a Martian pit. MAPS also uses the sensing information from each image in order to determine the apparent depth (h) of the pit. h is defined as the depth at the point where the shadow extends into the interior of the pit. MAPS is intended to be used as a post-processing tool after Martian pits have been automatically detected, perhaps through the use of machine learning. While MAPS is currently specific only to the HiRISE sensor, there is scope for the method to be applied to data from other sensors and other planetary surfaces.

## Code
This repository contains three scripts: `run_MAPS.py`, `analyse_HiRISE_image.py`, and `analyse_HiRISE_DTM.py`:

`run_MAPS.py`:
- This script compiles all of the necessary functions in the correct order, in order to carry out the method of the MAPS tool and save the necessary outputs. `run_MAPS.py` takes nine variables as input:, `MODE`, `CROPPING`, `INPUT_DIR`, `METADATA_DIR`, `LABELS_DIR`, `OUTPUT_DIR` , `DTM_DIR`, `ORTHO_DIR`, and `N_CLUSTERS`.
- `MODE` **SHOULD NOT** be change from `rdr` as this means that MAPS will only look for HiRISE Reduced Data Record (RDR) images to analyse.
- `CROPPING` should be set to a boolean value based on whether or not the user needs MAPS to crop images for them. If `CROPPPING = True`, then the user will need to provide ESRI-Shapefile labels for each pit, in order for MAPS to crop the larger HiRISE image down. Whereas if `CROPPING = False` then these labels are not needed.
- All of the variables ending in `_DIR` define the directories to all of the input files and **SHOULD NOT** be changed.
- `N_CLUSTERS` is a list of integers which the K-Means clustering function `sklearn.cluster.KMeans` will segment the image into. This also **SHOULD NOT** be changed.

`analyse_HiRISE_image.py`
- All of the functions for reading and cropping HiRISE RDR images, as well as performing the shadow extraction and apparent depth calculation, are contained in this script.

`analyse_HiRISE_DTM.py`
- This script was only used for development of the MAPS tool, and is of no relevance to the user.

## Usage
1. Clone the repository
> `git clone https://github.com/dlecorre387/Sentinel2ImageDownload.git`

2. Copy or move your input files into the correct files
> All input images (cropped or uncropped) should be placed in the /data/input/ folder. If the input images are not cropped, then the necessary labels should be copied to the /data/labels/ folder. The metadata .LBL files for each HiRISE image should be placed in the /data/metadata/ folder.

3. Build the docker image from within the cloned repository
> `docker build -t maps .`

4. Run the docker container
> `docker run -it maps`

5. Run the MAPS tool
> The current working directory is already set to /app, but in case this changes, cd back into /app where `run_MAPS.py` is stored. Then run `python run_MAPS.py` from the command line.

## Further reading
More information about the method of the MAPS tool, as well as the results from preliminary testing, can be found in the 'MAPS_Write_Up' PDF file.
