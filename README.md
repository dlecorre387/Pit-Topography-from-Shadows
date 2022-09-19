# **MAPS** (**Ma**rtian **P**it **S**hadow Extractor)
## Introduction
MAPS is an automated Python framework which employs K-Means clustering and Silhouette Analysis to extract the shadow from a cropped red-band Mars Reconnaissance Orbiter HiRISE (High Resolution Imaging Experiment) image of a Martian pit.

MAPS also uses the sensing information from each image in order to determine the apparent depth (h) of the pit. h is defined as the depth at the point where the shadow extends into the interior of the pit.

MAPS is intended to be used as a post-processing tool after significant numbers of Martian pits have either been automatically detected, perhaps through the use of machine learning, or manually catalogued.

While MAPS is currently specific only to the HiRISE sensor, there is scope for the method to be applied to data from other sensors and other planetary surfaces.

## Link to MAPS video presentation
Coming soon! However, the slides are already available to view. Check out the file `MAPS_Slides.pptx`

## Requirements
- Python (version 3.x)
- Docker (version 4.8.x or above)
- Visual Studio Code along with the 'Docker' extension is highly recommended to be able to manage files within the docker container.

## Code
This repository contains two scripts: `MAPS_functions.py` and `run_MAPS.py`:

`MAPS_functions.py` includes all of the functions that are required for MAPS to read the user-inputted raster images and sensing information, to automatically extract the shadow via K-means clustering, and to calculate the apparent depths of pits. These functions have no user input and should not be altered.

This script calls all of the necessary functions from `MAPS_functions.py` in the correct order, in order to carry out the method of the MAPS tool and save the necessary outputs. `run_MAPS.py` takes nine variables as input, two of which are optional and 7 which should not be changed.

Optional parameters in `run_MAPS.py` are:
- `CROPPING`: Are you providing MAPS with pit location polygon ESRI shapefile labels to crop larger HiRISE images to the extents of the pit feature? Type: boolean.
- `TRAINING`: Are you providing ESRI shapefile polygon labels of the main pit shadow(s) in your image(s) in order to test the accuracy of MAPS' shadow extraction? Type: boolean.

Mandatory parameters are (DO NOT CHANGE THESE):
- `CLUSTERS`: The range of values for the number of K-means clusters (k). Type: list of int.
- `MISS_RATE` and `FD_RATE`: The average miss rate and false discovery rate of shadow pixel detections when previously testing MAPS on 20 labelled HiRISE images of Martian pits. These values will be overwritten by the new average rates as calculated on your training data if `TRAINING=True`. Types: float.
- `INPUT_DIR`, `METADATA_DIR`, `LABELS_DIR`, `TRAINING_DIR`, `OUTPUT_DIR`: The paths to the input (for raster images), metadata (sensing information .LBL files), labels (pit location shapefiles), training (labelled training data), and output (where results are stored) folders. Types: str

## Usage
1. Clone the repository
> `git clone https://github.com/dlecorre387/MartianPitShadowExtractor.git`

2. Copy or move your input files into the correct files
- All input images (cropped or uncropped) should be placed in the /data/input/ folder.
- If the input images are not cropped, then the necessary pit location labels should be copied to the /data/labels/ folder.
- The metadata .LBL files for each HiRISE image should be placed in the /data/metadata/ folder.
- If labels of the shadow(s) have been provided for testing MAPS, then they should be placed in the /data/training/shadows/ folder.
- NOTE: You can easily drag and drop new data into the relevant folders once the docker container has already been built by using the 'Docker' extension in Visual Studio Code.

3. Build the docker image from within the cloned repository
> `docker build -t maps .`
- NOTE: The full stop is essential here. Do not forget it.

4. Run the docker container
> `docker run -it maps`
- NOTE: This may take a while to run, don't worry!

5. Set your optional parameters
- Open the python script `run_MAPS.py` and set the optional parameters `CROPPING` and `TRAINING` accordingly (see 'Code' above). These can be found on lines 58 and 59, respectively.

5. Run the MAPS tool
- The current working directory is already set to /app, but in case this changes, cd back into /app where `run_MAPS.py` is stored. Then run the following line in the command line:
> `python run_MAPS.py`

6. Try using MAPS on the demo data that is already stored in the repository.
- /data/ and /data/training/ folders contain one image each, all with the necessary corresponding data. You can try running MAPS with `Training=True` to get performance scores based on the accuracy of MAPS' shadow extraction and `Training=False` to get results as if you don't have prior knowledge of the location of the shadow. The demo raster imagery that is already in the repository is not already cropped. Therefore, to use this demo data, please ensure `CROPPING=True`.
