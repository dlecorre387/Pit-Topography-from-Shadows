# The Pit Topography from Shadows (**PITS**) Tool
## Introduction
### What are pits?
Pits, or pit craters, are near-circular depressions found in planetary surfaces which are generally formed through gravitational collapse. Pits will be primary targets for future space exploration and habitability for their presence on most rocky Solar System surfaces and their potential to be entrances to sub-surface cavities. This is particularly true on Mars, where caves are thought to harbour stable reserves of water ice across much of the surface - on which astronauts will also be exposed to high radiation dosages. There are two main catalogues for pits: the [Mars Global Cave Candidate Catalog (MGC<sup>3</sup>)](https://astrogeology.usgs.gov/search/map/Mars/MarsCaveCatalog/mars_cave_catalog) and the [Lunar Pit Atlas](http://lroc.sese.asu.edu/pits/list). Since pits are rarely found to have corresponding high-resolution elevation data, tools are required for approximating their depths in order to find those which are the ideal candidates for exploration.

### How does PITS operate?

The Pit Topography from Shadows (PITS) tool is a dockerised Python framework which can automatically calculate a profile of the apparent depth (*h*) of Martian or Lunar pit from only one cropped, single- or multi-band, remote-sensing image. *h* is the relative depth of the pit between its rim and the edge of the shadow cast by the Sun - with the principle being that a deeper pit would cast a wider shadow.

PITS does this by employing image segmentation (in the form of unsupervised *k*-means clustering and silhouette analysis for automatic cluster suggestion) in order to produce a binary mask of shadow or non-shadow pixels. Then, by rotating the shadow mask by the Sun's azimuth angle relative to North ($\varphi$), PITS can measure the width of the shadow along the Sun's line of sight as observed by the satellite (*S<sub>obs</sub>*) at each pixel in the shadows length. *S<sub>obs</sub>* is then corrected for non-nadir observations to obtain the true shadow width (*S<sub>true</sub>*) as if the satellite was pointing straight downwards at the surface. *h* is then derived from these *S<sub>true</sub>* measurements by considering the incidence angle of the Sun ($\alpha$) for this particular image.

As well as the *h* profile, PITS saves the extents of the detected shadow as a geo-referenced ESRI shapefile for visualisation in GIS software such as QGIS. This can be used to enhance the contrast of the pixels within the shadow to search for any deeper-shaded regions - possibly due to a cave entrance.

PITS currently works with Mars Reconnaissance Orbiter (MRO) High Resolution Science Imaging Experiment (HiRISE) and Lunar Reconnaissance Orbiter (LRO) Narrow Angle Camera (NAC) imagery of Mars and the Moon, respectively. Despite these being the highest resolution sensors available, there are plans to expand the number of satellite whose data PITS can work with. PITS is intended to be used on known/catalogued pits, or as a post-processing tool after pits have been automatically detected perhaps through the use of Machine/Deep Learning.

### Testing Performance

Across 19 shadow-labelled MRO red-band HiRISE images of MGC<sup>3</sup> Atypical Pit Craters (APCs), PITS detected 99.6% of all shadow pixels (with 94.8% of all detections being true shadow pixels). This equates to an expected average F1 score of 97.1% when applying PITS to HiRISE red-band images. Testing upon 12 HiRISE colour images found that a small improvement (<2%) in F1 was achieved compared to performance upon their corresponding red-band versions. However, since the run-time of the PITS tool will increase by a factor of *n* for an *n*-band image, the recommendation is to use single-band imagery due to already high performance and lower run-time.

When applying PITS to 123 HiRISE images of 88 MGC<sup>3</sup> APCs, PITS exhibited a minimum and maximum run-time of roughly 5 and 500 s for images with 0.02 and 4.80 Mpx, respectively. This was achieved when using a laptop with an 11th generation Intel Core i5 processor and 8 GB of RAM.

## Requirements
- Python (version 3.x)
- Docker (version 4.8.x or above)
- Visual Studio Code along with the 'Docker' extension is highly recommended to be able to manage files within the docker container.

## Code
This repository contains three scripts: `PITS_functions.py`, `run_PITS.py` and `PITS_plotter.py`:

`PITS_functions.py`:

This includes all of the functions that are required for PITS to read the user-inputted raster images and sensing information, to automatically extract the shadow via *k*-means clustering, and to calculate the apparent depths of pits. These functions have no user input and should not be edited.

`run_PITS.py`:

This script calls all of the necessary functions from `PITS_functions.py` in the correct order, in order to carry out the method of the PITS tool and save the necessary outputs. `run_PITS.py` takes 5 user-inputted arguments (2 required, 3 optional) which are called in the command line.

Required arguments in `run_PITS.py` are:
- `-d` (`--dataset`):
  - The name of the dataset whose images will be used to calculate apparent depths. Currently supported options are `hirise-rdr` (for MRO HiRISE RDR version 1.1 images of Mars) and `lronac-edr` (for LRO NAC EDR images of the Moon). This is required since there is a different process for retrieving sensing information for each dataset. (Type: str)
- `-c` (`--cropping`/`--no-cropping`):
  - Crop each larger input image to the extents of the pit feature using user-provided ESRI shapefile rectangular labels of the pit's location. These shapefiles must include or be equal to the full product name of the corresponding image file, e.g. label_ESP_033342_1660_RED.shp for the HiRISE image ESP_033342_1660_RED.JP2. (Type: bool)

Optional arguments in `run_PITS.py` are:
- `-s` (`--shadows`/`--no-shadows`):
  - Save the aligned detected shadow in each image as a PDF file for viewing. This includes the binary shadow mask, but also the detected shadow edge and pit rim overlaid upon the input image to serve as a reference for where the shadow width was measured between. (Default: False / Type: bool)
- `-t` (`--testing`/`--no-testing`):
  - Calculate the precision, recall and F1 score of shadow pixel detections in each image using user-provided ESRI shapefile labels of the pit's shadow. (Default: False / Type: bool)
- `-f` (`--factor`):
  - The factor by which the cropped input image and labels will be down-scaled when calculating the silhouette coefficients during shadow extraction. (Default: 0.1 / Type: float)

`PITS_plotter.py`:

This script is for plotting the *h* profiles calculated across the entire imagery dataset provided to PITS. `PITS_plotter.py` takes just one optional argument which is called in the command line.

Optional arguments in `PITS_plotter.py` are:
- `-r` (`--raw`/`--no-raw`):
  - Plot the raw apparent depth measurements which have not been corrected for a non-zero satellite emission angle at the time when the image was taken. (Default: False / Type: bool)

## Usage
1. Clone the repository
> `git clone https://github.com/dlecorre387/Pit-Topography-from-Shadows.git`

2. Copy or move your input files into the correct folders (these folders will be copied when building the docker image in step 3):
  - All input images (cropped or uncropped) should be placed in the **/data/input/** folder.
  - If the input images are not cropped, then the necessary pit location labels should be placed in the **/data/labels/** folder.
  - PITS accesses all sensing metadata from the cumulative PDS3 index .TAB files, which should be placed in the **/data/metadata/** folder. These can be acquired from NASA's [Planetary Data System](https://pds.nasa.gov/) for the relevant dataset (e.g. RDRCUMINDEX.TAB for HiRISE RDR images). To improve run-time, these should be filtered to only contain the relevant rows for your input images. The BASH script provided in this repository (`filter_index_files.sh`) can do this for you by placing the index file in the same folder and running:
  >`bash filter_index_files.sh [path-to-imagery-folder]`.

  - If labels of the shadow(s) have been provided for testing PITS, then they should be placed in the **/data/testing/** folder.
  - **NOTE:** You can easily drag and drop new data into the relevant folders once the docker container has already been built by using the 'Docker' extension in Visual Studio Code.


3. Build the docker image from within the cloned repository. **NOTE:** The full stop is essential here.
> `docker build -t pits .`

4. Run the docker container. **NOTE:** This may take a while to run, don't worry!
> `docker run -it pits`

6. Run the PITS tool. The current working directory is already set to **/app/**, but in case this changes, cd back into **/app/** where `run_PITS.py` is found. Then run the following line in the terminal of the docker container. **NOTE:** This is what should be ran if PITS is to be applied to unlabelled HiRISE RDR images which also require cropping.
> `python run_PITS.py -d hirise-rdr -c`

7. Try using PITS on the demo data that is already stored in the repository:
  - The **/data/** folder contain the necessary labels and metadata files for one HiRISE image. Due to the file sizes, the HiRISE image is not given in the repository, but can be acquired through the Mars Orbital Data Explorer of NASA's Planetary Data System's [Geosciences Node](https://ode.rsl.wustl.edu/mars/). **NOTE:** The raster image products from the Mars ODE will not be cropped to the extents of the pit. Therefore, to run this demo please ensure you provide location labels and use `--cropping`.
  - After acquiring the image and placing in the **/data/input/** folder, you can run the above line to get a .CSV file of the *h* profile.
  - You can also try running PITS with `--testing` to get performance scores based on the accuracy of PITS' shadow extraction upon the HiRISE image ESP_033342_1660_RED.


8. Run the `PITS_plotter.py` script to plot the *h* profiles calculated for each image. To do this, without plotting the uncorrected *h* profile, run the following line in the terminal window.
> `python PITS_plotter.py`
