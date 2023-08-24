# The Pit Topography from Shadows (**PITS**) Tool
## Introduction
The Pit Topography from Shadows (**PITS**) tool is a dockerised Python framework which can automatically calculate an apparent depth (*h*) profile for a Martian or Lunar pit from just a single cropped satellite image. These images can also be single- or multi-band. *h* is the relative depth of the pit between its rim and the edge of the shadow cast by the Sun - with the principle being that a deeper pit would cast a wider shadow. These *h* profiles can be used to assess which pits are the best candidate cave entrances on the Moon and Mars. If you'd like to learn more about PITS, you can head to the research paper titled ["Automatically calculating the apparent depths of pits using the Pit Topography from Shadows (PITS) tool"](https://academic.oup.com/rasti/article/2/1/492/7241547) and published by the Royal Astronomical Society's Techniques and Instruments (RASTI) journal in August 2023.

### What are pits?
Pits, or pit craters, are near-circular depressions found in planetary surfaces which are generally formed through gravitational collapse. Pits will be primary targets for future space exploration and habitability for their presence on most rocky Solar System surfaces and their potential to be entrances to sub-surface cavities. This is particularly true on Mars, where caves are thought to harbour stable reserves of water ice across much of the surface - on which astronauts will also be exposed to high radiation dosages. There are two main catalogues for pits: the [Mars Global Cave Candidate Catalog (MGC<sup>3</sup>)](https://astrogeology.usgs.gov/search/map/Mars/MarsCaveCatalog/mars_cave_catalog) and the [Lunar Pit Atlas](http://lroc.sese.asu.edu/pits/list). Since pits are rarely found to have corresponding high-resolution elevation data, tools such as **PITS** are required for approximating their depths in order to find those which are the ideal candidates for exploration.

### How does **PITS** operate?
**PITS** works by employing image segmentation (in the form of unsupervised *k*-means clustering and silhouette analysis for automatic cluster suggestion) in order to produce a binary mask of shadow or non-shadow pixels. Then, by rotating the shadow mask by the Sun's azimuth angle relative to north ($\varphi$), **PITS** can measure the width of the shadow along the Sun's line of sight as observed by the satellite (*S<sub>obs</sub>*) at each pixel in the shadows length. *S<sub>obs</sub>* is then corrected for non-nadir observations to obtain the true shadow width (*S<sub>true</sub>*) as if the satellite was pointing straight downwards at the surface. *h* is then derived from these *S<sub>true</sub>* measurements by considering the incidence angle of the Sun ($\alpha$) for this particular image.

As well as the *h* profile, **PITS** saves the extents of the detected shadow as a geo-referenced ESRI shapefile for visualisation in GIS software such as QGIS. This can be used to enhance the contrast of the pixels within the shadow to search for any deeper-shaded regions - possibly due to a cave entrance.

**PITS** currently works with Mars Reconnaissance Orbiter (MRO) High Resolution Science Imaging Experiment (HiRISE) and Lunar Reconnaissance Orbiter (LRO) Narrow Angle Camera (NAC) imagery of Mars and the Moon, respectively. Despite these being the highest resolution sensors available, there are plans to expand the number of satellite whose data **PITS** can work with. **PITS** is well-positioned to be used on catalogued pits, or as a post-processing tool after pits have been automatically detected perhaps through the use of Machine/Deep Learning.

### Shadow Extraction Testing Performance
Across 19 shadow-labelled MRO red-band HiRISE images of MGC<sup>3</sup> Atypical Pit Craters (APCs), **PITS** detected 99.6% of all shadow pixels (with 94.8% of all detections being true shadow pixels). This equates to an expected average **F1 score of 97.1%** when applying **PITS** to HiRISE red-band images. Testing upon 12 HiRISE colour images found that a small improvement (< 2%) in F1 was achieved compared to performance upon their corresponding red-band versions. However, since the run-time of the **PITS** tool will increase by a factor of *n* for an *n*-band image, the recommendation is to use single-band imagery due to already high performance and lower run-time.

When applying **PITS** to 123 HiRISE images of 88 MGC<sup>3</sup> APCs, **PITS** exhibited a minimum and maximum run-time of roughly 5 and 500 s for images with 0.02 and 4.80 Mpx, respectively. This was achieved when using a laptop with an 11th generation Intel Core i5 processor and 8 GB of RAM.

## Requirements
- Python (version 3.x)
- Docker (version 4.8.x or above)
- Visual Studio Code along with the 'Docker' extension is highly recommended (if not using JupyterLab) to be able to manage files within the Docker container.

## Code
This repository contains three main scripts: [`PITS_functions.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/PITS_functions.py), [`run_PITS.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/run_PITS.py) and [`PITS_plotter.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/PITS_plotter.py). A tutorial in the form of a Jupyter notebook is also available in [`PITS_tutorial.ipynb`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/PITS_tutorial.ipynb).

### PITS_functions.py:

This includes all of the functions that are required for **PITS** to read the user-inputted raster images and sensing information, to automatically extract the shadow via *k*-means clustering, and to calculate the apparent depths of pits. These functions have no user input and should not be edited. The functions have also been grouped together into several Python classes according to their overall purposes, since many will require the same inputs.

### run_PITS.py:

This script calls all of the necessary functions from [`PITS_functions.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/PITS_functions.py) in the correct order, in order to carry out **PITS'** methodology and save the necessary outputs. [`run_PITS.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/run_PITS.py) takes six user-inputted arguments (two required, four optional) which are called in the command line.

Required arguments are:
- `-d` (`--dataset`):

  The name of the dataset whose images will be used to calculate apparent depths. Currently supported options are `hirise-rdr` (for MRO HiRISE RDR version 1.1 images of Mars) and `lronac-edr` (for LRO NAC EDR images of the Moon). This is required since there is a different process for retrieving sensing information for each dataset. **(Type:** str**)**

- `-c` (`--cropping`/`--no-cropping`):

  Crop each larger input image to the extents of the pit feature using user-provided ESRI shapefile rectangular labels of the pit's location. These shapefiles must include or be equal to the full product name of the corresponding image file, e.g. label_ESP_033342_1660_RED.shp for the HiRISE image ESP_033342_1660_RED.JP2. **(Type:** bool**)**

Optional arguments are:
- `-p` (`--path`):

  The path to the directory where all of the necessary input data is stored. This will automatically be set when running the Docker container via the installation instructions below. Four folders should be present 'input', 'metadata', 'labels', and 'testing'. **(Default:** '/data/' **/ Type:** str**)**

- `-s` (`--shadows`/`--no-shadows`):

  Save the aligned detected shadow in each image as a PDF file for viewing. This includes the binary shadow mask, but also the detected shadow edge and pit rim overlaid upon the input image to serve as a reference for where the shadow width was measured between. **(Default:** False **/ Type:** bool**)**

- `-t` (`--testing`/`--no-testing`):

  Calculate the precision, recall and F1 score of shadow pixel detections in each image using user-provided ESRI shapefile labels of the pit's shadow. **(Default:** False **/ Type:** bool**)**

- `-f` (`--factor`):

  The factor by which the cropped input image and labels will be down-scaled when calculating the silhouette coefficients during shadow extraction. **(Default:** 0.1 **/ Type:** float**)**

### PITS_plotter.py:

This script is for plotting the *h* profiles calculated across the entire imagery dataset provided to **PITS**. [`PITS_plotter.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/PITS_plotter.py) takes two optional arguments which are called in the command line.

Optional arguments are:
- `-o` (`--outputpath`):

  Path to the directory where the outputs of the PITS algorithm have been saved. This will also be automatically set if using the installation instructions below. **(Default:** '/data/output/' **/ Type:** bool**)**

- `-r` (`--raw`/`--no-raw`):

  Plot the raw apparent depth measurements which have not been corrected for a non-zero satellite emission angle at the time when the image was taken. **(Default:** False **/ **Type:**** bool**)**

## Usage
#### Step 1 - Clone or Download PITS Repository
Clone the **PITS** repository to a useful location on your local file system. In this folder, open a terminal window and run this in the command line. This is assuming you have Git installed, otherwise the repository can be downloaded from [GitHub](https://github.com/dlecorre387/Pit-Topography-from-Shadows).

  > `git clone https://github.com/dlecorre387/Pit-Topography-from-Shadows.git`

#### Step 2 - Acquire Some Data
[`Instructions.txt`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/data/input/Instructions.txt) contains instructions for how to acquire a particular set of MRO HiRISE images known to contain a Martian pit from the MGC<sup>3</sup> catalogue. You can acquire your own by finding the names of the [MRO HiRISE](https://ode.rsl.wustl.edu/mars/tools) or [LRO NAC](https://ode.rsl.wustl.edu/moon/tools) products which contain pits by cross-referencing in GIS software the catalogues mentioned above with the footprints of the images.

These large image products will need to be cropped to the extents of the pit in question for **PITS** to extract its shadow. This can also be done by in GIS by loading in your image, creating a new ESRI shapefile layer for each image, and drawing a rectangular polygon around the pit. This polygon should contain the entire rim, while minimising the amount of pixels from the surrounding surface that are included. These 'location shapefiles' must include or be equal to the full product name of the corresponding image file, e.g. label_ESP_033342_1660_RED.shp for the HiRISE image ESP_033342_1660_RED.JP2.

**PITS** accesses all sensing metadata from the cumulative PDS3 index .TAB files. These can be acquired from NASA's [Planetary Data System](https://pds.nasa.gov/) for the relevant dataset (e.g. RDRCUMINDEX.TAB for HiRISE RDR images). To improve run-time, these should be filtered to only contain the relevant rows for your input images. The BASH script provided in this repository ([`filter_index_files.sh`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/filter_index_files.sh)) can do this for you by placing the index file in the same folder and running:
>`bash filter_index_files.sh [path-to-imagery-folder]`

Lastly, to test the performance of **PITS'** automated shadow extraction, the shadows in each image will need to be labelled. This is done by drawing polygons and assigning an attribute field (called 'class') that described what the particular region of pixels represented. A class of '1' is assigned to the largest continuous shadow in the image which had clearly been cast by the pit's rim. A class of '2' is given to any bright features which were wholly contained within the shadow polygon. As before, these 'validation shapefiles' must include or be equal to the full product name of the corresponding image file.

#### Step 3 - Move Input Data to Correct Folders
Copy or move your input files into the correct folders. These folders will be mounted to the Docker container, meaning that no data needs to be copied/downloaded between it and your local file system. The following input data should be placed in the following folders:
- All input images (cropped or uncropped) should be placed in the '/data/input/' folder.
- If the input images are not cropped, then the necessary pit location labels should be placed in the '/data/labels/' folder.
- If labels of the shadow(s) have been provided for testing **PITS**, then they should be placed in the '/data/testing/' folder.
- The filtered cumulative PDS3 index .TAB file should be placed in the '/data/metadata/' folder.

#### Step 4 - Build or Load the Docker Image
Open a terminal in the cloned repository folder and run the following in the command line to begin building the Docker image. **NOTE:** This Docker image can also be loaded from [Docker Hub](https://hub.Docker.com/repository/Docker/danlecorre/pits-tutorial/general) or downloaded from [Zenodo](https://zenodo.org/record/8168047).
> `docker build -t pits .`

#### Step 5 - Run the Docker Container
Run the Docker container by pasting the following into the command line after the Docker image has been successfully built. This will print a link which you can click on to open the container within JupyterLab in your default browser.
  > `docker run -it -p 8888:8888 -v "$(PWD)\data:/data" pits` - for Windows

  > `docker run -it -p 8888:8888 -v "$(PWD)/data:/data" pits` - for Linux

#### Step 6 (Optional) - Try the PITS Tutorial!
Take a look at the **PITS** tutorial Jupyter notebook! This demo breaks down and explains each element of the **PITS** algorithm and displays the result of each cell. In the '/data/' folder you can find some example data for the red-band and colour versions of the MRO HiRISE image ESP_033342_1660, but not the images themselves. These images contain the MGC<sup>3</sup> feature "APC071". Follow the steps in [`Instructions.txt`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/data/input/Instructions.txt) to download these images and place them in the '/data/input/' folder.

The figure below displays the red version of the tutorial ESP_033342_1660, as well as the detected shadow that you will receive by following the various steps.

![](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/Detected_Shadow.png)

**Figure 1** - Red version of the Mars Reconnaissance Orbiter (MRO) HiRISE image ESP_033342_1660 containing MGC<sup>3</sup> feature "APC071", along with the detected *k*-means clusters and binary shadow mask.

#### Step 7 - Run the PITS Tool
Run the **PITS** tool on your dataset. Change directory into the '/app/' folder where [`run_PITS.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/run_PITS.py) is found. Then run the following line in the terminal of the Docker container. **NOTE:** This is what should be ran if **PITS** is to be applied to non-shadow-labelled HiRISE RDR images which also require cropping.
> `python run_PITS.py -d hirise-rdr -c`

#### Step 8 - Plot the Apparent Depth (*h*) Profiles
Run the [`PITS_plotter.py`](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/scripts/PITS_plotter.py) script to plot the *h* profiles calculated for each image. To do this, without plotting the uncorrected *h* profile, run the following line in the terminal window. The figure below is an example of the *h* profile that is derived by **PITS** (this is actually the *h* profile for the tutorial image ESP_033342_1660_RED).
> `python PITS_plotter.py`

![](https://github.com/dlecorre387/Pit-Topography-from-Shadows/blob/master/Apparent_Depth_Profile.png)

**Figure 2** - Apparent depth (*h*) profile for the red version of the Mars Reconnaissance Orbiter (MRO) HiRISE image ESP_033342_1660 containing MGC<sup>3</sup> feature "APC071".

## Citation Policy
If you have used the PITS tool in your research, please cite the following:
- Daniel Le Corre, David Mary, Nigel Mason, Jeronimo Bernard-Salas and Nick Cox, Automatically calculating the apparent depths of pits using the Pit Topography from Shadows (PITS) tool, RAS Techniques and Instruments, Volume 2, Issue 1, January 2023, Pages 492–509, https://doi.org/10.1093/rasti/rzad037

## Acknowledgements
This project is part of the Europlanet 2024 RI which has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871149.
