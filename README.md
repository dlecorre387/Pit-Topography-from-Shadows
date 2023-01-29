# The **PITS** (**Pi**t **T**opology from **S**hadows) Tool
## Introduction
### What are pits?
Pits, or pit craters, are near-circular depressions found in planetary surfaces which are generally formed through gravitational collapse. Pits will be primary targets for future space exploration and habitability for their presence on most rocky Solar System surfaces and their potential to be entrances to sub-surface cavities. This is particularly true on Mars, where ice caves are thought to be stable across much of the surface - on which astronauts will also be exposed to high radiation dosages. Since pits are rarely found to have corresponding high-resolution elevation data, tools are required for approximating their depths in order to find those which are the ideal candidates for exploration.

### How does PITS operate?

The PITS (Pit Topology from Shadows) tool is a dockerised Python framework which can automatically calculate a profile of the apparent depth (*h*) of an extra-terrestrial pit from just one cropped single- or multi-band remote-sensing image. *h* is the relative depth of the pit between its rim and the edge of the shadow cast by the Sun - with the principle being that a deeper pit would cast a wider shadow.

PITS does this by employing image segmentation (in the form of unsupervised *k*-means clustering and silhouette analysis for automatic cluster suggestion) in order to produce a binary mask of shadow or non-shadow pixels. Then, by rotating the shadow mask by the solar azimuth angle, PITS can measure the width of the shadow along the Sun's line of sight (*S*) at each pixel in the shadows length. *h* is the derived from these *S* measurements by considering the solar incidence angle for this particular image.

PITS is intended to be used on known/catalogued pits, or as a post-processing tool after pits have been automatically detected perhaps through the use of Machine/Deep Learning. While PITS has so far only been tested upon Mars Reconnaissance Orbiter HiRISE image sof Martian pits, the tool is highly applicable to data from other sensors and other planetary surfaces.

### Testing Performance

Across 19 shadow-labelled Mars Reconnaissance Orbiter red-band HiRISE images of Atypical Pit Craters (APCs) from the Mars Global Cave Candidate Catalog (MGC3), PITS detected 99.5\% of all shadow pixels (with 93.1\% of all detections being true shadow pixels). This equates to an expected average F1 score of 96.1\% when applying PITS to HiRISE red-band images. Testing upon 12 HiRISE colour images found that a small improvement (~0.3\%) in F1 was achieved compared to performance upon their corresponding red-band versions. However, since the run-time of the PITS tool will increase by a factor of *n* for an *n*-band image, the recommendation is to use single-band imagery due to already high performance and lower run-time.

When applying PITS to 123 HiRISE images of MGC3 APCs, PITS exhibited an average run-speed of 86.74 s/Mpx to analyse a red-band HiRISE image of a Martian pit. This was achieved when using a laptop with an 11th generation Intel Core i5 processor and 8GB of RAM. As a result, the actual run-time per image varied according to its size. This equated to a minimum and maximum run-time of 5.72 and 503.92~s for images with 0.02 and 4.80 Mpx, respectively.

## Requirements
- Python (version 3.x)
- Docker (version 4.8.x or above)
- Visual Studio Code along with the 'Docker' extension is highly recommended to be able to manage files within the docker container.

## Code
This repository contains three scripts: `PITS_functions.py`, `run_PITS.py` and `PITS_plotter.py`:

`PITS_functions.py`:

This includes all of the functions that are required for PITS to read the user-inputted raster images and sensing information, to automatically extract the shadow via *k*-means clustering, and to calculate the apparent depths of pits. These functions have no user input and should not be altered.

`run_PITS.py`:

This script calls all of the necessary functions from `PITS_functions.py` in the correct order, in order to carry out the method of the PITS tool and save the necessary outputs. `run_PITS.py` takes 3 user-inputted arguments which are called in the command line.

Optional parameters in `run_PITS.py` are:
- `-c` (`--cropping`): Are you providing pit location polygon ESRI shapefile labels to crop larger HiRISE images to the extents of the pit feature? This argument has not default and is required for PITS to run. Type: bool.
- `-t` (`--training`): Are you providing ESRI shapefile polygon labels of the main pit shadow(s) in your image(s) in order to test the accuracy of PITS' shadow extraction? The apparent depth profiles will still be calculated for each image even if `--training True`. The default is False since this is the most common use case of the PITS tool. Type: bool.
- `-f` (`--factor`): What down-scaling factor should be applied to the cropped input image and cluster labels in order to conserve run-time when calculating silhouette coefficients. The default is 0.1 (i.e. arrays are a tenth of the original resolution) which was used to produce the above testing results. Type: float.

`PITS_plotter.py`:

This script is for plotting the apparent depths calculated across the entire imagery dataset provided to PITS as a histogram, as well as the individual apparent depth profiles calculated in each image of a pit. `PITS_plotter.py` takes just one optional argument which is called in the command line.

- `-e` (`--elimit`): What limit (in degrees) should be applied to the emission angle that an image can have for its apparent depths to be included in the histogram and for its *h* profile to be plotted. The default limit is 10 degrees since this balanced the reliability of the profiles but also the amount of data lost in limiting emission angles. To remove this limit, set `--elimit None`. Type: float or NoneType.

## Usage
1. Clone the repository
> `git clone https://github.com/dlecorre387/Pit-Topology-from-Shadows.git`

2. Copy or move your input files into the correct folders (these folders will be copied when building the docker image in step 3). All input images (cropped or uncropped) should be placed in the **/data/input/** folder. If the input images are not cropped, then the necessary pit location labels should be copied to the **/data/labels/** folder. The metadata .LBL files for each HiRISE image should be placed in the **/data/metadata/** folder. If labels of the shadow(s) have been provided for testing PITS, then they should be placed in the **/data/training/** folder. **NOTE:** You can easily drag and drop new data into the relevant folders once the docker container has already been built by using the 'Docker' extension in Visual Studio Code.

3. Build the docker image from within the cloned repository. **NOTE:** The full stop is essential here.
> `docker build -t pits .`

4. Run the docker container. **NOTE:** This may take a while to run, don't worry!
> `docker run -it pits`

6. Run the PITS tool. The current working directory is already set to **/app/**, but in case this changes, cd back into **/app/** where `run_PITS.py` is found. Then run the following line in the terminal of the docker container. **NOTE** This is what should be ran if PITS is to be applied to unlabelled images which also require cropping.
> `python run_PITS.py -c True -t False`

7. Try using PITS on the demo data that is already stored in the repository. The **/data/** folder contain the necessary labels and metadata files for one HiRISE image. Due to the file sizes, the HiRISE image is not given in the repository, but can be acquired through the Mars Orbital Data Explorer of NASA's Planetary Data System's Geosciences Node (go to https://ode.rsl.wustl.edu/mars/). You can try running PITS with `--training True` to get performance scores based on the accuracy of PITS' shadow extraction upon the HiRISE image ESP_033342_1660_RED. Set `--training False` to get results for the same HiRISE image as if we did not know the location of the shadow. The raster image products from the Mars ODE will not be cropped to the extents of the pit. Therefore, to run this demo please ensure `--cropping True`.

8. Run the `PITS_plotter.py` script to plot the histogram of the apparent depths calculated by the PITS tool across you entire imagery dataset. This script will also plot the apparent depth profiles calculated for each image. To do this, run the following line in the terminal window.
> `python PITS_plotter.py`
