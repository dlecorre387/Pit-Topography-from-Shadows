import os
import shutil
import argparse
import logging
import numpy as np
from osgeo import gdal, ogr
from tqdm import tqdm
from scipy.stats import mode

# Import PITS functions
from PITS_functions import *

# Configure logger
logging.basicConfig(level = logging.INFO,
                    format='| %(asctime)s | %(levelname)s | Message: %(message)s',
                    datefmt='%d/%m/%y @ %H:%M:%S')

'''
Optional Parameters:

-c (--cropping):    Are you providing MAPS with labels to crop your images down to the pit feature? 
                    Labels must be equal to or include the name of the corresponding image file 
                    e.g. label_ESP_033342_1660_RED.shp for image ESP_033342_1660_RED.JP2.
                    (No default / Type: bool)

-t (--training):    Are you providing some ground truth labels of pit shadows in order to calibrate
                    MAPS? 
                    (Default: False / Type: bool)
            
-f (--factor):      The factor by which the cropped input image and labels will be down-scaled when
                    calculating the silhouette coefficients during shadow extraction. 
                    (Default: 0.1 / Type: float)


DO NOT CHANGE THESE VARIABLES:

CLUSTERS:       The range of k values to iterate over. (Type: list of str)

MISS_RATE:      The miss rate of shadow detections. This will be overwritten if training is
                True with the new average value for your labelled images. (Type: float)

FD_RATE:        The false discovery rate of shadow detections. This will be overwritten if 
                training is True with the new average value for your labelled images. (Type: float)

INPUT_DIR:      Path to the input images (either cropped or un-cropped). (Type: str)

METADATA_DIR:   Path to the HiRISE metadata files containing sensing info. (Type: str)

LABELS_DIR:     Path to the labels used for cropping images down to the pit feature. (Type: str)

TRAINING_DIR:   Path to the ground truth labels of pit shadows. (Type: str)

OUTPUT_DIR:     Path to the output directory where all of the results and plots will be saved. (Type: str)

'''

# Initialize arguments parser and define arguments
PARSER = argparse.ArgumentParser()
PARSER.add_argument("-c", "--cropping", type=bool, help = "Will cropping be required?")
PARSER.add_argument("-t", "--training", type=bool, default=False, help = "Have shadow labels been provided for testing?")
PARSER.add_argument("-f", "--factor", type=float, default=0.1, help = "Down-scaling factor for silhouette coefficient calculation")
ARGS = PARSER.parse_args()

# Do not change these variables
CLUSTERS = np.arange(4, 14, 1)
MISS_RATES = [0.00473, 0.00681]
FD_RATES = [0.06939, 0.08432]
INPUT_DIR = '/data/input/'
METADATA_DIR = '/data/metadata/'
LABELS_DIR = '/data/labels/'
TRAINING_DIR = '/data/training/'
OUTPUT_DIR = '/data/output/'

def main(cropping,
        training,
        factor,
        clusters,
        miss_rates,
        FD_rates,
        input_dir, 
        metadata_dir, 
        labels_dir,
        training_dir,
        output_dir):

    # Clean or create output folder
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file, e))
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the subfolders to store shadow shapefile, and csv of results
    shadows_dir = os.path.join(output_dir, 'shadows/')
    results_dir = os.path.join(output_dir, 'results/')
    for subfolder in [shadows_dir, results_dir]:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    
    logging.info("Output folder and sub-folders created/cleaned")

    # List all filenames to be fed through MAPS without .XML files
    filenames = [file for file in os.listdir(input_dir) if not file.endswith('.xml')]

    logging.info("The following files will be analysed: {}".format(filenames))
    
    if training:

        # Set up arrays for storing testing metrics
        P = np.empty(len(filenames))
        R = np.empty(len(filenames))
        F1 = np.empty(len(filenames))

    # Open emtpy arrays to store sensing info and APP results
    res = np.empty((len(filenames)))
    inc = np.empty((len(filenames)))
    azim = np.empty((len(filenames)))
    em = np.empty((len(filenames)))
    h_cs = np.empty((len(filenames)))
    pos_h_cs = np.empty((len(filenames)))
    neg_h_cs = np.empty((len(filenames)))
    h_ms = np.empty((len(filenames)))
    pos_h_ms = np.empty((len(filenames)))
    neg_h_ms = np.empty((len(filenames)))
    
    # Store the appropriate clusters via silhouette analysis
    dark_ind = np.empty(len(filenames), dtype=int)
    silhouette_darkests = np.empty((len(filenames), len(clusters)))

    # Open progress bar to monitor clustering
    pbar = tqdm(total=len(filenames)*clusters.size, desc='PITS progress')

    # Loop through each image
    for i, filename in enumerate(filenames):

        # Initialise ImageAnalyser class
        ImAn = ImageAnalyser(filename=filename,
                            input_dir=input_dir, 
                            metadata_dir=metadata_dir,
                            labels_dir=labels_dir,
                            training_dir=training_dir,
                            output_dir=output_dir)

        # Crop the image using provided labels and return geotransform and projection
        if cropping:
            cropped_im, geot, proj, n_bands, xsize, ysize = ImAn.crop_image()

        # Read the pre-cropped image
        elif not cropping:
            cropped_im, geot, proj, n_bands, xsize, ysize = ImAn.read_cropped_im()

        # Retrieve the correct miss and false discovery rates for the number of bands
        if n_bands == 1:
            miss_rate = miss_rates[0]
            FD_rate = FD_rates[0]
        elif n_bands > 1:
            miss_rate = miss_rates[1]
            FD_rate = FD_rates[1]
        else:
            raise ValueError("Number of bands should not be zero.")
        
        # Read in ground truth if training
        if training:
            val_array = ImAn.read_ground_truth(cropped_im, n_bands, geot, proj)
            background = np.where(val_array == 1, 0, 1)

        # Open an empty array to store the unsorted labels and number of iterations for each appropriate value of k   
        labels = np.empty((len(clusters), xsize, ysize))

        # Loop over different numbers of kmeans clusters and shadow cluster threshold
        for c, cluster in enumerate(clusters):

            # Cluster the image and sort the labels for a single-band image
            if n_bands == 1:
                
                # Cluster image
                label = kmeans_clustering(cropped_im, cluster)
                                
                # Sort the labels and save the silhouette coefficient
                sorted_labels, silhouette_darkest = sort_clusters(cropped_im, label, factor)
                silhouette_darkests[i, c] = silhouette_darkest
            
            # Cluster and sort the labels of each band in a multi-band image
            elif n_bands > 1:
                
                # Store the silhouette coefficient and labels for each band
                darkests = np.empty(len(filenames))
                all_labels = np.empty((n_bands, xsize, ysize))
                
                # Loop through each band
                for band in np.arange(0, n_bands):
                    
                    # Cluster the individual band
                    label = kmeans_clustering(cropped_im[band, :, :], cluster)
                    
                    # Sort the labels and find the silhouette coefficient for this band
                    all_labels[band, :, :], silhouette_darkest = sort_clusters(cropped_im, label, factor)
                    darkests[i] = silhouette_darkest
                
                # Average the silhouette coefficients across the bands    
                silhouette_darkests[i, c] = np.mean(darkests)
                    
                # Calculate modal labels if applied to colour images
                sorted_labels = (mode(all_labels, axis=0).mode).astype(int)
            
            else:
                raise ValueError("Number of bands should not be zero.")
            
            # Store the sorted labels for later and save the number of iterations for this value of k
            labels[c, :, :] = sorted_labels
        
            pbar.update(1)

        # Find the labels for k which gave the highest silhouette coefficient and F1
        dark_ind[i] = int(np.argmax(silhouette_darkests[i, :]))
                
        # Use the labels which maximised the darkest clusters silhouette coefficient
        shadow = np.where(labels[dark_ind[i], :, :] == np.amin(labels[dark_ind[i], :, :]), 1, 0)                 
        shadow = postprocessing(shadow)
        
        # Plot the shadow extraction testing results
        if training:
            
            # Calculate the precision, recall and F1 this training sample
            TP = np.sum(shadow*val_array)
            FP = np.sum(shadow*background)
            FN = np.sum(val_array) - np.sum(shadow*val_array)
            P[i] = TP/(TP + FP)
            R[i] = TP/(TP + FN)
            F1[i] = (2*P[i]*R[i])/(P[i] + R[i])
            
            # Use the errors (miss rate and FD rate) calculated by comparing to the ground truth
            miss_rate = 1 - R[i]
            FD_rate = 1 - P[i]

        # Retrieve HiRISE metadata
        resolution, inc_angle, em_angle, azim_angle = ImAn.read_HiRISE_metadata()
        res[i], inc[i], azim[i], em[i] = resolution, inc_angle, azim_angle, em_angle 

        # Extract the shadow and measure its width at every coordinate
        x = measure_shadow(shadow, azim_angle, resolution)
        x = x[x != 0]

        # Define the positive and negative uncertainty bounds for the shadow width
        pos_x = miss_rate*x
        neg_x = FD_rate*x

        # Calculate the apparent depth at all points along the shadow
        h, pos_h, neg_h = calculate_h(x, pos_x, neg_x, inc_angle)
        
        # Find the centre and minimum shadow apparent depths
        m = int(h.size/2)
        h_cs[i], pos_h_cs[i], neg_h_cs[i] = h[m], pos_h[m], neg_h[m]
        if len(h == np.amax(h)) == 1:
            h_ms[i], pos_h_ms[i], neg_h_ms[i] = np.amax(h), pos_h[h == np.amax(h)], neg_h[h == np.amax(h)]
        else:
            h_ms[i], pos_h_ms[i], neg_h_ms[i] = np.amax(h), pos_h[h == np.amax(h)][0], neg_h[h == np.amax(h)][0]

        # Save the remaining outputs of the MAPS tool: shadow shapefile and apparent depth profile
        ImAn.save_outputs(geot, proj, shadow, resolution, x, h, pos_h, neg_h)
        
    pbar.close()

    # Save the training results to a csv file
    dt = np.dtype([('filename', 'U32'),
                    ('res', float), ('inc', float), ('azim', float), ('em', float),
                    ('h_c', float), ('pos_h_c', float), ('neg_h_c', float), 
                    ('h_m', float), ('pos_h_m', float), ('neg_h_m', float)])
    array = np.empty(len(filenames), dtype=dt)
    array['filename'] = [os.path.splitext(filename)[0] for filename in filenames]
    array['res'] = res
    array['inc'] = inc
    array['azim'] = azim
    array['em'] = em
    array['h_c'] = h_cs
    array['pos_h_c'] = pos_h_cs
    array['neg_h_c'] = neg_h_cs
    array['h_m'] = h_ms
    array['pos_h_m'] = pos_h_ms
    array['neg_h_m'] = neg_h_ms

    # Save to a csv file for reading later
    np.savetxt(os.path.join(results_dir, 'PITS_results.csv'), 
            array,
            delimiter=',',
            fmt='%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f',
            header='Image Name, Resolution [m], Solar Incidence Angle [deg], Solar Azimuth Angle [deg], Emission Angle [deg], Centre h [m], +, -, Maximum h [m], +, -')

    logging.info("All {} images analysed and outputs saved to {}".format(len(filenames), output_dir))

    if training:
                
        logging.info("Average miss rate: {}".format(1 - np.mean(R)))
        logging.info("Average false discovery rate: {}".format(1 - np.mean(P)))
        logging.info("Average F1 score: {}".format(np.mean(F1)))
        
        dt = np.dtype([('i', 'U32'),
                    ('k', float), ('P', float), ('R', float), ('F1', float)])
        array = np.empty(len(filenames), dtype=dt)
        array['i'] = [os.path.splitext(filename)[0] for filename in filenames]
        array['k'] = [clusters[i] for i in dark_ind]
        array['P'] = P
        array['R'] = R
        array['F1'] = F1

        np.savetxt(os.path.join(results_dir, 'PITS_testing_results.csv'), 
                array,
                delimiter=',',
                fmt='%s, %f, %f, %f, %f',
                header='Image Name, k, P [%], R [%], F1 [%]')

if __name__ == "__main__":
    main(ARGS.cropping,
        ARGS.training,
        ARGS.factor,
        CLUSTERS,
        MISS_RATES,
        FD_RATES,
        INPUT_DIR,
        METADATA_DIR,
        LABELS_DIR,
        TRAINING_DIR,
        OUTPUT_DIR)