import os

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

# Configure logger
logging.basicConfig(level = logging.INFO,
                    format='| %(asctime)s | %(levelname)s | Message: %(message)s',
                    datefmt='%d/%m/%y @ %H:%M:%S')

import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from tqdm import tqdm
from MAPS_functions import *

# Define drivers for writing vector and raster data
driver1 = gdal.GetDriverByName("GTiff")
driver2 = ogr.GetDriverByName("ESRI Shapefile")

'''
Optional Parameters:

CROPPING:   Are you providing MAPS with labels to crop your images down to the pit feature? 
            Labels must be equal to or include the name of the corresponding image file 
            e.g. label_ESP_033342_1660_RED.shp for image ESP_033342_1660_RED.JP2 (Type: bool)

TRAINING:   Are you providing some ground truth labels of pit shadows in order to calibrate
            MAPS? This is False by default. (Type: bool)

Do Not Change These:

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

# Optional parameters
CROPPING = False
TRAINING = True

# Do not change these
CLUSTERS = [4, 5, 6, 7, 8, 9, 10, 11, 12]
MISS_RATE = 0.025436094042679658
FD_RATE = 0.06256515403386054
INPUT_DIR = '/data/input/'
METADATA_DIR = '/data/metadata/'
LABELS_DIR = '/data/labels/'
TRAINING_DIR = '/data/training/'
OUTPUT_DIR = '/data/output/'

def main(cropping,
        training,
        clusters,
        miss_rate,
        FD_rate,
        input_dir, 
        metadata_dir, 
        labels_dir,
        training_dir,
        output_dir):

    if training:

        # Redefine input and output directories
        input_dir = os.path.join(training_dir, 'input/')
        labels_dir = os.path.join(training_dir, 'labels/')
        metadata_dir = os.path.join(training_dir, 'metadata/')
        output_dir = os.path.join(training_dir, 'output/')

        # List all filenames to be fed through MAPS without .XML files
        filenames = [file for file in os.listdir(input_dir) if not file.endswith('.xml')]

        # Testing metrics
        P = np.empty((len(filenames), len(clusters)))
        R = np.empty((len(filenames), len(clusters)))
        F1 = np.empty((len(filenames), len(clusters)))

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

    logging.info("Output folder created/cleaned")

    # List all filenames to be fed through MAPS without .XML files
    filenames = [file for file in os.listdir(input_dir) if not file.endswith('.xml')]

    logging.info("The following files will be analysed: {}".format(filenames))
        
    # For storing indexes of recommended clusters and silhouette coefficients
    silh_ind = []
    silhouette_avgs = np.empty((len(filenames), len(clusters)))

    # Create the subfolders to store MAPS' results
    shadows_dir = os.path.join(output_dir, 'shadows/')
    plots_dir = os.path.join(output_dir, 'plots/')
    results_dir = os.path.join(output_dir, 'results/')
    for subfolder in [shadows_dir, plots_dir, results_dir]:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    # Open emtpy arrays to store sensing info and MAPS results
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

    # Loop through each image
    for i, filename in enumerate(filenames):

        # Get product name of file without file extension
        name = os.path.splitext(filename)[0]

        logging.info("Analysing image {}".format(name))

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
            
        # Extract one band of a mulitband image
        if n_bands > 1:
            cropped_im = cropped_im[0, :, :]

        # Find the size of the input image
        im_size = cropped_im.size

        if training:

            # Read in ground truth if training
            val_array = ImAn.read_ground_truth(cropped_im, geot, proj)
            background = np.where(val_array == 1, 0, 1)

        # Open progress bar to monitor clustering
        c_pbar = tqdm(total=len(clusters), desc='Clustering {}'.format(name))

        # Open an empty array to store the unsorted labels
        labels = np.empty((len(clusters), xsize, ysize))

        # Loop over different numbers of kmeans clusters and shadow cluster threshold
        for c, cluster in enumerate(clusters):

            # Label the cropped image using K-Means clustering and calculate mean silhouette coefficient
            label, iterations, inertia, silhouette_avg = kmeans_clustering(cropped_im, cluster)
            silhouette_avgs[i, c] = silhouette_avg
            labels[c, :, :] = label

            c_pbar.update(1)
        c_pbar.close()

        # Find the labels for k which gave the highest silhouette coefficient
        silh_ind.append(int(np.argmax(silhouette_avgs[i, :])))
        sorted_labels = sort_clusters(cropped_im, labels[silh_ind[i], :, :])
    
        # Extract darkest cluster
        shadow = np.where(sorted_labels == np.amin(sorted_labels), 1, 0)                           
        shadow = postprocessing(shadow)
        
        # Get the average brightness of the shadow
        brightness = (cropped_im*np.where(shadow == 1, 1, -1)).flatten()
        cluster_val = np.mean(brightness[brightness >= 0])

        if training:
                    
            # Calculate the TDP and FDP for this training sample
            TP = np.sum(shadow*val_array)
            FP = np.sum(shadow*background)
            FN = np.sum(val_array) - np.sum(shadow*val_array)
            P[i, silh_ind[i]] = TP/(TP + FP)
            R[i, silh_ind[i]] = TP/(TP + FN)
            F1[i, silh_ind[i]] = (2*P[i, silh_ind[i]]*R[i, silh_ind[i]])/(P[i, silh_ind[i]] + R[i, silh_ind[i]])

            # Turn shadow detections into rgb array
            output_R = np.where(shadow*background == 1, 1, 0)
            output_G = np.where(shadow*val_array == 1, 1, 0)
            output_B = np.where(val_array - shadow == 1, 1, 0)
            colour = np.empty((xsize, ysize, 3))
            colour[:, :, 0] = output_R
            colour[:, :, 1] = output_G
            colour[:, :, 2] = output_B
                    
            # Plot the histogram and thresholded image
            fig, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1]})
            ax0.imshow(cropped_im, cmap='gray')
            ax0.axis('off')
            ax0.set_title("Cropped Input Image ({})".format(name))
            ax1.imshow(sorted_labels, cmap='viridis')
            ax1.axis('off')
            ax1.set_title("K-means Clusters Via\n" + r"Silhouette Analysis ($k={}$)".format(clusters[silh_ind[i]]))
            ax2.imshow(colour)
            ax2.axis('off')
            ax2.set_title("Processed Shadow Mask")
            fig.savefig(os.path.join(plots_dir, name + '_shadow.png'), bbox_inches='tight')

            # Use the errors (miss rate and FP rate) calculated by comparing to the ground truth
            precision = P[i, silh_ind[i]]
            recall = R[i, silh_ind[i]]
            miss_rate = 1 - recall
            FD_rate = 1 - precision

        # Retrieve HiRISE metadata
        resolution, inc_angle, em_angle, azim_angle = ImAn.read_HiRISE_metadata()
        res[i], inc[i], azim[i], em[i] = resolution, inc_angle, azim_angle, em_angle 

        # Extract the shadow and measure its width at every coordinate
        x = measure_shadow(shadow, geot, proj, azim_angle, resolution)
        x = x[x != 0]

        # Define the positive and negative uncertainty bounds for the shadow width
        pos_x = miss_rate*x
        neg_x = FD_rate*x

        # Calculate the apparent depth at all points along the shadow
        h, pos_h, neg_h = calculate_h(x, pos_x, neg_x, inc_angle)
        
        # Find the centre and minimum shadow apparent depths
        m = int(h.size/2)
        h_cs[i], pos_h_cs[i], neg_h_cs[i] = h[m], pos_h[m], neg_h[m]
        h_ms[i], pos_h_ms[i], neg_h_ms[i] = np.amax(h), pos_h[h == np.amax(h)], neg_h[h == np.amax(h)]

        # Save the remaining outputs of the MAPS tool: shadow shapefile and apparent depth profile
        ImAn.save_outputs(geot, proj, shadow, cluster_val, x, h, pos_h, neg_h, np.amax(h))

    logging.info("All {} images analysed and outputs saved to {}".format(len(filenames), output_dir))

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
    np.savetxt(os.path.join(results_dir, 'MAPS_results.csv'), array,
            delimiter=',',
            fmt='%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f',
            header='Image Name, Resolution [m], Solar Incidence Angle [deg], Solar Azimuth Angle [deg], Emission Angle [deg], Centre h [m], +, -, Maximum h [m], +, -')

    if training:

        logging.info("Average miss rate: {}".format(1 - np.mean(np.diagonal(R[:, silh_ind]))))
        logging.info("Average FP rate: {}".format(1 - np.mean(np.diagonal(P[:, silh_ind]))))
        logging.info("Average F1 score using silhouette analysis: {}".format(np.mean(np.diagonal(F1[:, silh_ind]))))
        
        # Save the training results to a csv file
        dt = np.dtype([('i', 'U32'), ('k_s', float), ('P_s', float), ('R_s', float), ('F1_s', float)])
        array = np.empty(len(filenames), dtype=dt)
        array['i'] = [os.path.splitext(filename)[0] for filename in filenames]
        array['k_s'] = [clusters[i] for i in silh_ind]
        array['P_s'] = np.diagonal(P[:, silh_ind])
        array['R_s'] = np.diagonal(R[:, silh_ind])
        array['F1_s'] = np.diagonal(F1[:, silh_ind])

        # Save to a csv file for reading later
        np.savetxt(os.path.join(results_dir, 'testing_results.csv'), array,
                delimiter=',',
                fmt='%s, %f, %f, %f, %f',
                header='Image Name, Number of Clusters (k), Precision [%], Recall [%], F1 score [%]')

if __name__ == "__main__":
    main(CROPPING,
        TRAINING,
        CLUSTERS,
        MISS_RATE,
        FD_RATE,
        INPUT_DIR,
        METADATA_DIR,
        LABELS_DIR,
        TRAINING_DIR,
        OUTPUT_DIR)