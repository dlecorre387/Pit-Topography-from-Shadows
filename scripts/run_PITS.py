'''
Created by Daniel Le Corre (1,2)* 
Last edited on 07/07/2023
1   Centre for Astrophysics and Planetary Science, University of Kent, Canterbury, United Kingdom
2   Centres d'Etudes et de Recherches de Grasse, ACRI-ST, Grasse, France
*   Correspondence: dl387@kent.ac.uk
    Website: https://www.danlecorre.com/
    
This project is part of the Europlanet 2024 RI which has received
funding from the European Union’s Horizon 2020 research and innovation
programme under grant agreement No 871149.

'''

import os
import shutil
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mode
from skimage.transform import rotate

# Import PITS functions
from PITS_functions import *

# Configure logger
logging.basicConfig(level = logging.INFO,
                    format='| %(asctime)s | %(levelname)s | Message: %(message)s',
                    datefmt='%d/%m/%y @ %H:%M:%S')

'''
Required Parameters:

-d (--dataset):     The name of the dataset whose images will be used to calculate apparent depths.
                    Currently supported options are "hirise-rdr" (for MRO HiRISE RDR version 1.1 
                    images of Mars) and "lronac-edr" (for LROC NAC EDR images of the Moon). This 
                    is required since there is a different process for retrieving sensing 
                    information for each dataset.
                    (No default / Type: str)

-c (--cropping):    Crop each larger input image to the extents of the pit feature using user-provided
                    ESRI shapefile rectangular labels of the pit's location. These shapefiles must 
                    include or be equal to the full product name of the corresponding image file, e.g. 
                    label_ESP_033342_1660_RED.shp for the HiRISE image ESP_033342_1660_RED.JP2.
                    (No default / Type: bool)

Optional Parameters:

-s (--shadows):     Save the aligned detected shadow in each image as a PDF file for viewing. This 
                    includes the binary shadow mask, but also the detected shadow edge and pit rim 
                    overlaid upon the input image to serve as a reference for where the shadow width 
                    was measured between.
                    (Default: False / Type: bool)

-t (--testing):     Calculate the precision, recall and F1 score of shadow pixel detections in each image 
                    using user-provided ESRI shapefile labels of the pit's shadow. 
                    (Default: False / Type: bool)
            
-f (--factor):      The factor by which the cropped input image and labels will be down-scaled when
                    calculating the silhouette coefficients during shadow extraction. 
                    (Default: 0.1 / Type: float)


DO NOT CHANGE THESE VARIABLES:

CLUSTER_RANGE:  The range of k values to iterate over. 
                (Type: list of str)

MISS_RATE:      The miss rate of shadow detections. This will be overwritten if --testing is
                passed with the new average value for your labelled images. 
                (Type: float)

FD_RATE:        The false discovery rate of shadow detections. This will be overwritten if 
                --testing is passed with the new average value for your labelled images. 
                (Type: float)

INPUT_DIR:      Path to the folder containing input images (either cropped or un-cropped). 
                (Type: str)

METADATA_DIR:   Path to the folder containing the metadata files containing sensing info. 
                (Type: str)

LABELS_DIR:     Path to the folder containing the shapefiles used for cropping images to the pit feature. 
                (Type: str)

TESTING_DIR:    Path to the folder containing the user-created ground truth shadow shapefiles. 
                (Type: str)

OUTPUT_DIR:     Path to the output directory where all of the results and plots will be saved. 
                (Type: str)
'''

# Initialise arguments parser and define arguments
PARSER = argparse.ArgumentParser()
PARSER.add_argument("-d", "--dataset", type=str, required=True, help = "Which dataset is being used? ['hirise-rdr'|'lronac-edr']")
PARSER.add_argument("-c", "--cropping", action=argparse.BooleanOptionalAction, required=True, help = "Do images require cropping to extents of the target feature? [--cropping|--no-cropping]")
PARSER.add_argument("-s", "--shadows", action=argparse.BooleanOptionalAction, default=False, help = "Should the detected aligned shadows be saved for reference? [--shadows|--no-shadows]")
PARSER.add_argument("-t", "--testing", action=argparse.BooleanOptionalAction, default=False, help = "Have validation shapefiles been provided for all images to test shadow extraction? [--testing|--no-testing]")
PARSER.add_argument("-f", "--factor", type=float, default=10, help = "Down-scaling factor for silhouette coefficient calculation [float]")
ARGS = PARSER.parse_args()

# Do not change these variables
CLUSTER_RANGE = np.arange(4, 14, 1)
MISS_RATES = [0.004280421, 0.00611175]
FD_RATES = [0.052279632, 0.059128667]
INPUT_DIR = '/data/input/'
METADATA_DIR = '/data/metadata/'
LABELS_DIR = '/data/labels/'
TESTING_DIR = '/data/testing/'
OUTPUT_DIR = '/data/output/'

def main(dataset,
        cropping,
        shadows,
        testing,
        factor,
        cluster_range,
        miss_rates,
        FD_rates,
        input_dir, 
        metadata_dir,
        labels_dir,
        testing_dir,
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
        logging.info(f"Output folder '{output_dir}' cleaned.")
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Output folder '{output_dir}' created.")
       
    # Create sub-folders for separating, h profiles, shadow shapefiles and other results
    plots_dir = os.path.join(output_dir, 'figures/')
    profiles_dir = os.path.join(output_dir, 'profiles/')
    shadows_dir = os.path.join(output_dir, 'shadows/')
    subfolders = [plots_dir, profiles_dir, shadows_dir]
    for subfolder in subfolders:
        if not os.path.exists(subfolder):
            try:
                os.makedirs(subfolder)
            except Exception as e:
                print('Failed to create subfolder %s. Reason %s' % (subfolder, e))
    logging.info(f"Sub-folders created.")
    
    # List all filenames to be analysed without .XML files
    filenames = [file for file in os.listdir(input_dir) if not file.endswith('.xml')]
    
    logging.info(f"The following files will be analysed ({len(filenames)} in total):")
    for file_n, filename in enumerate(filenames):
        print(f"File {file_n+1}:     {filename}")

    # Open empty arrays to store sensing info and results
    resolutions = np.empty((len(filenames)))
    inc_angles = np.empty((len(filenames)))
    solar_azim_angles = np.empty((len(filenames)))
    sc_azim_angles = np.empty((len(filenames)))
    em_angles = np.empty((len(filenames)))
    centre_raw_hs = np.empty((len(filenames)))
    pos_centre_raw_hs = np.empty((len(filenames)))
    neg_centre_raw_hs = np.empty((len(filenames)))
    centre_hs = np.empty((len(filenames)))
    pos_centre_hs = np.empty((len(filenames)))
    neg_centre_hs = np.empty((len(filenames)))
    max_raw_hs = np.empty((len(filenames)))
    pos_max_raw_hs = np.empty((len(filenames)))
    neg_max_raw_hs = np.empty((len(filenames)))
    max_hs = np.empty((len(filenames)))
    pos_max_hs = np.empty((len(filenames)))
    neg_max_hs = np.empty((len(filenames)))
    
    # Set up arrays for storing testing metrics
    if testing:
        P_shadow = np.empty(len(filenames))
        R_shadow = np.empty(len(filenames))
        F1_shadow = np.empty(len(filenames))
        P_bright = np.empty(len(filenames))
        R_bright = np.empty(len(filenames))
        F1_bright = np.empty(len(filenames))

    # Open progress bar, store the silhouette coefficients/scores, and find the target k/F1
    pbar = tqdm(total=len(filenames) * cluster_range.size, desc='Progress')
    silhouettes = np.empty((len(filenames), len(cluster_range)))

    # Loop through each image
    for i, filename in enumerate(filenames):
            
        # Initialise DataPreparer class
        DataPrep = DataPreparer(filename=filename,
                                input_dir=input_dir, 
                                metadata_dir=metadata_dir,
                                labels_dir=labels_dir,
                                testing_dir=testing_dir,
                                output_dir=output_dir)
        
        if cropping:
            
            # Crop the image using provided shapefile labels and return raster information [resolution in m, lat/lon in deg]
            cropped_image, resolution, min_longitude, max_longitude, min_latitude, max_latitude, geotransform, projection, n_bands, x_size, y_size = DataPrep.crop_image()

            # Retrieve sensing angles at time of acquisition [in radians]
            inc_angle, solar_azim_angle, sc_azim_angle, phase_angle, em_angle, delta_em_angle, em_angle_par, em_angle_perp = DataPrep.read_metadata(dataset, min_longitude, max_longitude, min_latitude, max_latitude)
            
        elif not cropping:
            
            # Read the pre-cropped image and return raster information [resolution in m]
            cropped_image, resolution, geotransform, projection, n_bands, x_size, y_size = DataPrep.read_cropped_im()

            # Retrieve sensing angles at time of acquisition [in radians]
            inc_angle, solar_azim_angle, sc_azim_angle, phase_angle, em_angle, delta_em_angle, em_angle_par, em_angle_perp = DataPrep.read_metadata(dataset, None, None)
        
        # Store metadata in arrays for saving later (converting angles back to degrees)
        resolutions[i], inc_angles[i], solar_azim_angles[i], sc_azim_angles[i], em_angles[i] = resolution, inc_angle * (180 / np.pi), solar_azim_angle * (180 / np.pi), sc_azim_angle * (180 / np.pi), em_angle * (180 / np.pi)
        
        # Open an empty array to store the all the sorted labels for each value of k   
        all_sorted_labels = np.empty((len(cluster_range), x_size, y_size))

        # Loop over different numbers of kmeans clusters and shadow cluster threshold
        for c, n_clusters in enumerate(cluster_range):

            # Cluster the image and sort the labels for a single-band image
            if n_bands == 1:
                
                # Initialise the ShadowExtractor class
                ShadExt = ShadowExtractor(cropped_image=cropped_image,
                                        n_clusters=n_clusters,
                                        factor=factor)
                
                # Cluster image so that each pixel is assigned a label
                labels = ShadExt.kmeans_clustering()
                
                # Sort the labels by brightness
                sorted_labels = ShadExt.sort_clusters(labels)
                
                # Calculate and save the average silhouette coefficient for the darkest cluster
                silhouettes[i, c] = ShadExt.calc_silh_coefficient(sorted_labels)
                    
            # Cluster and sort the labels of each band in a multi-band image
            elif n_bands > 1:
                
                # Store the silhouette coefficient/score and labels for each band
                band_silhouettes = np.empty((n_bands))
                band_labels = np.empty((n_bands, x_size, y_size))
                
                # Loop through each band
                for band in np.arange(n_bands):
                    
                    # Initialise the ShadowExtractor class
                    ShadExt = ShadowExtractor(cropped_image=cropped_image[band, :, :],
                                            n_clusters=n_clusters,
                                            factor=factor)
                    
                    # Cluster the individual band so that each pixel is assigned a label
                    labels = ShadExt.kmeans_clustering()
                    
                    # Sort the labels for this band by brightness
                    band_labels[band, :, :] = ShadExt.sort_clusters(labels)
                    
                    # Calculate and save the average silhouette coefficient for the darkest cluster
                    band_silhouettes[band] = ShadExt.calc_silh_coefficient(band_labels[band, :, :])
                
                # Average the silhouette coefficients/scores across the bands
                silhouettes[i, c] = np.mean(band_silhouettes)
                    
                # Calculate modal labels if applied to colour images
                sorted_labels = (mode(band_labels, axis=0).mode).astype(int)
            
            else:
                raise ValueError("Number of bands should not be zero.")
            
            # Store the sorted labels for later and save the number of iterations for this value of k
            all_sorted_labels[c, :, :] = sorted_labels

            # Update the progress bar
            pbar.update(1)

        # Find the labels for k which gave the highest silhouette coefficient/score
        ind = int(np.argmax(silhouettes[i, :]))
                
        # Use the labels which maximised the darkest clusters silhouette coefficient
        raw_shadow = np.where(all_sorted_labels[ind, :, :] == np.amin(all_sorted_labels[ind, :, :]), 1, 0)
        
        # Initialise the PostProcessor class
        PostProc = PostProcessor(shadow=raw_shadow)
        
        # Remove all small shadow detections so only the main shadow remains, then detect and fill bright features in shadow mask
        main_shadow, filled_shadow = PostProc.post_processing()
        
        # Compare the detected shadows to the manually-labelled shadow shapefiles to get testing metrics
        if testing:
            
            # Read in ground truth if testing
            true_shadow, true_bright = DataPrep.read_ground_truth(n_bands, cropped_image, geotransform, projection)
        
            # Calculate the difference between the filled and non-filled shadow to get a mask of any bright features
            if filled_shadow is not None:
                bright_features = filled_shadow - main_shadow
            elif filled_shadow is None:
                bright_features = np.zeros(main_shadow.shape)  
                
            # Initialise ShadowTester class
            ShadTest = ShadowTester(main_shadow=main_shadow,
                                    true_shadow=true_shadow,
                                    bright_features=bright_features,
                                    true_bright=true_bright)
        
            # Calculate and store the precision, recall and F1 scores
            P_shadow[i], R_shadow[i], F1_shadow[i] = ShadTest.calc_shadow_metrics()
            
            # Use the errors (miss rate and FD rate) calculated by comparing to the ground truth
            miss_rate = 1 - R_shadow[i]
            FD_rate = 1 - P_shadow[i]
            
            # Calculate and store the precision, recall and F1 scores
            P_bright[i], R_bright[i], F1_bright[i] = ShadTest.calc_bright_metrics()
            
        # Don't calculate apparent depths if only testing shadow extraction performance
        if not testing:
            
            # Retrieve the correct miss and false discovery rates for the number of bands
            if n_bands == 1:
                miss_rate, FD_rate = miss_rates[0], FD_rates[0]
            elif n_bands > 1:
                miss_rate, FD_rate = miss_rates[1], FD_rates[1]
            else:
                raise ValueError("Number of bands should not be zero.") 
        
        # If no bright features were found within the shadow mask
        if filled_shadow is None:
        
            # Initialise the DepthCalculator class
            DepCalc = DepthCalculator(shadow_list=[main_shadow],
                                    resolution=resolution,
                                    inc_angle=inc_angle,
                                    em_angle=em_angle,
                                    em_angle_par=em_angle_par,
                                    em_angle_perp=em_angle_perp,
                                    solar_azim_angle=solar_azim_angle,
                                    phase_angle=phase_angle)
            
            # Align the main shadow to the Sun's line of sight
            aligned_shadow = DepCalc.align_shadow()
            
            # Measure the observed shadow widths of the aligned shadow [in m]
            S_obs, coords, edge, rim = DepCalc.measure_shadow(aligned_shadow)
            S_obs = S_obs[S_obs != 0]
            
            # Save the rotated image with the detected shadow edge and pit rim overlaid
            if shadows:
                azim_angle = solar_azim_angle * (180 / np.pi)
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(aligned_shadow, cmap='gray', aspect='equal', interpolation='none')
                ax1.axis('off')
                ax1.set_title("Aligned Shadow Mask")
                if n_bands == 1:
                    rotated_image = rotate(cropped_image, azim_angle - 180, resize=True, order=0, mode='constant', cval=np.amax(cropped_image))
                elif n_bands > 1:
                    rotated_image = rotate(cropped_image[0, :, :], azim_angle - 180, resize=True, order=0, mode='constant', cval=np.amax(cropped_image[0, :, :]))
                ax2.imshow(rotated_image, cmap='gray', aspect='equal', interpolation='none')
                ax2.plot(coords, edge, 'r-', linewidth=1, label='Shadow edge')
                ax2.plot(coords, rim, 'c-', linewidth=1, label='Pit rim')
                ax2.axis('off')
                ax2.legend(loc='upper left', bbox_to_anchor=(0, 0), ncol=2, frameon=False)
                ax2.set_title("Aligned Cropped Input Image")
                fig.savefig(os.path.join(plots_dir, os.path.splitext(filename)[0] + '.pdf'), bbox_inches='tight')
            
            # Calculate the upper and lower bounds of the uncertainty in the observed shadow width [in m]
            pos_delta_S_obs = miss_rate * S_obs
            neg_delta_S_obs = FD_rate * S_obs
            
        # If there were bright features found within the shadow mask
        elif filled_shadow is not None:
            
            # Initialise the DepthCalculator class
            DepCalc = DepthCalculator(shadow_list=[main_shadow, filled_shadow],
                                    resolution=resolution,
                                    inc_angle=inc_angle,
                                    em_angle=em_angle,
                                    em_angle_par=em_angle_par,
                                    em_angle_perp=em_angle_perp,
                                    solar_azim_angle=solar_azim_angle,
                                    phase_angle=phase_angle)
            
            # Align the main (non-filled) and filled shadows to the Sun's line of sight
            aligned_main_shadow, aligned_filled_shadow = DepCalc.align_shadow()
            
            # Filter out all shadow pixels which may be caused by bright features within the main shadow
            aligned_filtered_shadow = DepCalc.remove_bright_features(aligned_main_shadow, aligned_filled_shadow)
            
            # Measure the observed shadow widths of the filtered shadow [in m]
            S_obs_filtered, coords_filtered, edge_filtered, rim_filtered = DepCalc.measure_shadow(aligned_filtered_shadow)
            
            # Measure the observed shadow widths of the filled shadow [in m]
            S_obs_filled, coords_filled, edge_filled, rim_filled = DepCalc.measure_shadow(aligned_filled_shadow)
            
            # Find and remove elements where both of the filtered and filled observed width measurements are zero
            zero_filter = np.logical_and(S_obs_filtered != 0, S_obs_filled != 0)
            S_obs_filtered = S_obs_filtered[zero_filter]
            S_obs_filled = S_obs_filled[zero_filter]
            
            # Calculate the average of the filled and filtered observed shadow widths [in m]
            S_obs = (S_obs_filled + S_obs_filtered) / 2
            
            # Calculate the upper and lower bounds of the averaged observed shadow width [in m]
            pos_delta_S_obs = np.maximum(S_obs_filled, S_obs_filtered) - S_obs + (S_obs * miss_rate)
            neg_delta_S_obs = S_obs - np.minimum(S_obs_filled, S_obs_filtered) + (S_obs * FD_rate)                
            
            # Save the rotated image with the detected filtered/filled shadow edge and pit rim overlaid
            if shadows:
                azim_angle = solar_azim_angle * (180 / np.pi)
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(aligned_filtered_shadow, cmap='gray', aspect='equal', interpolation='none')
                ax1.axis('off')
                ax1.set_title("Aligned Shadow Mask")
                if n_bands == 1:
                    rotated_image = rotate(cropped_image, azim_angle - 180, resize=True, order=0, mode='constant', cval=np.amax(cropped_image))
                elif n_bands > 1:
                    rotated_image = rotate(cropped_image[0, :, :], azim_angle - 180, resize=True, order=0, mode='constant', cval=np.amax(cropped_image[0, :, :]))
                ax2.imshow(rotated_image, cmap='gray', aspect='equal', interpolation='none')
                ax2.plot(coords_filled, edge_filled, 'r-', linewidth=1, label='Shadow edge (filled)')
                ax2.plot(coords_filtered, edge_filtered, 'r--', linewidth=1, label='Shadow edge (filtered)')
                ax2.plot(coords_filled, rim_filled, 'c-', linewidth=1, label='Pit rim (filled)')
                ax2.plot(coords_filtered, rim_filtered, 'c--', linewidth=1, label='Pit rim (filtered)')
                ax2.axis('off')
                ax2.legend(loc='upper left', bbox_to_anchor=(0, 0), ncol=2, frameon=False)
                ax2.set_title("Aligned Cropped Input Image")
                fig.savefig(os.path.join(plots_dir, os.path.splitext(filename)[0] + '.pdf'), bbox_inches='tight')
                
        # Calculate the observed apparent depth before correcting the shadow width [in m]
        h_obs = DepCalc.calculate_h(S_obs)
        
        # Calculate the observed length of the shadow [in m]
        L_obs = resolution * np.arange(0, S_obs.size)
        
        # Find the true shadow width and length by correcting for the satellite emission angle [in m]
        S_true, L_true = DepCalc.correct_shadow_width(S_obs, L_obs)
        
        # Calculate the true apparent depth now that the width has been corrected [in m]
        h_true = DepCalc.calculate_h(S_true)
        
        # Propagate the uncertainties in S_obs and the emission angle to h_obs and h_true
        pos_delta_h_obs, neg_delta_h_obs, pos_delta_h_true, neg_delta_h_true = DepCalc.propagate_uncertainties(S_obs, pos_delta_S_obs, neg_delta_S_obs, delta_em_angle)
        
        # Save the apparent depth profile as a CSV file for plotting later
        DataPrep.save_h_profile(L_obs, h_obs, pos_delta_h_obs, neg_delta_h_obs, L_true, h_true, pos_delta_h_true, neg_delta_h_true)
        
        # Save the shadow as a geo-referenced shapefile
        DataPrep.save_shadow(main_shadow, geotransform, projection, h_true)
        
        # Find the centre observed and true apparent depths to add to the results table
        ind = int(h_obs.size / 2)
        centre_raw_hs[i], pos_centre_raw_hs[i], neg_centre_raw_hs[i] = h_obs[ind], pos_delta_h_obs[ind], neg_delta_h_obs[ind]
        centre_hs[i], pos_centre_hs[i], neg_centre_hs[i] = h_true[ind], pos_delta_h_true[ind], neg_delta_h_true[ind]
        
        # Find the maximum observed and true apparent depths to add to the results table
        max_raw_hs[i], pos_max_raw_hs[i], neg_max_raw_hs[i] = np.amax(h_obs), np.amax(pos_delta_h_obs), np.amax(neg_delta_h_obs)
        max_hs[i], pos_max_hs[i], neg_max_hs[i] = np.amax(h_true), np.amax(pos_delta_h_true), np.amax(neg_delta_h_true)
                
    pbar.close()

    # Save testing performances
    if testing:
        
        logging.info("Shadow extraction performance metrics:")
        logging.info("Average miss rate: {}".format(1 - np.mean(R_shadow)))
        logging.info("Average false discovery rate: {}".format(1 - np.mean(P_shadow)))
        logging.info("Average F1 score: {}".format(np.mean(F1_shadow)))
        
        logging.info("Bright feature detection performance metrics:")
        logging.info("Average miss rate: {}".format(1 - np.mean(R_bright)))
        logging.info("Average false discovery rate: {}".format(1 - np.mean(P_bright)))
        logging.info("Average F1 score: {}".format(np.mean(F1_bright)))
        
        # Store testing results in a structured array
        dt = np.dtype([('i', 'U32'),
                    ('P_shadow', float), ('R_shadow', float), ('F1_shadow', float),
                    ('P_bright', float), ('R_bright', float), ('F1_bright', float)])
        array = np.empty(len(filenames), dtype=dt)
        
        # Store the filenames, and the corresponding testing metrics
        array['i'] = [os.path.splitext(filename)[0] for filename in filenames]
        array['P_shadow'] = P_shadow
        array['R_shadow'] = R_shadow
        array['F1_shadow'] = F1_shadow
        array['P_bright'] = P_bright
        array['R_bright'] = R_bright
        array['F1_bright'] = F1_bright
        
        # Save to a csv file with appropriate headers for reference
        np.savetxt(os.path.join(output_dir, 'PITS_testing.csv'), 
                array,
                delimiter=',',
                fmt='%s, %f, %f, %f, %f, %f, %f',
                header='Image Name, P (Shadow), R (Shadow), F1 (Shadow), P (B Spots), R (B Spots), F1 (B Spots)')       
        
    # Store results and image information in a structured array
    dt = np.dtype([('filename', 'U32'),
                    ('res', float), ('inc', float), ('s_azim', float), ('em', float), ('sc_azim', float),
                    ('raw_h_c', float), ('pos_raw_h_c', float), ('neg_raw_h_c', float),
                    ('h_c', float), ('pos_h_c', float), ('neg_h_c', float), 
                    ('raw_h_m', float), ('pos_raw_h_m', float), ('neg_raw_h_m', float),
                    ('h_m', float), ('pos_h_m', float), ('neg_h_m', float)])
    array = np.empty(len(filenames), dtype=dt)
    
    # Store filenames and sensing information
    array['filename'] = [os.path.splitext(filename)[0] for filename in filenames]
    array['res'] = resolutions
    array['inc'] = inc_angles
    array['s_azim'] = solar_azim_angles
    array['em'] = em_angles
    array['sc_azim'] = sc_azim_angles
    
    # Store the uncorrected apparent depths (and uncertainties) at the shadow's centre
    array['raw_h_c'] = centre_raw_hs
    array['pos_raw_h_c'] = pos_centre_raw_hs
    array['neg_raw_h_c'] = neg_centre_raw_hs
    
    # Store the corrected apparent depths (and uncertainties) at the shadow's centre
    array['h_c'] = centre_hs
    array['pos_h_c'] = pos_centre_hs
    array['neg_h_c'] = neg_centre_hs
    
    # Store the uncorrected maximum apparent depths (and uncertainties)
    array['raw_h_m'] = max_raw_hs
    array['pos_raw_h_m'] = pos_max_raw_hs
    array['neg_raw_h_m'] = neg_max_raw_hs
    
    # Store the corrected maximum apparent depths (and uncertainties)
    array['h_m'] = max_hs
    array['pos_h_m'] = pos_max_hs
    array['neg_h_m'] = neg_max_hs

    # Save to a csv file with appropriate headers for reference
    np.savetxt(os.path.join(output_dir, 'PITS_results.csv'), 
            array,
            delimiter=',',
            fmt='%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f',
            header='Image Name, Resolution [m], Incidence Angle [deg], Solar Azimuth Angle [deg], Emission Angle [deg], Spacecraft Azimuth Angle [deg], Uncorrected Centre h [m], +, -, Centre h [m], +, -, Uncorrected Maximum h [m], +, -, Maximum h [m], +, -')

    logging.info("All {} images analysed and outputs saved to {}".format(len(filenames), output_dir))

if __name__ == "__main__":
    main(ARGS.dataset,
        ARGS.cropping,
        ARGS.shadows,
        ARGS.testing,
        ARGS.factor,
        CLUSTER_RANGE,
        MISS_RATES,
        FD_RATES,
        INPUT_DIR,
        METADATA_DIR,
        LABELS_DIR,
        TESTING_DIR,
        OUTPUT_DIR)