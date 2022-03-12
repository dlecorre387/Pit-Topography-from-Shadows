import os
import json
import numpy as np
from analyse_HiRISE_image import ImageAnalyser
from analyse_HiRISE_DTM import DTMAnalyser

MODE = 'rdr'
CROPPING = True
INPUT_DIR='/data/input/'
METADATA_DIR='/data/metadata/'
LABELS_DIR='/data/labels/'
OUTPUT_DIR='/data/output/'
DTM_DIR='/data/dtm/'
ORTHO_DIR='/data/ortho/'
N_CLUSTERS=[4, 5, 6, 7, 8]

def main(mode, 
        cropping,
        input_dir, 
        metadata_dir, 
        labels_dir, 
        output_dir, 
        dtm_dir,
        ortho_dir,
        n_clusters):

    # Clean output folder
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Open dictionary to store data
    data_dict = dict()

    # Check mode
    if mode == 'rdr' or mode == 'both':

        # Loop through each image
        for filename in os.listdir(input_dir):
            
            # Skip any aux.xml files in the folder
            if not filename.endswith('.xml'):

                # Get product name of file
                name = os.path.splitext(filename)[0]

                print("-"*100)
                print("Analysing HiRISE RDR Product: {}".format(name))

                # Create a dict entry for each file
                data_dict[name] = {}

                # Initialise ImageAnalyser class
                ImAn = ImageAnalyser(filename,
                                    input_dir, 
                                    metadata_dir, 
                                    labels_dir, 
                                    output_dir, 
                                    n_clusters)
                
                # Retrieve HiRISE metadata
                resolution, inc_angle, em_angle, azim_angle = ImAn.read_HiRISE_metadata()

                # Populate the dictionary with pit info
                data_dict[name]['Resolution [m]'] = str(resolution)
                data_dict[name]['Solar Incidence Angle [deg]'] = str(inc_angle)
                data_dict[name]['Sub-Solar Azimuth Angle [deg]'] = str(azim_angle)
                data_dict[name]['Emission Angle [deg]'] = str(em_angle)
                
                # Check if input images need to be cropped
                if cropping:

                    # Crop the HiRISE image using provided labels
                    cropped_im, geot, proj = ImAn.crop_HiRISE_image()

                elif not cropping:

                    # Read in the pre-cropped raster file as an array
                    cropped_im, geot, proj = ImAn.read_cropped_im()

                else:
                    raise ValueError("'CROPPING' in run_MAPS.py should be equal to True or False.")

                # Label the cropped image using KMeans clustering
                widths, shadows, sorted_labels = ImAn.cluster_image(cropped_im, azim_angle, resolution)

                # Save the output raster and vector layers
                ImAn.save_outputs(geot, proj, shadows)

                # Check that a shadow was found
                if len(widths) == 0:
                    print("Apparent depth, h could not be measured for file {}, as no shadow widths could be measured.\n".format(filename))
                
                # Print shadow width and apparent depth estimates
                else:

                    # Average the shadow widths for each number of clusters
                    sigma_S = np.std(widths)
                    if sigma_S < 0.25:
                        sigma_S = 0.25
                    av_S = np.mean(widths)
                    print("Average Shadow Width, S = {} +/- {} m".format(np.around(av_S, decimals=1), np.around(sigma_S, decimals=1)))

                    # Estimate the apparent pit depth, h
                    h, sigma_h = ImAn.estimate_pit_depth(av_S, sigma_S, inc_angle)
                    print("Apparent Depth at Non-Rim-Edge of Shadow, h = {} +/- {} m".format(np.around(h, decimals=1), np.around(sigma_h, decimals=1)))

                    # Populate the dictionary with pit info
                    data_dict[name]['Apparent Depth, h [m]'] = str(h) + ' +/- ' + str(sigma_h)
                    data_dict[name]['Average Shadow Width [m]'] = str(av_S) + ' +/- ' + str(sigma_S)

    # Check mode
    if mode == 'dtm' or mode == 'both':
        
        # If DTM files have been given
        if os.path.exists(dtm_dir) and len(os.listdir(dtm_dir)) > 0:

            for filename in os.listdir(dtm_dir):

                print("-"*100)
                print("Analysing HiRISE DTM Product: {}".format(filename))

                # Find the first and second orbit numbers of the DTM
                orbit_no1 = filename.split('_')[1]
                orbit_no2 = filename.split('_')[3]

                DTMAn = DTMAnalyser(filename,
                                    None, 
                                    dtm_dir, 
                                    labels_dir, 
                                    output_dir,
                                    n_clusters)

                if cropping:
                    
                    # Read and crop the DTM file
                    cropped_dtm, geot, proj = DTMAn.read_and_crop_DTM()

                elif not cropping:

                    # Read in pre-cropped DTM
                    cropped_dtm, geot, proj = DTMAn.read_cropped_DTM()

                else:
                    raise ValueError("'CROPPING' in run_MAPS.py should be equal to True or False.")

                # Find the rate of change of gradient of DTM
                gradient, curvature, footprint = DTMAn.find_footprint(cropped_dtm, geot, proj)

                # Find the corresponding orthographic images
                for orbit_no in [orbit_no1, orbit_no2]:
                    
                    filename = [fname for fname in os.listdir(ortho_dir) if orbit_no in fname][0]

                    print("Analysing HiRISE Orthographic Product: {}".format(filename))

                    ImAn = ImageAnalyser(filename,
                                        ortho_dir,
                                        None,
                                        labels_dir,
                                        output_dir,
                                        n_clusters)

                    # Check if input images need to be cropped
                    if cropping:

                        # Crop the HiRISE ortho image using provided labels
                        cropped_ortho, geot, proj = ImAn.crop_HiRISE_image()

                    elif not cropping:

                        # Read in the pre-cropped raster file as an array
                        cropped_ortho, geot, proj = ImAn.read_cropped_im()

                    else:
                        raise ValueError("'CROPPING' in run_MAPS.py should be equal to True or False.")

                    DTMAn = DTMAnalyser(None,
                                    filename, 
                                    dtm_dir, 
                                    labels_dir, 
                                    output_dir,
                                    n_clusters)

                    # Produce shadow masks of orthographic image using Kmeans clustering
                    shadows = DTMAn.cluster_ortho_image(cropped_ortho)

                    # Save the output raster and vector layers
                    DTMAn.save_outputs(geot, proj, shadows)

        else:
            raise SystemError("DTM path not found or no DTM files present.")

    elif mode != 'rdr' and mode != 'dtm' and mode != 'both':
        raise SystemError("'MODE' in run_MAPS.py must be one of 'rdr', 'dtm' or 'both'.")

    # Save the data dictionary
    with open(output_dir + 'pit_data_dict.txt', 'w') as f:
        f.write(json.dumps(data_dict, sort_keys=True, indent=5))

if __name__ == "__main__":
    main(MODE, 
        CROPPING,
        INPUT_DIR,
        METADATA_DIR,
        LABELS_DIR,
        OUTPUT_DIR,
        DTM_DIR,
        ORTHO_DIR,
        N_CLUSTERS)