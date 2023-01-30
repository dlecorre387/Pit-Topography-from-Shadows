'''
Created by Daniel Le Corre (1,2)* on 29/01/2023
1 - Centre for Astrophysics nd Planetary Science, University of Kent, Canterbury, United Kingdom
2 - Centres d'Etudes et de Recherches de Grasse, ACRI-ST, Grasse, France
* Correspondence email: dl387@kent.ac.uk
'''

import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Configure logger
logging.basicConfig(level = logging.INFO,
                    format='| %(asctime)s | %(levelname)s | Message: %(message)s',
                    datefmt='%d/%m/%y @ %H:%M:%S')

'''
Optional Parameters:

-e (--elimit):  Limit to apply to images' emission angle before plotting results in deg.
                Default: 10 / Type: float


DO NOT CHANGE THESE VARIABLES:

OUTPUT_DIR:     Path to the output directory where all of the results and plots will be saved. (Type: str)

'''

# Initialize arguments parser and define arguments
PARSER = argparse.ArgumentParser()
PARSER.add_argument("-e", "--elimit", type=float, default=10, help = "Emission angle limit [deg]")
ARGS = PARSER.parse_args()

# Do not change these variables
OUTPUT_DIR = '/data/output/'

def main(e_limit, output_dir):
    
    # Create the directory to save plots
    plots_dir = os.path.join(output_dir, 'plots/')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    logging.info("Plots output folder created")
    
    # Define the results directory
    results_dir = os.path.join(output_dir, 'results/')
    
    # Define the path to the results csv file
    csv_path = os.path.join(results_dir, 'PITS_results.csv')
    
    # Read in the data in the APP_results csv file
    im_name, res, inc_angle, azim_angle, em_angle, h_c, pos_h_c, neg_h_c, h_m, pos_h_m, neg_h_m, = np.genfromtxt(csv_path, 
                                                                                                                delimiter=',',
                                                                                                                unpack=True, 
                                                                                                                skip_header=1,
                                                                                                                dtype=str)

    # Convert relevant data to float
    em_angle = em_angle.astype(float)
    h_c = h_c.astype(float)
    pos_h_c = pos_h_c.astype(float)
    neg_h_c = neg_h_c.astype(float)
    h_m = h_m.astype(float)
    pos_h_m = pos_h_m.astype(float)
    neg_h_m = neg_h_m.astype(float)
    
    # If a limit to the emission is to be applied (10 deg by default)
    if e_limit != None:
        
        # Find the images with allowed emission angles
        mask = em_angle <= e_limit
        
        # Filter out the results for images with too large emission angles
        h_c = h_c[mask]
        h_m = h_m[mask]
    
    # Plot a histogram of the apparent depth values
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Find the largest predicted depth
    maximum = max(np.amax(h_c), np.amax(h_m))
    
    # Plot the histograms for the centre and max h (h_c/h_m)
    ax.hist(h_c, label=r'$h_c$', range=(0, maximum), alpha=1, histtype='bar', color='SteelBlue', zorder=3.5)
    ax.hist(h_m, label=r'$h_m$', range=(0, maximum), alpha=1, histtype='step', color='tomato', zorder=3.5)
    
    # Format axes
    ax.set_ylim(0, maximum)
    ax.set_xlim(0)
    ax.set_ylabel('Count')
    ax.set_xlabel(r'Apparent depth ($h$) [m]')
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=2, frameon=False)
    
    # Save the figure as a pdf
    save_path = os.path.join(plots_dir, 'h_histogram.pdf')
    fig.savefig(save_path, bbox_inches='tight')
    
    logging.info("Histogram of apparent depths plotted and saved to {}".format(save_path))
    
    # Find all h profile CSV files
    filenames = [file for file in os.listdir(results_dir) if file.endswith('profile.csv')]

    for i, filename in enumerate(filenames):
        
        # Get product name of file
        name = os.path.splitext(filename)[0]
        
        # Define the path to the h profile CSV file
        path = os.path.join(results_dir, filename)
        
        # Read the profile CSV file
        lengths, h, pos_h, neg_h = np.genfromtxt(path, delimiter=',', unpack=True, skip_header=1, dtype=float)
        
        # Interpolate h values to get h at centre
        step = np.amin(np.abs(lengths))
        if step != 0:
            new_lengths = np.arange(lengths[0], lengths[-1], step)
            f_h = interp1d(lengths, h)
            new_h = f_h(new_lengths)
            f_pos_h = interp1d(lengths, pos_h)
            new_pos_h = f_pos_h(new_lengths)
            f_neg_h = interp1d(lengths, neg_h)
            new_neg_h = f_neg_h(new_lengths)
            h_c = new_h[new_lengths == 0][0]
            pos_h_c = new_pos_h[new_lengths == 0][0]
            neg_h_c = new_neg_h[new_lengths == 0][0]
        else:
            h_c = h[lengths == 0][0]
            pos_h_c = pos_h[lengths == 0][0]
            neg_h_c = neg_h[lengths == 0][0]
        
        # Find the maximum apparent depth
        h_m = np.amax(h)
        pos_h_m, neg_h_m = pos_h[h == h_m][0], neg_h[h == h_m][0]
        max_depth = np.amax(h) + pos_h_m
        
        # Find the ratio to scale the x and y axis by
        ratio = lengths.size/max_depth

        # Plot the apparent depth profile
        fig, ax = plt.subplots(figsize=(ratio, 2))
        ax.plot(lengths, h, color='steelblue')

        # Plot the uncertainties
        ax.fill_between(lengths, h - neg_h, h + pos_h, alpha=0.3, color='steelblue')

        # Plot the maximum and centre apparent depth
        label = r"$h_{c}$ = " + f"{np.around(h_c, decimals=2)} (+ {np.around(pos_h_c, decimals=2)} / - {np.around(neg_h_c, decimals=2)}) m"
        ax.vlines(x=0, ymin=0, ymax=np.ceil(max_depth/10)*10, linestyle='dashed', color='black', label=label)
        label = r"$h_{m}$ = " + f"{np.around(h_m, decimals=2)} (+ {np.around(pos_h_m, decimals=2)} / - {np.around(neg_h_m, decimals=2)}) m"
        ax.vlines(x=lengths[h == h_m], ymin=0, ymax=np.ceil(max_depth/10)*10, linestyle='dashed', color='tomato', label=label)

        # Format the axes        
        ax.set_xlabel("Lengthways distance from centre of shadow [m]")
        ax.set_ylabel(r"Apparent depth ($h$) [m]")
        ax.set_ylim(0, np.ceil(max_depth/10)*10)
        ax.set_xlim(lengths[0], lengths[-1])
        ax.invert_yaxis()
        ax.grid('both')
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=2, frameon=False)

        # Save the figure to the output path
        fig.savefig(os.path.join(plots_dir, name + '.pdf'), bbox_inches='tight')
        
    logging.info("Apparent depth profiles plotted and saved to {}".format(plots_dir))
    
if __name__ == '__main__':
    main(ARGS.elimit,
        OUTPUT_DIR)