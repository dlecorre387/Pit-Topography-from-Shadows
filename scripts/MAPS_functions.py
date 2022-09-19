import os

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import math as m
import matplotlib.pyplot as plt

from osgeo import gdal, ogr
from sklearn.cluster import *
from sklearn.metrics import silhouette_score
from skimage.transform import rotate, rescale
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from scipy.stats import mode

# Define drivers for writing vector and raster data
driver1 = gdal.GetDriverByName("GTiff")
driver2 = ogr.GetDriverByName("ESRI Shapefile")

class ImageAnalyser(object):
    
    def __init__(self,
                filename,
                input_dir,
                metadata_dir,
                labels_dir,
                training_dir,
                output_dir):

        self.filename = filename
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.labels_dir = labels_dir
        self.training_dir = training_dir
        self.output_dir = output_dir
        
    def read_HiRISE_metadata(self):
        
        # Get product name of file
        name = os.path.splitext(self.filename)[0]

        # Open the .LBL file containing sensing info
        try:
            lines = open(os.path.join(self.metadata_dir, name +'.LBL'), 'r').readlines()
        except:
            raise SystemError("Label file could not be opened for " + name)

        # Check that no files are empty
        if len(lines) == 0:
            raise ValueError(".LBL file is empty")

        # Retrieve viewing parameters from metadata file
        param_names = ['MAP_SCALE    ','INCIDENCE_ANGLE', 'EMISSION_ANGLE', 'SUB_SOLAR_AZIMUTH']
        params = []
        for param_name in param_names:
            param_line = [line for line in lines if param_name in line]
            param = param_line[0].split()[2]
            params.append(param)
        
        # Convert from strings to floats
        resolution, inc_angle, em_angle, azim_angle = float(params[0]), float(params[1]), float(params[2]), float(params[3])

        return resolution, inc_angle, em_angle, azim_angle      

    def crop_image(self):
        
        # Get product name of file
        name = os.path.splitext(self.filename)[0]
        
        # For augmented images, use the label for the original image
        if self.filename.startswith('blurred_') or self.filename.startswith('noised_'):
            
            # Removed the prefix from the filename
            name = '_'.join(name.split('_')[1:])
        
        # Rename labels to match corresponding image
        for label in os.listdir(self.labels_dir):
        
            # Find label name and extension
            label_name, ext = os.path.splitext(label)[0], os.path.splitext(label)[1]
            
            # Skip all labels except the one corresponding to the input image
            if not name in label:
                continue
            
            # If the label name contains the same product code
            elif name in label and name != label_name:
                old_path = os.path.join(self.labels_dir, label)
                new_path = os.path.join(self.labels_dir, str(name + ext))
                os.rename(old_path, new_path)

            # Do nothing if the names are already equal
            elif name in label and name == label_name:
                continue

        # Define the folder to store the cropped images
        cropped_dir = os.path.join(self.output_dir, 'cropped/')
        if not os.path.exists(cropped_dir):
            os.makedirs(cropped_dir)

        # Define filenames for the labels, input image, and cropped output
        input_path = os.path.join(self.input_dir, self.filename)
        shp_path = os.path.join(self.labels_dir, name + '.shp')
        output_path = os.path.join(cropped_dir, os.path.splitext(self.filename)[0] + '.tif')

        # Clip the input image using the shapefile label
        crop = gdal.Warp(output_path, 
                    input_path, 
                    cutlineDSName=shp_path,
                    cropToCutline=True,
                    dstNodata=None)
        
        # Close the raster dataset to save it
        crop = None

        # Read croped raster file as a NumPy array
        counter = 0
        if (os.path.splitext(self.filename)[0] + '.tif') in os.listdir(cropped_dir):
            ds = gdal.Open(output_path) # GDAL Dataset
            geot = ds.GetGeoTransform() # GeoTransform
            proj = ds.GetProjection()   # Projection
            xsize = ds.RasterYSize
            ysize = ds.RasterXSize
            n_bands = ds.RasterCount
            if n_bands > 1:
                cropped_im = np.empty([n_bands, xsize, ysize])
                for n in range(n_bands):
                    band = ds.GetRasterBand(n+1)
                    cropped_im[n, :, :] = band.ReadAsArray()
            elif n_bands == 1:
                band = ds.GetRasterBand(1)
                cropped_im = band.ReadAsArray()
            else:
                raise ValueError("No raster bands are present in image.")
        else:
            raise ValueError("No cropped image file present.")

        # # Remove the cropped image
        # os.remove(output_path)

        return cropped_im, geot, proj, n_bands, xsize, ysize

    def read_cropped_im(self):
        
        # Define the path to the image
        im_path = os.path.join(self.input_dir, self.filename)
        
        # Open the cropped raster file as a NumPy array
        ds = gdal.Open(im_path)            # GDAL Dataset
        geot = ds.GetGeoTransform() # GeoTransform
        proj = ds.GetProjection()   # Projection
        n_bands = ds.RasterCount
        xsize = ds.RasterYSize
        ysize = ds.RasterXSize
        if n_bands > 1:
            cropped_im = np.empty([n_bands, band.shape[0], band.shape[1]])
            for n in range(n_bands):
                band = ds.GetRasterBand(n+1)
                cropped_im[n, :, :] = band.ReadAsArray()
        elif n_bands == 1:
            band = ds.GetRasterBand(1)
            cropped_im = band.ReadAsArray()
        else:
            raise ValueError("No raster bands are present in image.")

        return cropped_im, geot, proj, n_bands, xsize, ysize

    def read_ground_truth(self, im, geot, proj):

        # Get product name of file
        name = os.path.splitext(self.filename)[0]

        # Define path to the folder containing shadow labels
        shadow_labels_dir = os.path.join(self.training_dir, 'shadows/')

        # Find the path to the shadow ground truth
        validation_path = os.path.join(shadow_labels_dir, name + '.shp')

        # Open the shapefile layer
        val_ds = ogr.Open(validation_path)
        val_layer = val_ds.GetLayer()

        # Create the raster data source to store the shadow ground truth
        target_ds = driver1.Create(os.path.join(self.output_dir,'validation_{}.tif'.format(name)), im.shape[1], im.shape[0], 1, gdal.GDT_Int16)
        target_ds.SetGeoTransform(geot)
        target_ds.SetProjection(proj)
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(np.nan)
        band.FlushCache()

        # Rasterise the validation polygons so that you get a raster where 1=shadow and 0=background
        gdal.RasterizeLayer(target_ds, [1], val_layer, None, None, options=['ATTRIBUTE=class'], callback=None)
        
        # Close the raster dataset and band
        target_ds = band = None

        # Read the rasterised validation data as an array
        val_array = gdal.Open(os.path.join(self.output_dir,'validation_{}.tif'.format(name))).ReadAsArray()

        # Remove the temporary raster file
        os.remove(os.path.join(self.output_dir,'validation_{}.tif'.format(name)))

        return val_array

    def save_outputs(self, geot, proj, shadow_mask, mask_av_val, x_sh, h, pos_sigma_h, neg_sigma_h, av_h):

        # Get product name of file
        name = os.path.splitext(self.filename)[0]

        # Define the shadows directory
        shadows_dir = os.path.join(self.output_dir, 'shadows/')

        # Rasterise the shadow mask
        shadow_ds = driver1.Create(os.path.join(shadows_dir, name + '_shadow.tif'), 
                                shadow_mask.shape[1], 
                                shadow_mask.shape[0], 
                                1, 
                                gdal.GDT_Int16)
        shadow_ds.SetGeoTransform(geot)
        shadow_ds.SetProjection(proj)
        shadow_band = shadow_ds.GetRasterBand(1)
        shadow_band.WriteArray(shadow_mask)
        shadow_band.SetNoDataValue(np.nan)
        shadow_band.FlushCache()

        # Create the shapefile layer to store the shadow polygon
        shp_ds = driver2.CreateDataSource(os.path.join(shadows_dir, name + '_shadow.shp'))
        shp_layer = shp_ds.CreateLayer('shadow', srs=None)

        # Create the attribute fields to store info
        product = ogr.FieldDefn("prod_code", ogr.OFTString)
        conf = ogr.FieldDefn("conf", ogr.OFTString)
        h_av = ogr.FieldDefn("max_h", ogr.OFTReal)
        x_sh_av = ogr.FieldDefn("max_x", ogr.OFTReal)
        h_c = ogr.FieldDefn("centre_h", ogr.OFTReal)
        x_sh_c = ogr.FieldDefn("centre_x", ogr.OFTReal)
        shp_layer.CreateField(product)
        shp_layer.CreateField(conf)
        shp_layer.CreateField(x_sh_av)
        shp_layer.CreateField(h_av)
        shp_layer.CreateField(x_sh_c)
        shp_layer.CreateField(h_c)

        # Polygonise the raster dataset into the shapefile layer
        gdal.Polygonize(shadow_band, shadow_band, shp_layer, -1, [], callback=None)

        # Set up a multipolygon wkb to merge the individual features
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        # Loop through each feature to add it to the multipolygon
        for feat in shp_layer:
            
            # Get the feature's geometry and add it to the multipolygon
            geom = feat.GetGeometryRef()
            multipolygon.AddGeometry(geom)

            # Delete the feature
            shp_layer.DeleteFeature(feat.GetFID())

        # Get the merged feature from the layer
        feature = ogr.Feature(shp_layer.GetLayerDefn())

        # Fill in the product code field
        feature.SetField("prod_code", name)

        # Fill in the confidence field based on if the shadow mask was sufficiently dark        
        if mask_av_val > 100:
            confidence = 'low'
        else:
            confidence = 'high'
        feature.SetField("conf", confidence)

        # Fill in the shadow width and apparent depth fields (when taking x_sh at centre and as an average)
        feature.SetField("centre_h", h[int(h.size/2)])
        feature.SetField("centre_x", x_sh[int(x_sh.size/2)])
        feature.SetField("max_h", av_h)
        feature.SetField("max_x", np.max(x_sh))

        # Set the feature geometry
        feature.SetGeometry(multipolygon)
            
        # Create the feature in the shapefile layer
        shp_layer.CreateFeature(feature)

        # Close the datasets, bands and layers
        shadow_ds = shadow_band = None
        shp_ds = shp_layer = feature = None

        # Find the maximum calculated apparent depth of the pit
        max_depth = np.amax(h) + pos_sigma_h[h == np.amax(h)][0]

        # Calculate the total length of the shadow
        length = x_sh.size
        lengths = np.arange(-x_sh.size/2, x_sh.size/2, 1)

        # Find the ratio to scale the x and y axis by
        ratio = length/max_depth

        # Plot the apparent depth profile
        fig, ax = plt.subplots(figsize=(8, 8/ratio))
        ax.plot(lengths, h)

        # Plot the uncertainties
        ax.fill_between(lengths, h - neg_sigma_h, h + pos_sigma_h, alpha=0.3)

        # Plot the apparent depth calculated when using the average shadow width
        ax.hlines(y=av_h, xmin=-x_sh.size/2, xmax=x_sh.size/2, linestyle='dashed', color='red')
        text = r"$h_{max}$ = " + str(np.around(av_h, decimals=2)) + ' m'
        plt.text(x=lengths[-1]+length/80, y=av_h, s=text, ha='left', va='center', color='red')

        # # Plot the apparent depth when using the shadow width at the centre of the mask
        # ax.vlines(x=0, ymin=-10, ymax=np.ceil(max_depth/10)*10, linestyle='dotted', color='red')
        # text = r"$h_{c}$ = " + str(np.around(h[int(h.size/2)], decimals=2)) + ' m'
        # plt.text(x=0+length/80, y=0, s=text, ha='left', va='top', color='red')

        # Format the axes        
        ax.set_xlabel("Lengthways distance from centre of shadow [m]")
        ax.set_ylabel(r"Apparent depth, $h$ [m]")
        ax.set_ylim(-10, np.ceil(max_depth/10)*10)
        ax.set_xlim(lengths[0], lengths[-1])
        ax.invert_yaxis()
        ax.grid('both')

        # Define the plots directory
        plots_dir = os.path.join(self.output_dir, 'plots/')

        # Save the figure to the output path
        fig.savefig(os.path.join(plots_dir, name + '_h_profile.png'), bbox_inches='tight')

def kmeans_clustering(cropped_im, cluster):

    # Reshape the cropped image into a vector
    vec_img = cropped_im.reshape(-1, 1)

    # Perform K-means image segmentation into 'n' number of clusters
    kmeans = KMeans(n_clusters=cluster, init='k-means++', n_init=10, tol=0, algorithm='full').fit(vec_img)
    labels = kmeans.predict(vec_img).reshape(cropped_im.shape)

    # Get training info
    iterations = kmeans.n_iter_
    inertia = kmeans.inertia_

    # Calculate the silhouette coefficient
    x = rescale(cropped_im, 0.05, anti_aliasing=False)
    X = x.reshape(-1, 1)
    y = rescale(labels, 0.05, anti_aliasing=False)
    Y = y.flatten()
    silhouette_avg = silhouette_score(X, Y)

    return labels, iterations, inertia, silhouette_avg

def sort_clusters(cropped_im, labels):

    # # If greyscale images are used
    # if n_bands == 1:
        
    # Loop through each cluster to find its average pixel value
    av_val = []
    for n in np.unique(labels):

        # Multiply the crop by the cluster mask to get just the pixels underneath
        masked_im = cropped_im*np.where(labels == n, 1, -1)

        # Flatten the pixel values into a list and remove zeroes
        pixel_list = masked_im.flatten()
        pixel_list = pixel_list[pixel_list > 0]
        
        # Append the average pixel value of this cluster to the list av_val
        av_val.append(np.mean(pixel_list))

    # Sort the clusters' average pixel values from lowest to highest
    sorted_av_val = sorted(av_val)

    # Reassign label numbers in order of average pixel value (i.e. brightness)
    sorted_labels = np.copy(labels)
    for v, value in enumerate(sorted_av_val):
        index = np.where(av_val==value)
        sorted_labels = np.where(labels == index, v, sorted_labels)

    # # For multi-band images
    # if n_bands > 1:

    #     # Sort the labels of each band
    #     for band in np.arange(0, n_bands, 1):
            
    #         # Loop through each cluster to find its average pixel value
    #         av_val = []
    #         for n in np.unique(labels[band, :, :]):

    #             # Multiply the crop by the cluster mask to get just the pixels underneath
    #             masked_im = cropped_im[band, :, :]*np.where(labels[band, :, :] == n, 1, -1)

    #             # Flatten the pixel values into a list and remove zeroes
    #             pixel_list = masked_im.flatten()
    #             pixel_list = pixel_list[pixel_list > 0]
                
    #             # Append the average pixel value of this cluster to the list av_val
    #             av_val.append(np.mean(pixel_list))

    #         # Sort the clusters' average pixel values from lowest to highest
    #         sorted_av_val = sorted(av_val)

    #         # Reassign label numbers in order of average pixel value (i.e. brightness)
    #         sorted_labels = np.copy(labels[band, :, :])
    #         for v, value in enumerate(sorted_av_val):
    #             index = np.where(av_val==value)
    #             sorted_labels = np.where(labels[band, :, :] == index, v, sorted_labels)
    #         labels[band, :, :] = sorted_labels

    #     # Calculate average labels if applied to colour images
    #     sorted_labels = (mode(labels, axis=0).mode).astype(int)

    return sorted_labels

def postprocessing(shadow):

    # Label the shadow mask into regions and retrieve properties
    labelled_mask = label(shadow, background=0)
    regions = regionprops(labelled_mask)

    # Find the connected region with the largest area
    region_areas = []
    for region in regions:
        region_areas.append(region.area)
    shadow = np.where(labelled_mask == (region_areas.index(max(region_areas))+1), 1, 0)

    # Fill in holes in the holes in the shadow mask
    shadow = binary_fill_holes(shadow)

    return shadow

def measure_shadow(shadow_mask, geot, proj, azim_angle, resolution):

    # Align shadow mask using the sub-solar azimuth angle
    aligned_mask = rotate(shadow_mask.astype(np.uint8), azim_angle-90, resize=True, order=1, mode='constant', cval=0, preserve_range=True)

    # Find the x coordinates of the shadow mask
    xs = np.where(aligned_mask != 0)[1]
    x_coords = np.arange(min(xs), max(xs) + 1)
    
    # # Measure shadow width along the centre of the mask
    # x_m = int(np.around(((max(x_coords) - min(x_coords))/2) + min(x_coords)))
    # x_sh = resolution*np.sum(aligned_mask[:, x_m])

    # Measure the shadow width at every point
    x_sh = resolution*np.sum(aligned_mask[:, x_coords], axis=1)

    return x_sh

def calculate_h(x_sh, pos_sigma_x_sh, neg_sigma_x_sh, inc_angle):
        
    # Find angle between solar line of sight and surface
    solar_angle = 90 - inc_angle

    # Convert the solar angle from degrees to rads
    solar_angle = solar_angle*(np.pi/180)

    # Calculate the apparent depth of the pit in metres
    h = x_sh*m.tan(solar_angle)

    # Propagate uncertainty
    pos_sigma_h = pos_sigma_x_sh*m.tan(solar_angle)
    neg_sigma_h = neg_sigma_x_sh*m.tan(solar_angle)

    return h, pos_sigma_h, neg_sigma_h