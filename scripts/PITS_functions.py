'''
Created by Daniel Le Corre (1,2)* on 29/01/2023
1 - Centre for Astrophysics nd Planetary Science, University of Kent, Canterbury, United Kingdom
2 - Centres d'Etudes et de Recherches de Grasse, ACRI-ST, Grasse, France
* Correspondence email: dl387@kent.ac.uk
'''

import os
import numpy as np
import math as m
from osgeo import gdal, ogr
from sklearn.cluster import *
from sklearn.metrics import silhouette_samples
from skimage.transform import rotate, rescale
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes

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
            cropped_im = np.empty([n_bands, xsize, ysize])
            for n in range(n_bands):
                band = ds.GetRasterBand(n+1)
                cropped_im[n, :, :] = band.ReadAsArray()
        elif n_bands == 1:
            band = ds.GetRasterBand(1)
            cropped_im = band.ReadAsArray()
        else:
            raise ValueError("No raster bands are present in image.")

        return cropped_im, geot, proj, n_bands, xsize, ysize

    def read_ground_truth(self, im, n_bands, geot, proj):

        # Get product name of file
        shp_name = os.path.splitext(self.filename)[0]

        # Find the path to the shadow ground truth
        validation_path = os.path.join(self.training_dir, shp_name + '.shp')

        # Open the shapefile layer
        val_ds = ogr.Open(validation_path)
        val_layer = val_ds.GetLayer()

        # Create the raster data source to store the shadow ground truth
        if n_bands == 1:
            target_ds = driver1.Create(os.path.join(self.output_dir,'validation_{}.tif'.format(shp_name)), im.shape[1], im.shape[0], 1, gdal.GDT_Int16)
        elif n_bands > 1:
            target_ds = driver1.Create(os.path.join(self.output_dir,'validation_{}.tif'.format(shp_name)), im.shape[2], im.shape[1], 1, gdal.GDT_Int16)
        else:
            raise ValueError("Number of bands should not be zero.")
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
        val_array = gdal.Open(os.path.join(self.output_dir,'validation_{}.tif'.format(shp_name))).ReadAsArray()

        # Remove the temporary raster file
        os.remove(os.path.join(self.output_dir,'validation_{}.tif'.format(shp_name)))

        return val_array

    def save_outputs(self, geot, proj, shadow_mask, resolution, x_sh, h, pos_h, neg_h):

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
        h_av = ogr.FieldDefn("max_h", ogr.OFTReal)
        x_sh_av = ogr.FieldDefn("max_x", ogr.OFTReal)
        h_c = ogr.FieldDefn("centre_h", ogr.OFTReal)
        x_sh_c = ogr.FieldDefn("centre_x", ogr.OFTReal)
        shp_layer.CreateField(product)
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

        # Fill in the shadow width and apparent depth fields (when taking x_sh at centre and as an average)
        feature.SetField("centre_h", h[int(h.size/2)])
        feature.SetField("centre_x", x_sh[int(x_sh.size/2)])
        feature.SetField("max_h", np.amax(h))
        feature.SetField("max_x", np.amax(x_sh))

        # Set the feature geometry
        feature.SetGeometry(multipolygon)
            
        # Create the feature in the shapefile layer
        shp_layer.CreateFeature(feature)

        # Close the datasets, bands and layers
        shadow_ds = shadow_band = None
        shp_ds = shp_layer = feature = None

        # Calculate the total length of the shadow
        lengths = resolution*np.arange(-x_sh.size/2, x_sh.size/2, 1)
        
        # Define the results directory
        results_dir = os.path.join(self.output_dir, 'results/')
        
        # Save the depth profile as a csv
        dt = np.dtype([('length', float), ('h', float), ('+', float), ('-', float)])
        array = np.empty(lengths.size, dtype=dt)
        array['length'] = lengths
        array['h'] = h
        array['+'] = pos_h
        array['-'] = neg_h
        np.savetxt(os.path.join(results_dir, '{}_profile.csv'.format(name)), 
                   array, 
                   delimiter=',', 
                   fmt='%f, %f, %f, %f', 
                   header='Shadow Length [m], Apparent Depth [m], +, -')

def kmeans_clustering(cropped_im, cluster):

    # Reshape the cropped image into a vector
    vec_img = cropped_im.reshape(-1, 1)

    # Perform K-means image segmentation into 'n' number of clusters
    kmeans = KMeans(n_clusters=cluster, init='k-means++', n_init=10, tol=0, algorithm='full').fit(vec_img)
    labels = kmeans.predict(vec_img).reshape(cropped_im.shape)

    return labels

def sort_clusters(cropped_im, labels, factor):
    
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

    # Copy the arrays
    cropped_im_copy = np.copy(cropped_im)
    sorted_labels_copy = np.copy(sorted_labels)

    # Downscale the input image and labels to speed up sorting
    x = rescale(cropped_im_copy, factor, anti_aliasing=False)
    X = x.reshape(-1, 1)
    y = rescale(sorted_labels_copy, factor, anti_aliasing=False)
    Y = y.flatten()

    # Calculate the silhouette coefficient
    silhouette_darkest = np.mean(silhouette_samples(X, Y)[Y == 0])
    
    return sorted_labels, silhouette_darkest

def postprocessing(shadow):

    # Label the shadow mask into regions and retrieve properties
    labelled_mask = label(shadow, background=0)
    regions = regionprops(labelled_mask)

    # Find the connected region with the largest area
    region_areas = []
    for region in regions:
        region_areas.append(region.area)
    new_shadow = np.where(labelled_mask == (region_areas.index(max(region_areas))+1), 1, 0)

    # Fill in holes in the holes in the shadow mask
    new_shadow = binary_fill_holes(new_shadow)

    return new_shadow

def measure_shadow(shadow_mask, azim_angle, resolution):

    # Align shadow mask using the sub-solar azimuth angle
    aligned_mask = rotate(shadow_mask.astype(np.uint8), azim_angle-90, resize=True, order=0, mode='constant', cval=0, preserve_range=True)
    
    # Find shadow edge and rim
    coords = np.where(aligned_mask == 1)
    xmin, xmax, ymin, ymax = np.amin(coords[1]), np.amax(coords[1]), np.amin(coords[0]), np.amax(coords[0])
    length = np.arange(xmin, xmax + 1)
    edge = np.empty(length.size)
    rim = np.empty(length.size)
    for e, l in enumerate(length):
        edge[e] = np.amin(np.where(aligned_mask[:, l] == 1)[0])
        rim[e] = np.amax(np.where(aligned_mask[:, l] == 1)[0])

    # Measure the shadow width at every point
    x_sh = resolution*np.abs(edge-rim)
    # x_sh = resolution*np.sum(aligned_mask, axis=0)

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