'''
Created by Daniel Le Corre (1,2)* 
Last edited on 21/07/2023
1   Centre for Astrophysics and Planetary Science, University of Kent, Canterbury, United Kingdom
2   Centres d'Etudes et de Recherches de Grasse, ACRI-ST, Grasse, France
*   Correspondence: dl387@kent.ac.uk
    Website: https://www.danlecorre.com/
    
This project is part of the Europlanet 2024 RI which has received
funding from the European Union’s Horizon 2020 research and innovation
programme under grant agreement No 871149.

'''

import os
import numpy as np
from osgeo import gdal, ogr
from sklearn.cluster import *
from sklearn.metrics import silhouette_samples
from skimage.transform import rotate, rescale
from skimage.measure import label, regionprops
from skimage.morphology import area_closing
from scipy.ndimage import binary_fill_holes

# Define drivers for writing vector and raster data
driver1 = gdal.GetDriverByName("GTiff")
driver2 = ogr.GetDriverByName("ESRI Shapefile")

class DataPreparer(object):
    
    '''
    DataPreparer is responsible for preparing all the necessary data for running the PITS algorithm. 
    This includes:
    -   Reading in all geo-referenced input imagery as NumPy arrays using the GDAL and OGR Python APIs
    -   Cropping these images to the extents of the pit feature if necessary using the user-provided 
        pit location shapefiles
    -   Retrieving the image metadata such as the solar incidence, emission, sub-solar azimuth and sub-
        spacecraft azimuth angles
    '''
    
    def __init__(self,
                filename,
                input_dir,
                metadata_dir,
                labels_dir,
                testing_dir,
                output_dir):

        self.filename = filename
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.labels_dir = labels_dir
        self.testing_dir = testing_dir
        self.output_dir = output_dir
    
    def crop_image(self):
        
        # Get the product identification of the image from the filename
        name = os.path.splitext(self.filename)[0]
                
        # Rename shapefile location label to match corresponding image
        for label in os.listdir(self.labels_dir):
        
            # Find label name and extension
            label_name, ext = os.path.splitext(label)[0], os.path.splitext(label)[1]
            
            # Skip all labels except the one corresponding to the input image
            if not name in label:
                continue
            
            # If the label name contains the same product code
            elif name in label and name != label_name and label_name.endswith(name):
                old_path = os.path.join(self.labels_dir, label)
                new_path = os.path.join(self.labels_dir, str(name + ext))
                os.rename(old_path, new_path)

            # Do nothing if the names are already equal
            elif name in label and name == label_name:
                continue

        # Define the folder to store the cropped image
        cropped_dir = os.path.join(self.output_dir, 'cropped/')
        if not os.path.exists(cropped_dir):
            os.makedirs(cropped_dir)

        # Define the path to the input image, location label and cropped output
        input_path = os.path.join(self.input_dir, self.filename)
        labels_path = os.path.join(self.labels_dir, name + '.shp')
        cropped_path = os.path.join(cropped_dir, name + '.tiff')
        
        # Open the shapefile location label as a vector dataset
        label_dataset = driver2.Open(labels_path, 1)
        
        # Get vector layer
        label_layer = label_dataset.GetLayer()

        # Get the extent of the shapefile location label
        label_extent = label_layer.GetExtent()
        
        # Get the min/max latitude and longitude of the cropped image
        min_longitude, max_longitude, min_latitude, max_latitude = label_extent
                
        # Clip the input image to the extents of the user-provided shapefile location label
        crop = gdal.Warp(cropped_path, 
                        input_path, 
                        cutlineDSName=labels_path,
                        cropToCutline=True,
                        dstNodata=None)
        
        # Close the raster dataset to save it
        crop = None

        # Check that cropped image is present
        if (name + '.tiff') in os.listdir(cropped_dir):
            
            # Open the cropped image as a GDAL raster dataset
            dataset = gdal.Open(cropped_path)
            
            # Get the geotransform of the cropped image
            geotransform = dataset.GetGeoTransform()
            
            # Get the resolution of the cropped image
            resolution = geotransform[1]
            
            # Get the map-projection of the cropped image
            projection = dataset.GetProjection()
            
            # Get the horizontal and vertical pixel sizes
            x_size = dataset.RasterYSize
            y_size = dataset.RasterXSize
            
            # Get the number of bands in the cropped image
            n_bands = dataset.RasterCount
            n_bands = int(n_bands)
            
            # If the cropped image is multi-spectral (i.e. more than one raster band)
            if n_bands > 1:
                
                # Create an empty image as a NumPy array with "n_bands" bands
                cropped_image = np.empty([n_bands, x_size, y_size])
                
                # Assign each raster band in the cropped image to the NumPy array
                for n in np.arange(n_bands):
                    band = dataset.GetRasterBand(int(n + 1))
                    cropped_image[n, :, :] = band.ReadAsArray()
            
            # If the cropped image is greyscale (i.e. one raster band)
            elif n_bands == 1:
                
                # Get the raster band of the cropped image                
                band = dataset.GetRasterBand(1)
                
                # Read in the cropped image as a NumPy array
                cropped_image = band.ReadAsArray()
                
                # Normalise the cropped image to a 0-255 range
                cropped_image = 255 * ((cropped_image - np.amin(cropped_image)) / (np.amax(cropped_image) - np.amin(cropped_image)))
            
            else:
                raise ValueError("No raster bands are present in image.")
        
        else:
            raise OSError("No cropped image file present.")

        return cropped_image, resolution, min_longitude, max_longitude, min_latitude, max_latitude, geotransform, projection, n_bands, x_size, y_size

    def read_cropped_im(self):
        
        # Define the path to the cropped input image
        input_path = os.path.join(self.input_dir, self.filename)
        
        # Open the cropped image as a GDAL raster dataset
        dataset = gdal.Open(input_path)
        
        # Get the geotransform of the cropped image
        geotransform = dataset.GetGeoTransform()
    
        # Get the resolution of the cropped image
        resolution = geotransform[1]
    
        # Get the map-projection of the cropped image
        projection = dataset.GetProjection()
        
        # Get the horizontal and vertical pixel sizes
        x_size = dataset.RasterYSize
        y_size = dataset.RasterXSize
        
        # Get the number of bands in the cropped image
        n_bands = dataset.RasterCount
        
        # If the cropped image is multi-spectral (i.e. more than one raster band)
        if n_bands > 1:
            
            # Create an empty image as a NumPy array with "n_bands" bands
            cropped_image = np.empty([n_bands, x_size, y_size])
            
            # Assign each raster band in the cropped image to the NumPy array
            for n in np.arange(n_bands):
                band = dataset.GetRasterBand(int(n + 1))
                cropped_image[n, :, :] = band.ReadAsArray()
        
        # If the cropped image is greyscale (i.e. one raster band)
        elif n_bands == 1:
            
            # Get the raster band of the cropped image                
            band = dataset.GetRasterBand(1)
            
            # Read in the cropped image as a NumPy array
            cropped_image = band.ReadAsArray()
        
        else:
            raise ValueError("No raster bands are present in image.")

        return cropped_image, resolution, geotransform, projection, n_bands, x_size, y_size
    
    def read_metadata(self, dataset, min_longitude, max_longitude, min_latitude, max_latitude):
        
        # Get product identification of the image from the filename
        name = os.path.splitext(self.filename)[0]
        
        # Read in MRO HiRISE RDR version 1.1 metadata from index file
        if dataset == 'hirise-rdr':
            
            # Remove any suffix from image name
            if not name.endswith('_RED') and not name.endswith('_COLOR'):
                name = name[:-2]
    
            # Define the path to the metadata file
            metadata_path = os.path.join(self.metadata_dir, 'RDRCUMINDEX.TAB')
            
            # Read in the product names from the metadata file to find the relevant row
            product_names = np.genfromtxt(metadata_path, delimiter=',', usecols=5, unpack=True, autostrip=True, dtype=str, ndmin=1)
            
            # Remove quotes and whitespace from product names
            for p in np.arange(product_names.size):
                product_names[p] = product_names[p].replace('"','')
                product_names[p] = product_names[p].replace(' ','')           
            
            # Get the index for the relevant row for this image
            index = np.where(product_names == name)[0][0]
            
            # If image lat/lon has been calculated from pit location shapefile
            if min_latitude is not None:
                
                # Read in the metadata for this particular MRO HiRISE image
                em_angle, inc_angle, sc_distance, north_azim_angle, solar_azim_angle, sc_latitude, sc_longitude = np.genfromtxt(metadata_path, 
                                                                                                                    delimiter=',',
                                                                                                                    usecols=(19,20,24,25,26,29,30),
                                                                                                                    skip_header=index, 
                                                                                                                    max_rows=1, 
                                                                                                                    dtype=float, 
                                                                                                                    ndmin=1)
            
                # Convert min/max image lon coords to a 0/360 domain if in -180/180
                if max_longitude < 0:
                    max_longitude += 360
                if min_longitude < 0:
                    min_longitude += 360
            
            # If image lat/lon has not been calculated from pit location shapefile
            elif min_latitude is None:
            
                # Read in the metadata for this particular MRO HiRISE image
                em_angle, inc_angle, sc_distance, north_azim_angle, solar_azim_angle, sc_latitude, sc_longitude, min_latitude, max_latitude, min_longitude, max_longitude = np.genfromtxt(metadata_path,
                            delimiter=',', 
                            usecols=(19,20,24,25,26,29,30,35,36,37,38),
                            skip_header=index, 
                            max_rows=1, 
                            dtype=float,
                            ndmin=1)
            
            # Calculate the solar azimuth angle from due-North [in radians]
            if north_azim_angle > solar_azim_angle:
                solar_azim_angle = 360 - (north_azim_angle - solar_azim_angle)
            elif north_azim_angle < solar_azim_angle:
                solar_azim_angle = solar_azim_angle - north_azim_angle
            
            # Calculate the centre latitude and longitude of the image
            image_latitude, image_longitude = (max_latitude + min_latitude) / 2, (max_longitude + min_longitude) / 2
         
            # Convert the spacecraft distance to m from km
            sc_distance = sc_distance * 1e3
            
            # Define the equatorial and polar radii of Mars
            equatorial_radius, polar_radius = 3396.19e3, 3376.20e3
        
        # Read in LRO NAC EDR metadata from index file
        elif dataset == 'lronac-edr':
            
            # Define the path to the metadata file
            metadata_path = os.path.join(self.metadata_dir, 'CUMINDEX.TAB')
            
            # Read in the product names from the metadata file to find the relevant row
            product_names = np.genfromtxt(metadata_path, delimiter=',', usecols=5, unpack=True, autostrip=True, dtype=str)
            
            # Remove quotes and whitespace from product names
            for p in np.arange(product_names.size):
                product_names[p] = product_names[p].replace('"','')
                product_names[p] = product_names[p].replace(' ','')           
            
            # Get the index for the relevant row for this image
            index = np.where(product_names == name)[0][0]
            
            # If image lat/lon has been calculated from pit location shapefile
            if min_latitude is not None:
            
                # Read in the metadata for this particular LROC NAC image (lat/lon coords in 0-360 domain)
                em_angle, inc_angle, north_azim_angle, solar_azim_angle, sc_latitude, sc_longitude, centre_distance = np.genfromtxt(metadata_path, 
                                                                                                                    delimiter=',',
                                                                                                                    usecols=(58,59,61,62,65,66,80),
                                                                                                                    skip_header=index, 
                                                                                                                    max_rows=1, 
                                                                                                                    dtype=float,
                                                                                                                    ndmin=1)
                
                # Convert min/max image lon coords to a 0/360 domain if in -180/180
                if max_longitude < 0:
                    max_longitude += 180
                if min_longitude < 0:
                    min_longitude += 180
                    
            # If image lat/lon has not been calculated from pit location shapefile
            elif min_latitude is None:
            
                # Read in the metadata for this particular LROC NAC image (lat/lon coords in 0-360 domain)
                em_angle, inc_angle, north_azim_angle, solar_azim_angle, sc_latitude, sc_longitude, u_r_lat, u_r_lon, l_r_lat, l_r_lon, l_l_lat, l_l_lon, u_l_lat, u_l_lon, centre_distance = np.genfromtxt(metadata_path, 
                            delimiter=',',
                            usecols=(58,59,61,62,65,66,71,72,73,74,75,76,77,78,80),
                            skip_header=index, 
                            max_rows=1, 
                            dtype=float,
                            ndmin=1)
            
                # Calculate the min/max lat/lon from the coords of the 4 corners
                max_latitude = max(u_r_lat, u_l_lat)
                min_latitude = min(u_r_lat, u_l_lat)
                max_longitude = max(u_r_lon, u_l_lon)
                min_longitude = min(u_r_lon, u_l_lon)
            
            # Calculate the solar azimuth angle from due-North [in deg]
            if north_azim_angle < 180:
                north_azim_angle = 360 - north_azim_angle
                solar_azim_angle = solar_azim_angle - north_azim_angle 
            
            # Calculate the centre latitude and longitude of the image
            image_latitude, image_longitude = (max_latitude + min_latitude) / 2, (max_longitude + min_longitude) / 2
            
            # Convert the spacecraft distance to m from km
            centre_distance = centre_distance * 1e3
            
            # Define the radius of the Moon
            equatorial_radius = polar_radius = 1737.4e3
            
            # Calculate the great circle angle between the satellite and image
            great_angle = np.arccos((np.sin(image_latitude * (np.pi/180)) * np.sin(sc_latitude * (np.pi/180))) + (np.cos(image_latitude * (np.pi/180)) * np.cos(sc_latitude * (np.pi/180)) * np.cos(abs(image_longitude - sc_longitude) * (np.pi/180))))
            
            # Calculate the great angle distance and r_1
            great_distance = 2 * equatorial_radius * np.sin((great_angle / 2))
            r_1 = equatorial_radius * np.sin(great_angle)
            r_2 = np.sqrt(great_distance**2 - r_1**2)
            
            # Calculate the correct satellite slant distance
            sc_distance = np.sqrt(((centre_distance - equatorial_radius) + r_2)**2 + r_1**2)
            
        else:
            raise ValueError("--dataset must equal 'hirise-rdr' or 'lronac-edr' for MRO HiRISE RDRV11 and LROC NAC EDR images, respectively.")
        
        # Convert all sensing angles to radians
        inc_angle = inc_angle * (np.pi / 180)
        em_angle = em_angle * (np.pi / 180)
        north_azim_angle = north_azim_angle * (np.pi / 180)
        solar_azim_angle = solar_azim_angle * (np.pi / 180)

        # Find the difference in the satellite and image lat/long [in deg]
        delta_latitude = sc_latitude - image_latitude
        delta_longitude = sc_longitude - image_longitude
        
        # Correct deltas to account for 0-360 longitude domain
        if delta_longitude > 180:
            delta_longitude -= 360
        elif delta_longitude < -180:
            delta_longitude += 360
            
        # Calculate the change in lat/lon between the sub-satellite point and the centre of the image [in m]
        delta_latitude_m = delta_latitude * (np.pi / 180) * polar_radius
        delta_longitude_m = delta_longitude * (np.pi / 180) * equatorial_radius
        
        # Calculate the angle between image and satellite [in radians]
        sc_azim_angle = np.arctan(abs(delta_longitude_m / delta_latitude_m))
        
        # Calculate the spacecraft's azimuth angle clockwise from due-north
        if delta_longitude < 0 and delta_latitude < 0:
            sc_azim_angle += np.pi
        elif delta_longitude < 0 and delta_latitude > 0:
            sc_azim_angle = (2*np.pi) - sc_azim_angle
        elif delta_longitude > 0 and delta_latitude < 0:
            sc_azim_angle = np.pi - sc_azim_angle
        
        # Find the ground phase angle between the Sun and satellite lines of sight [in radians]
        phase_angle = max(solar_azim_angle, sc_azim_angle) - min(solar_azim_angle, sc_azim_angle)
        
        # Ensure phase angle is between 0-2pi
        if phase_angle > np.pi:
            phase_angle = (2 * np.pi) - phase_angle
        
        # Calculate the ground distance from the satellite to the image centre [in m]
        d_g = sc_distance * np.sin(em_angle)
        
        # Find the maximum possible difference in ground distance in terms of the images max/min lat/lon coords [in m]    
        delta_d_g = max(((max_latitude - min_latitude) / 2) * (np.pi / 180) * polar_radius, ((max_longitude - min_longitude) / 2) * (np.pi / 180) * equatorial_radius)
        
        # Calculate the maximum and minimum ground distances [in m]
        max_d_g = d_g + delta_d_g
        min_d_g = d_g - delta_d_g
        
        # Calculate the altitude of the satellite above the horizon [in m]
        d_h = sc_distance * np.cos(em_angle)
        
        # Calculate the range of possible emission angles over the entire image [in radians]
        max_em_angle = np.arctan(max_d_g / d_h)
        min_em_angle = np.arctan(min_d_g / d_h)
        
        # Calculate the uncertainty in the emission angle [in degrees]
        delta_em_angle = abs((max_em_angle - min_em_angle) / 2)
        
        # Calculate the ground distances of the satellite that are perpendicular (u) and parallel (v) to the Sun direction [in m]
        u = d_g * np.sin(phase_angle)
        v = d_g * abs(np.cos(phase_angle))
    
        # Calculate the obliquity angles of the satellite parallel and perpendicular to the Sun's direction [in radians]
        em_angle_par = np.arctan(v / d_h)
        em_angle_perp = np.arctan(u / d_h)
        
        return inc_angle, solar_azim_angle, sc_azim_angle, phase_angle, em_angle, delta_em_angle, em_angle_par, em_angle_perp

    def read_ground_truth(self, n_bands, cropped_image, geotransform, projection):

        # Get product identification of the image from the filename
        name = os.path.splitext(self.filename)[0]

        # Find the path to the manually-labelled shadow shapefile
        validation_path = os.path.join(self.testing_dir, name + '.shp')

        # Open the shapefile layer
        validation_dataset = ogr.Open(validation_path)
        validation_layer = validation_dataset.GetLayer()

        # Define the output path to save the rasterised shadow shapefile to
        output_path = os.path.join(self.output_dir,'validation_{}.tiff'.format(name))

        # Create the raster data source to store the rasterised shadow shapefile
        if n_bands == 1:
            validation_raster_dataset = driver1.Create(output_path, cropped_image.shape[1], cropped_image.shape[0], 1, gdal.GDT_Int16)
        elif n_bands > 1:
            validation_raster_dataset = driver1.Create(output_path, cropped_image.shape[2], cropped_image.shape[1], 1, gdal.GDT_Int16)
        else:
            raise ValueError("Number of bands should not be zero.")
        
        # Set the geotransform and map-projection
        validation_raster_dataset.SetGeoTransform(geotransform)
        validation_raster_dataset.SetProjection(projection)
        
        # Get the raster band and assign the no data value
        band = validation_raster_dataset.GetRasterBand(1)
        band.SetNoDataValue(np.nan)
        band.FlushCache()

        # Rasterise the validation polygons so that you get a raster where 1=shadow and 0=background
        gdal.RasterizeLayer(validation_raster_dataset, [1], validation_layer, None, None, options=['ATTRIBUTE=class'], callback=None)
        
        # Close the raster dataset and band
        validation_raster_dataset = band = None

        # Read the rasterised validation data as an array
        validation_array = gdal.Open(output_path).ReadAsArray()
        
        # Get the true shadow and bright feature masks
        true_shadow = np.where(validation_array == 1, 1, 0)
        true_bright = np.where(validation_array == 2, 1, 0)
        
        # Remove the temporary raster file
        os.remove(output_path)

        return true_shadow, true_bright

    def save_shadow(self, main_shadow, filled_shadow, geotransform, projection):

        # Get product idenitification of the image from the filename
        name = os.path.splitext(self.filename)[0]

        # Define the directory where the shadows should be saved
        shadows_dir = os.path.join(self.output_dir, 'shadows/')

        # Define the output path to save the rasterised detected shadow to
        output_path = os.path.join(shadows_dir, name + '_main_shadow.tif')
        
        # Rasterise the shadow mask and save as a geo-referenced GeoTiff
        shadow_dataset = driver1.Create(output_path, main_shadow.shape[1], main_shadow.shape[0], 1, gdal.GDT_Int16)
        
        # Set the geotransform and projection
        shadow_dataset.SetGeoTransform(geotransform)
        shadow_dataset.SetProjection(projection)
        
        # Write the NumPy array containing the shadow mask to the band of the raster dataset
        shadow_band = shadow_dataset.GetRasterBand(1)
        shadow_band.WriteArray(main_shadow)
        shadow_band.SetNoDataValue(np.nan)
        shadow_band.FlushCache()
        
        # Define the output path to save the polygonised detected shadow to
        output_path = os.path.join(shadows_dir, name + '_shadow.shp')

        # Create the shapefile layer to store the detected shadow
        shapefile = driver2.CreateDataSource(output_path)
        shadow_layer = shapefile.CreateLayer('shadow', srs=None)

        # Create attribute to store the image's product name
        product = ogr.FieldDefn("prod_code", ogr.OFTString)
        shadow_layer.CreateField(product)

        # Polygonise the raster dataset into the shapefile layer
        gdal.Polygonize(shadow_band, shadow_band, shadow_layer, -1, [], callback=None)

        # Set up a multipolygon wkb to merge the individual features
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        # Loop through each feature to add it to the multipolygon
        for feature in shadow_layer:
            
            # Get the feature's geometry and add it to the multipolygon
            geometry = feature.GetGeometryRef()
            multipolygon.AddGeometry(geometry)

            # Delete the feature
            shadow_layer.DeleteFeature(feature.GetFID())

        # Get the merged feature from the layer
        features = ogr.Feature(shadow_layer.GetLayerDefn())

        # Fill in the product code field
        features.SetField("prod_code", name)

        # Set the feature geometry
        features.SetGeometry(multipolygon)
            
        # Create the feature in the shapefile layer
        shadow_layer.CreateFeature(features)

        # Close the datasets, bands and layers
        shadow_dataset = shadow_band = None
        shadow_layer = features = None
        
        # If bright features were detected
        if filled_shadow is not None:
            
            # Define the output path to save the filled shadow to
            filled_output_path = os.path.join(shadows_dir, name + '_filled_shadow.tif')
        
            # Rasterise the shadow mask and save as a geo-referenced GeoTiff
            filled_shadow_dataset = driver1.Create(filled_output_path, filled_shadow.shape[1], filled_shadow.shape[0], 1, gdal.GDT_Int16)
            
            # Set the geotransform and projection
            filled_shadow_dataset.SetGeoTransform(geotransform)
            filled_shadow_dataset.SetProjection(projection)
            
            # Write the NumPy array containing the shadow mask to the band of the raster dataset
            filled_shadow_band = filled_shadow_dataset.GetRasterBand(1)
            filled_shadow_band.WriteArray(filled_shadow)
            filled_shadow_band.SetNoDataValue(np.nan)
            filled_shadow_band.FlushCache()

            # Close the datasets
            filled_shadow_dataset = filled_shadow_band = None
            
    def read_shadow(self, shadowtype):

        # Get product idenitification of the image from the filename
        name = os.path.splitext(self.filename)[0]
        
        # Define the path to the shadow mask
        shadows_dir = os.path.join(self.output_dir, 'shadows/')
        input_path = os.path.join(shadows_dir, name + f'_{shadowtype}.tif')
        
        # Open the shadow mask as a GDAL raster dataset
        dataset = gdal.Open(input_path)
        
        # Get the number of bands in the shadow mask
        n_bands = dataset.RasterCount
        
        if n_bands > 1:
            raise ValueError("Shadow mask should be single-band")
        
        elif n_bands == 1:
            
            # Get the raster band of the shadow mask
            band = dataset.GetRasterBand(1)
            
            # Read in the shadow mask as a NumPy array
            shadow_array = band.ReadAsArray()
        
        else:
            raise ValueError("No raster bands are present in shadow mask.")

        return shadow_array

    def save_h_profile(self, L_obs, h_obs, pos_h_obs, neg_h_obs, L_true, h_true, pos_h_true, neg_h_true):

        # Get product idenitification of the image from the filename
        name = os.path.splitext(self.filename)[0]
        
        # Define the directory where the h profiles should be saved
        profiles_dir = os.path.join(self.output_dir, 'profiles/')
        
        # Save the depth profile as a csv
        dt = np.dtype([('L_obs', float), ('h_obs', float), ('pos_h_obs', float), ('neg_h_obs', float),
                       ('L_true', float), ('h_true', float), ('pos_h_true', float), ('neg_h_true', float)])
        array = np.empty(L_obs.size, dtype=dt)
        array['L_obs'] = L_obs
        array['h_obs'] = h_obs
        array['pos_h_obs'] = pos_h_obs
        array['neg_h_obs'] = neg_h_obs
        array['L_true'] = L_true
        array['h_true'] = h_true
        array['pos_h_true'] = pos_h_true
        array['neg_h_true'] = neg_h_true
        output_path = os.path.join(profiles_dir, name + '_profile.csv')
        np.savetxt(output_path, array, delimiter=',', fmt='%f, %f, %f, %f, %f, %f, %f, %f', header='Uncorrected Shadow Length [m], Uncorrected h [m], +, -, Shadow Length [m], h [m], +, -')

class ShadowExtractor(object):

    '''
    ShadowExtractor performs k-means clustering and silhouette analysis to automatically extract the 
    shadow from a cropped image of a Martian or Lunar pit. The result of this is a binary shadow mask 
    with the same dimensions as the input image in which 1 = shadow and 0 = non-shadow.
    '''

    def __init__(self,
                cropped_image,
                n_clusters,
                factor):
        
        self.cropped_image = cropped_image
        self.n_clusters = n_clusters
        self.factor = factor

    def kmeans_clustering(self):

        # Reshape the cropped image into a vector
        vector_image = self.cropped_image.reshape(-1, 1)

        # Apply the k-means algorithm to separate the cropped image into n clusters (where n = n_clusters)
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10, tol=0, algorithm='lloyd', random_state=0).fit(vector_image)
        
        # Resize the cluster labels such that each pixel in the input cropped image is now assigned to a cluster
        labels = kmeans.predict(vector_image).reshape(self.cropped_image.shape)

        return labels
    
    def sort_clusters(self, labels):
        
        # Find the unique cluster labels that have been assigned to the pixels in the input image
        unique_labels = np.unique(labels)
                
        # Store the average pixel intensity for each cluster as it appears in the input image
        av_intensity = []
        
        # Loop through all clusters found by k-means algorithm        
        for n in unique_labels:

            # Retrieve the pixel intensities from the input image for this particular cluster
            masked_image = self.cropped_image * np.where(labels == n, 1, -1)
            pixel_list = masked_image.flatten()
            pixel_list = pixel_list[pixel_list > 0]
            
            # Store the average pixel intensity of this cluster
            av_intensity.append(np.mean(pixel_list))

        # Sort the average pixel intensities of all clusters from darkest to brightest
        sorted_av_intensity = sorted(av_intensity)

        # Reassign label values in order of average pixel intensity
        sorted_labels = np.copy(labels)
        for v, value in enumerate(sorted_av_intensity):
            index = av_intensity.index(value)
            sorted_labels = np.where(labels == index, v, sorted_labels)
            
        return sorted_labels

    def calc_silh_coefficient(self, sorted_labels):

        # Get the image size
        image_size = self.cropped_image.size
        
        # Calculate the factor to downscale the image by
        if image_size > 1.5e6:
            downscale_factor = self.factor * 2 
        else:
            downscale_factor = self.factor
        
        # Downscale the input image and labels (by "factor") to optimise silhouette coefficient calculation
        if downscale_factor > 1:
            x = rescale(np.copy(self.cropped_image), 1 / downscale_factor, anti_aliasing=False)
            X = x.reshape(-1, 1)
            y = rescale(np.copy(sorted_labels), 1 / downscale_factor, anti_aliasing=False)
            Y = y.flatten()

        # Calculate the silhouette coefficient for the darkest cluster
        silh_coefficient = np.mean(silhouette_samples(X, Y)[Y == 0])
        
        return silh_coefficient    

class PostProcessor(object):
    
    '''
    PostProcessor contains the functions for applying the necessary post-processing to the extracted 
    shadow before it can be used for apparent depth calculation. 
    This includes:
    -   Extracting only the main shadow in the image, thus removing all small shadow detections due to
        unrelated features such as bolders in the image
    -   Detecting bright features contained within the shadow mask which would affect the shadow width 
        measurement - and consequently the apparent depth measured
    '''
    
    def __init__(self,
                shadow):
        
        self.shadow = shadow

    def post_processing(self):

        # Label the shadow mask into regions and retrieve properties
        labelled_mask = label(self.shadow, background=0)
        regions = regionprops(labelled_mask)

        # Find the connected region with the largest area
        region_areas = []
        for region in regions:
            region_areas.append(region.area)
        main_shadow = np.where(labelled_mask == (region_areas.index(max(region_areas))+1), 1, 0)

        # Close small holes in the main shadow
        main_shadow = area_closing(main_shadow, area_threshold=10, connectivity=1)

        # Fill in large holes in the shadow mask
        filled_shadow = binary_fill_holes(main_shadow).astype(int)
        
        # If no bright features were found, return None for the filled shadow
        if np.sum(filled_shadow == main_shadow) == main_shadow.size:
                        
            return main_shadow, None
        
        # Return both the filled and non-filled main shadows if bright features were found
        else:
            
            return main_shadow, filled_shadow

class ShadowTester(object):
    
    '''
    ShadowTester calculates the recall (R), precision (P) and F1 score (F1) of shadow and bright
    feature pixel detections for each image provided to PITS, assuming that validation shapefiles
    have been produced and are present in the testing directory (testing_dir).
    '''
    
    def __init__(self,
                main_shadow,
                true_shadow):
        
        self.main_shadow = main_shadow
        self.true_shadow = true_shadow
    
    def calc_shadow_metrics(self):
        
        # Find all non-shadow detections
        non_shadow = np.where(self.true_shadow == 1, 0, 1)
        
        # Calculate the true/false positives and false negatives for the main shadow (without holes filled)
        TP = np.sum(self.main_shadow * self.true_shadow)
        FP = np.sum(self.main_shadow * non_shadow)
        FN = np.sum(self.true_shadow) - np.sum(self.main_shadow * self.true_shadow)
        
        # Calculate the precision, recall and F1 scores
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = (2 * P * R)/ (P + R)
        
        return P, R, F1

class DepthCalculator(object):
    
    '''
    DepthCalculator takes the processed shadow detected by the PITS ShadowExtractor, as well as the 
    sensing information at the time of image acquisition, and calculates the apparent depth profile.
    It does this by:
    -   Rotating the extracted shadow according to the sub-solar azimuth angle such that the Sun's
        line of sight now passes from bottom to top
    -   Measuring the shadow width by finding the distance between the rim- and pit-interior-edge of 
        the shadow in pixels and then multiplying by the image resolution in metres
    -   Correcting the shadow widths measured along the shadow's length for any non-nadir observations 
        of the target feature
    -   Calculating the apparent depth profile using the shadow width and the solar incidence angle
    '''
    
    def __init__(self,
                shadow_list,
                resolution,
                inc_angle,
                em_angle,
                em_angle_par,
                em_angle_perp,
                solar_azim_angle,
                phase_angle):
        
        self.shadow_list = shadow_list
        self.resolution = resolution
        self.inc_angle = inc_angle
        self.em_angle = em_angle
        self.em_angle_par = em_angle_par
        self.em_angle_perp = em_angle_perp
        self.solar_azim_angle = solar_azim_angle
        self.phase_angle = phase_angle

    def align_shadow(self):
        
        # Open empty list to store all aligned shadows
        aligned_shadows = []

        # Loop through main and filled shadows if given
        for shadow in self.shadow_list:

            # Ensure that binary shadow mask is made of integer values
            shadow = shadow.astype(np.uint8)
            
            # Convert the solar azimuth angle back to degrees from radians
            azim_angle = self.solar_azim_angle * (180 / np.pi)

            # Rotate the binary shadow mask so that the Sun's line of sight passed from bottom to top
            aligned_shadow = rotate(shadow, azim_angle - 180, resize=True, order=0, mode='constant', cval=0, preserve_range=True)

            # Store the shadow
            aligned_shadows.append(aligned_shadow)

        if len(aligned_shadows) > 1:

            return aligned_shadows

        elif len(aligned_shadows) == 1:

            return aligned_shadows[0]
        
        else:
            raise ValueError("No aligned shadows were retrieved.")

    def remove_bright_features(self, aligned_main_shadow, aligned_filled_shadow):
        
        # Find the difference between the filled and non-filled shadow masks to get the bright features
        difference = aligned_filled_shadow - aligned_main_shadow
        
        # Get the coordinates of these features
        coords = np.where(difference == 1)
        
        # Find the unique x coordinates of all bright features
        unique_xs = np.unique(coords[1])
        
        for unique_x in unique_xs:
            
            # Get the edge and rim of the aligned shadow
            edge = np.amin(np.where(aligned_filled_shadow[:, unique_x] == 1)[0])
            rim = np.amax(np.where(aligned_filled_shadow[:, unique_x] == 1)[0])
            
            # Get the rim-side or shadow-edge side of the bright feature
            ymax = np.amax(np.where(difference[:, unique_x] == 1)[0])
            ymin = np.amin(np.where(difference[:, unique_x] == 1)[0])
            
            # Get the centre y coordinate of the shadow and bright feature
            shadow_centre = ((rim - edge) / 2) + edge
            bright_centre = ((ymax - ymin) / 2) + ymin
            
            # Remove the shadow pixels above or below depending on if it is nearer the edge or rim
            if shadow_centre > bright_centre:
                aligned_main_shadow[:ymin, unique_x] = 0
            elif shadow_centre < bright_centre:
                aligned_main_shadow[ymax:, unique_x] = 0
            else:
                aligned_main_shadow[ymax:, unique_x] = 0
        
        # Remove any small shadows which have now been separated from the main shadow
        labelled_mask = label(aligned_main_shadow, background=0)
        regions = regionprops(labelled_mask)
        region_areas = []
        for region in regions:
            region_areas.append(region.area)
        aligned_main_shadow = np.where(labelled_mask == (region_areas.index(max(region_areas))+1), 1, 0)
        
        return aligned_main_shadow

    def measure_shadow(self, aligned_shadow):
        
        # Find the pixel coordinates of the shadow edge and rim
        coords = np.where(aligned_shadow == 1)
        xmin, xmax, ymin, ymax = np.amin(coords[1]), np.amax(coords[1]), np.amin(coords[0]), np.amax(coords[0])
        length_coords = np.arange(xmin, xmax+1, 1)
        edge = np.empty(length_coords.size)
        rim = np.empty(length_coords.size)
        
        # Measure the observed shadow width at every point [in m]
        S_obs = np.zeros(aligned_shadow.shape[1])
        errors = []
        for e, l in enumerate(length_coords):
            try:
                edge[e] = np.amin(np.where(aligned_shadow[:, l] == 1)[0])
                rim[e] = np.amax(np.where(aligned_shadow[:, l] == 1)[0])
                S_obs[l] = self.resolution * np.abs(edge[e] - rim[e])
            except:
                errors.append(l)
                S_obs[l] = 0
        
        if len(errors) > 0:
            edge = np.delete(edge, np.array(errors))
            rim = np.delete(rim, np.array(errors))
            length_coords = np.delete(length_coords, np.array(errors))
        
        return S_obs, length_coords, edge, rim
    
    def propagate_uncertainties(self, S_obs, pos_delta_S_obs, neg_delta_S_obs, delta_em_angle):
        
        # Propagate the uncertainties to get the upper and lower bounds of the observed apparent depth [in m]
        pos_delta_h_obs = pos_delta_S_obs * np.tan((np.pi / 2) - self.inc_angle)
        neg_delta_h_obs = neg_delta_S_obs * np.tan((np.pi / 2) - self.inc_angle)
        
        # Calculate the uncertainty in the parallel and perpendicular obliquities to the Sun's line of sight [in radians]
        delta_em_angle_par = delta_em_angle * ((abs(np.cos(self.phase_angle)) * ((1 / np.cos(self.em_angle))**2)) / (((np.cos(self.phase_angle)**2) * (np.tan(self.em_angle)**2)) + 1))
        delta_em_angle_perp = delta_em_angle * ((np.sin(self.phase_angle) * ((1 / np.cos(self.em_angle))**2)) / (((np.sin(self.phase_angle)**2) * (np.tan(self.em_angle)**2)) + 1))
        
        # If the Sun and satellite are looking exactly or roughly in the same directions
        if self.phase_angle >= 0 and self.phase_angle < np.pi / 2:
            
            # Calculate the uncertainty in the true shadow width w.r.t the observed shadow width
            d_S_obs = (1 / np.cos(self.em_angle_par)) + ((np.tan(self.em_angle_par)) / ((np.cos(self.em_angle_par)**2) * (np.tan(self.inc_angle) - np.tan(self.em_angle_par))))
            
            # Calculate the uncertainty in the true shadow width w.r.t the parallel obliquity angle
            x = ((1 / np.cos(self.em_angle_par))**2) / (np.cos(self.em_angle_par) * ((np.tan(self.inc_angle) - np.tan(self.em_angle_par))**2))
            y = (np.sin(self.em_angle_par) * np.tan(self.em_angle_par)) / (np.tan(self.inc_angle) - np.tan(self.em_angle_par))
            z = (((1 / np.cos(self.em_angle_par)**2)) * np.tan(self.em_angle_par)) / ((np.tan(self.inc_angle) - np.tan(self.em_angle_par))**2)
            d_em_angle_par = (S_obs / np.cos(self.em_angle_par)) * (np.tan(self.em_angle_par) + x + y + z)
            
        # If the Sun and satellite are looking exactly or roughly in opposite directions
        elif self.phase_angle > np.pi / 2 and self.phase_angle <= np.pi:
            
            # Calculate the uncertainty in the true shadow width w.r.t the observed shadow width
            d_S_obs = np.sin(self.inc_angle) / np.cos((np.pi / 2) - self.inc_angle - self.em_angle_par)
            
            # Calculate the uncertainty in the true shadow width w.r.t the parallel obliquity angle
            d_em_angle_par = S_obs * np.sin(self.inc_angle) * (np.sin((np.pi / 2) - self.inc_angle - self.em_angle_par) / (np.cos((np.pi / 2) - self.inc_angle - self.em_angle_par)**2))
        
        # If the Sun and satellite are looking exactly perpendicular to each other
        elif self.phase_angle == np.pi / 2:
            
            # Define the uncertainty in the true shadow width w.r.t the observed shadow width
            d_S_obs = 1
            
            # Calculate the uncertainty in the true shadow width w.r.t the parallel obliquity angle
            d_em_angle_par = 0
            
        else:
            raise ValueError("Ground phase angle should be between 0 and 180 deg.")

        # Calculate the upper and lower bounds of the uncertainty in the true shadow width [in m]
        pos_delta_S_true = np.sqrt(((pos_delta_S_obs**2) * (d_S_obs**2)) + ((delta_em_angle_par**2) * (d_em_angle_par**2)))
        neg_delta_S_true = np.sqrt(((neg_delta_S_obs**2) * (d_S_obs**2)) + ((delta_em_angle_par**2) * (d_em_angle_par**2)))
        
        # Propagate the uncertainties to get the upper and lower bounds of the true apparent depth [in m]
        pos_delta_h_true = pos_delta_S_true * np.tan((np.pi / 2) - self.inc_angle)
        neg_delta_h_true = neg_delta_S_true * np.tan((np.pi / 2) - self.inc_angle)
        
        return pos_delta_h_obs, neg_delta_h_obs, pos_delta_h_true, neg_delta_h_true

    def correct_shadow_width(self, S_obs, L_obs):
        
        # If the Sun and satellite are looking roughly in the same direction
        if self.phase_angle >= 0 and self.phase_angle < np.pi / 2:
            
            # Calculate the true visible shadow width
            S_visible = S_obs / np.cos(self.em_angle_par)
            
            # Calculate the hidden element of the shadow width
            S_hidden = (S_visible * np.tan(self.em_angle_par)) / (np.tan(self.inc_angle) - np.tan(self.em_angle_par))
            
            # Calculate the full true shadow width
            S_true = S_visible + S_hidden
        
        # If the Sun and satellite are looking roughly in opposite directions
        elif self.phase_angle > np.pi / 2 and self.phase_angle <= np.pi:
            
            # Use theta to calculate distance from rim to shadow edge (d)
            theta = (np.pi / 2) - self.inc_angle - self.em_angle_par
            d = S_obs / np.cos(theta)
            
            # Calculate the full true shadow width [in metres]
            S_true = d * np.sin(self.inc_angle)
        
        # If the Sun and satellite are looking exactly perpendicular to each other
        elif self.phase_angle == np.pi / 2:
            
            # The true shadow width will be what was observed since the parallel obliquity is zero
            S_true = S_obs
            
        else:
            raise ValueError("Ground phase angle should be between 0 and 180 deg.")

        # Correct the measured shadow length
        L_true = L_obs / np.cos(self.em_angle_perp)

        return S_true, L_true
    
    def calculate_h(self, S):
            
        # Find the angle between the Sun and the horizon
        horizon_angle = (np.pi / 2) - self.inc_angle

        # Calculate the apparent depth of the pit [in metres]
        h = S * np.tan(horizon_angle)

        return h