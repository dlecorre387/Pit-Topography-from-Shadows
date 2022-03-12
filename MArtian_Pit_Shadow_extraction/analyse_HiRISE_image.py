import os
import time
import numpy as np
import math as m
from osgeo import gdal, ogr
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.transform import rotate
from skimage.measure import label, regionprops

class ImageAnalyser(object):
    
    def __init__(self,
                filename,
                input_dir,
                metadata_dir,
                labels_dir,
                output_dir,
                n_clusters):

        self.filename = filename
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        
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

    def crop_HiRISE_image(self):
        
        # Get product name of file
        name = os.path.splitext(self.filename)[0]

        # Define filenames for the labels, input image, and cropped output
        input_path = os.path.join(self.input_dir, self.filename)
        shp_path = os.path.join(self.labels_dir, 'converted_product_id_' + name + '.shp')
        output_path = os.path.join(self.output_dir, 'cropped_' + name + '.tif')

        # Clip the input image using the shapefile label
        crop = gdal.Warp(output_path, 
                    input_path, 
                    cutlineDSName=shp_path,
                    cropToCutline=True,
                    dstNodata = 0)
        
        # Close the raster dataset to save it
        crop = None

        # Read croped raster file as a NumPy array
        counter = 0
        if ('cropped_' + name + '.tif') in os.listdir(self.output_dir):
            ds = gdal.Open(output_path) # GDAL Dataset
            geot = ds.GetGeoTransform() # GeoTransform
            proj = ds.GetProjection() # Projection
            if ds.RasterCount > 1:
                raise ValueError("More than one raster band present")
            red = ds.GetRasterBand(1)
            cropped_im = red.ReadAsArray()
            
            # # Normalise the cropped image between 0 and 255 (UInt8)
            # cropped_im = np.around(255*((array - np.amin(array))/(np.amax(array) - np.min(array)))).astype(int)

        elif ('cropped_' + name + '.tif') not in os.listdir(self.output_dir) and counter < 3:
            time.sleep(60)
            counter += 1

        else:
            raise ValueError("No cropped image file present")

        return cropped_im, geot, proj

    def read_cropped_im(self):

        # Open the cropped raster file as a NumPy array
        ds = gdal.Open(os.path.join(self.input_dir, self.filename)) # GDAL Dataset
        geot = ds.GetGeoTransform() # GeoTransform
        proj = ds.GetProjection() # Projection
        if ds.RasterCount > 1:
            raise ValueError("More than one raster band present")
        red = ds.GetRasterBand(1)
        cropped_im = red.ReadAsArray()

        return cropped_im, geot, proj

    def cluster_image(self, cropped_im, azim_angle, resolution):

        # Create empty lists to store shadow masks and width estimates
        shadows = []
        widths = []

        # Create progress bar
        pbar = tqdm(total=len(self.n_clusters))

        # Calculate the shadow width for a range of K-Means clusters
        for clusters in self.n_clusters:

            # Perform K-means image segmentation into 'n' number of clusters
            vec_img = cropped_im.reshape(-1,1)
            kmeans = KMeans(n_clusters=clusters).fit(vec_img)
            labels = kmeans.predict(vec_img).reshape(cropped_im.shape)

            # Loop through each cluster to find its average pixel value
            av_val = []
            for n in range(0, np.amax(labels) + 1):

                # Multiply the crop by the cluster mask to get just the pixels underneath
                masked_im = cropped_im*np.where(labels == n, 1, 0)

                # Flatten the pixel values into a list and remove zeroes
                pixel_list = masked_im.flatten()
                pixel_list = [x for x in pixel_list if x != 0]
                
                # Append the average pixel value of this cluster to the list av_val
                av_val.append(np.mean(pixel_list))

            # Sort the clusters' average pixel values from lowest to highest
            sorted_av_val = sorted(av_val)

            # Reassign label numbers in order of average pixel value (i.e. brightness)
            sorted_labels = np.copy(labels)
            for value in sorted_av_val:
                index = np.where(av_val==value)
                sorted_labels = np.where(labels == index, sorted_av_val.index(value), sorted_labels)  

            # Return None if no cluster has an average pixel value low enough to be a shadow
            if len(np.where(np.array(sorted_av_val) > 30)) > 1:
                sorted_labels = None
            
            # Check to see if there are other shadow clusters and merge them
            else:
                for val in sorted_av_val[1:]:
                    if val < 30:
                        index = sorted_av_val.index(val)
                        sorted_labels = np.where(labels == index, 0, sorted_labels)

            # If a shadow has been extracted
            if sorted_labels is not None:

                # Convert the cluster mask to a binary image and fill any holes
                mask = np.where(sorted_labels == 0, 1, 0).astype(np.uint8)

                # Label the mask into regions and retrieve properties
                labelled_mask = label(mask, background=0)
                regions = regionprops(labelled_mask)

                # Extract the largest continuous feature in the shadow mask
                region_areas = []
                for region in regions:
                    region_areas.append(region.area)
                shadow_mask = np.where(labelled_mask == (region_areas.index(max(region_areas))+1), 1, 0)
                shadows.append(shadow_mask)

                # Align shadow mask using the sub-solar azimuth angle
                aligned_mask = rotate(shadow_mask, azim_angle-90, resize=True, order=1, mode='constant', cval=0, preserve_range=True)

                # Find the x coordinates of the shadow mask
                xs = np.where(aligned_mask != 0)[1]
                x_coords = np.arange(min(xs), max(xs) + 1)
                
                # Measure shadow width along the centre of the mask
                x_m = int(np.around(((max(x_coords) - min(x_coords))/2) + min(x_coords)))
                S = resolution*np.sum(aligned_mask[:, x_m])
                widths.append(S)

            # Update and close progress bar
            pbar.update(1)
        pbar.close()

        return widths, shadows, sorted_labels

    def estimate_pit_depth(self, av_S, sigma_S, inc_angle):
        
        # Find angle between solar line of sight and surface
        solar_angle = 90 - inc_angle

        # Convert the solar angle from degrees to rads
        solar_angle = solar_angle*(np.pi/180)

        # Calculate the apparent depth of the pit in metres
        h = av_S*m.tan(solar_angle)

        # Propagate uncertainty
        sigma_h = sigma_S*m.tan(solar_angle)

        return h, sigma_h

    def save_outputs(self, geot, proj, shadows):

        driver1 = gdal.GetDriverByName("GTiff")
        driver2 = ogr.GetDriverByName("ESRI Shapefile")

        # # Save the last set of clusters for visualisation
        # cl_ds = driver1.Create(os.path.join(self.output_dir, self.name + '_clusters.tif'),
        #                     sorted_labels.shape[1], 
        #                     sorted_labels.shape[0], 
        #                     1, 
        #                     gdal.GDT_Int16)
        # cl_ds.SetGeoTransform(geot)
        # cl_ds.SetProjection(proj)
        # cl_band = cl_ds.GetRasterBand(1)
        # cl_band.WriteArray(sorted_labels)
        # cl_band.SetNoDataValue(np.nan)
        # cl_band.FlushCache()

        # Average the shadow mask
        av_mask = sum(shadows)
        vals = np.unique(av_mask)

        # Get product name of file
        name = os.path.splitext(self.filename)[0]

        # Rasterise the average shadow mask
        shadow_ds = driver1.Create(os.path.join(self.output_dir, name + '_shadow.tif'), 
                                av_mask.shape[1], 
                                av_mask.shape[0], 
                                1, 
                                gdal.GDT_Int16)
        shadow_ds.SetGeoTransform(geot)
        shadow_ds.SetProjection(proj)
        shadow_band = shadow_ds.GetRasterBand(1)
        shadow_band.WriteArray(av_mask)
        shadow_band.SetNoDataValue(0)
        shadow_band.FlushCache()

        # Create the shapefile layer to store the shadow polygon
        merged_ds = driver2.CreateDataSource(os.path.join(self.output_dir, name + '_shadow.shp'))
        merged_layer = merged_ds.CreateLayer('merged', srs=None)

        # Create the attribute field to store confidence values
        field = ogr.FieldDefn("detections", ogr.OFTInteger)
        merged_layer.CreateField(field)

        # Loop through all confidence values
        for val in vals[1:]:

            # Create the shapefile layer to store the shadow polygon
            shp_ds = driver2.CreateDataSource(os.path.join(self.output_dir, name + '_conf.shp'))
            shp_layer = shp_ds.CreateLayer('conf', srs=None)

            # Return array with only confidence value of 'val'
            coords = np.where(av_mask==val, val, 0)

            # Create a raster dataset from the confidence array
            c_ds = driver1.Create(os.path.join(self.output_dir, name + '_conf.tif'), 
                                    coords.shape[1], 
                                    coords.shape[0], 
                                    1, 
                                    gdal.GDT_Int16)
            c_ds.SetGeoTransform(geot)
            c_ds.SetProjection(proj)
            c_band = c_ds.GetRasterBand(1)
            c_band.WriteArray(coords)    
            c_band.SetNoDataValue(0)
            c_band.FlushCache()

            # Polygonise the raster dataset into the shapefile layer
            gdal.Polygonize(c_band, c_band, shp_layer, -1, [], callback=None)

            # Set up a multipolygon wkb to merge the individual features
            multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

            # Loop through each feature to add it to the multipolygon
            for feat in shp_layer:
                geom = feat.GetGeometryRef()
                multipolygon.AddGeometry(geom)

            # Check features exist for this confidence value
            if np.sum(coords) != 0:

                # Set the feature geometry using the polygon
                feature = ogr.Feature(merged_layer.GetLayerDefn())
                feature.SetField("detections", int(val))
                feature.SetGeometry(multipolygon)
                
                # Create the feature in the shapefile layer
                merged_layer.CreateFeature(feature)
                feature = None

            # Remove the shapefile containing individual confidence
            try:
                os.remove(os.path.join(self.output_dir, name + '_conf.shp'))
                os.remove(os.path.join(self.output_dir, name + '_conf.shx'))
                os.remove(os.path.join(self.output_dir, name + '_conf.dbf'))
                os.remove(os.path.join(self.output_dir, name + '_conf.tif'))
            except:
                print("shp_layer could not be removed.")

        # Close the output dataset, bands and layers
        c_ds = shadow_ds = merged_ds = None
        c_band = shadow_band = merged_layer = None