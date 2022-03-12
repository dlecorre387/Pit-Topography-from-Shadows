import os
import time
import numpy as np
import math as m
from osgeo import gdal, ogr
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, opening
from scipy.ndimage.morphology import binary_fill_holes

class DTMAnalyser(object):

    def __init__(self,
                dtm_filename,
                ortho_filename,
                dtm_dir,
                labels_dir,
                output_dir,
                n_clusters):
        self.dtm_filename = dtm_filename
        self.ortho_filename = ortho_filename
        self.dtm_dir = dtm_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.n_clusters = n_clusters

    def read_and_crop_DTM(self):
        
        # Get product name of file
        name = os.path.splitext(self.dtm_filename)[0]

        # Define filenames for the labels, DTM, and cropped output
        input_path = os.path.join(self.dtm_dir, self.dtm_filename)
        shp_path = os.path.join(self.labels_dir, name + '.shp')
        output_path = os.path.join(self.output_dir, name + '_cropped.tif')

        # Clip the DTM using the shapefile label
        dtm_ds = gdal.Warp(output_path, 
                        input_path, 
                        cutlineDSName=shp_path,
                        cropToCutline=True,
                        dstNodata = 0)
        
        # Close the raster dataset to save it
        dtm_ds = None

        # Read croped raster file as a NumPy array
        counter = 0
        if (name + '_cropped.tif') in os.listdir(self.output_dir):
            ds = gdal.Open(output_path) # GDAL Dataset
            geot = ds.GetGeoTransform() # GeoTransform
            proj = ds.GetProjection() # Projection
            if ds.RasterCount > 1:
                raise ValueError("More than one raster band present")
            red = ds.GetRasterBand(1)
            cropped_dtm = red.ReadAsArray()
            
        elif (name + '_cropped.tif') not in os.listdir(self.output_dir) and counter < 3:
            time.sleep(60)
            counter += 1

        else:
            raise ValueError("No cropped image file present")

        return cropped_dtm, geot, proj

    def read_cropped_DTM(self):

        # Open the cropped raster file as a NumPy array
        ds = gdal.Open(os.path.join(self.dtm_dir, self.dtm_filename)) # GDAL Dataset
        geot = ds.GetGeoTransform() # GeoTransform
        proj = ds.GetProjection() # Projection
        if ds.RasterCount > 1:
            raise ValueError("More than one raster band present")
        red = ds.GetRasterBand(1)
        cropped_dtm = red.ReadAsArray()

        return cropped_dtm, geot, proj

    def find_footprint(self, cropped_dtm, geot, proj):

        # Get product name of file
        name = os.path.splitext(self.dtm_filename)[0]

        # Calculate gradient
        xgradient = np.gradient(cropped_dtm, axis=0)
        ygradient = np.gradient(cropped_dtm, axis=1)
        gradient = (np.sqrt(xgradient**2) + np.sqrt(ygradient**2))/2

        # Calculate curvature
        xcurv = np.sqrt(np.gradient(xgradient, axis=0)**2)
        ycurv = np.sqrt(np.gradient(ygradient, axis=1)**2)
        curvature = (xcurv + ycurv)/2
        
        # Find the footprint of the pit based on the st. dev. of the elevation values
        thresh = np.amin(cropped_dtm) + np.std(cropped_dtm)
        footprint = binary_fill_holes(np.where(cropped_dtm < thresh, 1, 0)).astype(int)

        # Define driver for saving raster files
        driver1 = gdal.GetDriverByName("GTiff")

        # Save the curvature array
        grad_ds = driver1.Create(os.path.join(self.output_dir, name + '_gradient.tif'),
                            gradient.shape[1], 
                            gradient.shape[0], 
                            1, 
                            gdal.GDT_Float32)
        grad_ds.SetGeoTransform(geot)
        grad_ds.SetProjection(proj)
        grad_band = grad_ds.GetRasterBand(1)
        grad_band.WriteArray(gradient)
        grad_band.SetNoDataValue(np.nan)
        grad_band.FlushCache()
        grad_band=None

        # Save the curvature array
        curv_ds = driver1.Create(os.path.join(self.output_dir, name + '_curvature.tif'),
                            curvature.shape[1], 
                            curvature.shape[0], 
                            1, 
                            gdal.GDT_Float32)
        curv_ds.SetGeoTransform(geot)
        curv_ds.SetProjection(proj)
        curv_band = curv_ds.GetRasterBand(1)
        curv_band.WriteArray(curvature)
        curv_band.SetNoDataValue(np.nan)
        curv_band.FlushCache()
        curv_band=None

        # Save the curvature array
        foot_ds = driver1.Create(os.path.join(self.output_dir, name + '_footprint.tif'),
                            footprint.shape[1], 
                            footprint.shape[0], 
                            1, 
                            gdal.GDT_Int16)
        foot_ds.SetGeoTransform(geot)
        foot_ds.SetProjection(proj)
        foot_band = foot_ds.GetRasterBand(1)
        foot_band.WriteArray(footprint)
        foot_band.SetNoDataValue(0)
        foot_band.FlushCache()
        foot_band=None

        return gradient, curvature, footprint

    def cluster_ortho_image(self, cropped_im):

        # Create empty lists to store shadow masks and width estimates
        shadows = []

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
        
            # Update and close progress bar
            pbar.update(1)
        pbar.close()

        return shadows

    def save_outputs(self, geot, proj, shadows):

        driver1 = gdal.GetDriverByName("GTiff")
        driver2 = ogr.GetDriverByName("ESRI Shapefile")

        # Average the shadow mask
        av_mask = sum(shadows)
        vals = np.unique(av_mask)

        # Get product name of file
        name = os.path.splitext(self.ortho_filename)[0]

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

    # def save_ortho_shadow(self, sorted_labels, geot, proj):

    #     driver1 = gdal.GetDriverByName("GTiff")
    #     driver2 = ogr.GetDriverByName("ESRI Shapefile

    #     # Create the shapefile layer to store the shadow polygon
    #     shp_ds = driver2.CreateDataSource(os.path.join(self.output_dir, self.name + '_shadow.shp'))
    #     shp_layer = shp_ds.CreateLayer('shadow', srs=None)
        
    #     # Create the attribute field to store confidence values
    #     field = ogr.FieldDefn("detections", ogr.OFTInteger)
    #     shp_layer.CreateField(field)
        
    #     # Create a raster dataset from the confidence array
    #     c_ds = driver1.Create(os.path.join(self.output_dir, self.name + '_shadow.tif'), 
    #                             new_mask.shape[1], 
    #                             new_mask.shape[0], 
    #                             1, 
    #                             gdal.GDT_Int16)
    #     c_ds.SetGeoTransform(geot)
    #     c_ds.SetProjection(proj)
    #     c_band = c_ds.GetRasterBand(1)
    #     c_band.WriteArray(new_mask)    
    #     c_band.SetNoDataValue(0)
    #     c_band.FlushCache()

    #     # Polygonise the raster dataset into the shapefile layer
    #     gdal.Polygonize(c_band, c_band, shp_layer, -1, [], callback=None)

    #     # Set up a multipolygon wkb to merge the individual features
    #     multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

    #     # Loop through each feature to add it to the multipolygon
    #     for feat in shp_layer:
    #         geom = feat.GetGeometryRef()
    #         multipolygon.AddGeometry(geom)

    #     # Check features exist for this confidence value
    #     if np.sum(new_mask) != 0:

    #         # Set the feature geometry using the polygon
    #         feature = ogr.Feature(shp_layer.GetLayerDefn())
    #         feature.SetField("detections", 1)
    #         feature.SetGeometry(multipolygon)
            
    #         # Create the feature in the shapefile layer
    #         shp_layer.CreateFeature(feature)
    #         feature = None

    #     # Close the output dataset, bands and layers
    #     cl_ds = c_ds = shp_ds = None
    #     cl_band = c_band = shp_layer = None