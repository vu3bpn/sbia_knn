#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:46:46 2021

@author: Bipin Chandran
"""

from osgeo import gdal, osr, ogr
from osgeo.gdalconst import *
import itertools
import numpy as np
from skimage import measure
from scipy.spatial import Delaunay, kdtree
from shapely.ops import cascaded_union, unary_union, nearest_points
from shapely.affinity import affine_transform
import shapely.geometry as sg
from data_conf import *
#from data_conf_for_paper import *
import json
from utilities import *
#import ee
from shapely.geometry import Polygon
import shapely


class raster_shp_iteraror:
    def __init__(self, raster_fn, shp_fn,offset=0):
        self.raster = gdal.Open(raster_fn, GA_ReadOnly)
        self.shp_file = ogr.Open(shp_fn, GA_ReadOnly)
        self.layer = self.shp_file.GetLayer()
        self.idx = 0
        self.num_features = self.layer.GetFeatureCount()
        self.band_list = [self.raster.GetRasterBand(
            band_idx+1) for band_idx in range(self.raster.RasterCount)]
        self.geotrans = self.raster.GetGeoTransform()
        self.crs = self.raster.GetProjectionRef()
        self.offset=offset

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.num_features:
            raise StopIteration
        feat = self.layer.GetFeature(self.idx)
        geom = feat.GetGeometryRef()
        geom_json = json.loads(geom.ExportToJson())
        coordinates = geom_json["coordinates"]
        min_lon, min_lat = np.min(np.array(coordinates).squeeze(), axis=0)
        max_lon, max_lat = np.max(np.array(coordinates).squeeze(), axis=0)
        min_lon = min_lon - self.offset
        max_lon = max_lon + self.offset
        min_lat = min_lat - self.offset
        max_lat = max_lat + self.offset
        ul_pixel = cord_to_pixel((min_lon, max_lat), self.geotrans)
        lr_pixel = cord_to_pixel((max_lon, min_lat), self.geotrans)

        data_list = [band.ReadAsArray(ul_pixel[0], ul_pixel[1], lr_pixel[0] -
                                      ul_pixel[0], lr_pixel[1]-ul_pixel[1]).T for band in self.band_list]
        self.idx += 1
        return np.array(data_list), ul_pixel


def clip_data_ee_sentinel(aoi_fn,
                          start_date='2020-01-01',
                          end_date='2020-02-01',
                          sentinel_ee_name='COPERNICUS/S1_GRD',
                          fn_temp="Sentinel_aoi_{0}",
                          data_folder="ee_data"
                          ):
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    shp_file = ogr.Open(aoi_fn, GA_ReadOnly)
    layer = shp_file.GetLayer()
    idx = 0
    num_features = layer.GetFeatureCount()

    for idx in range(num_features):
        feat = layer.GetFeature(idx)
        geom = feat.GetGeometryRef()
        geom_json = json.loads(geom.ExportToJson())

        lon_list, lat_list = zip(*geom_json['coordinates'][0])

        rectangle_aoi = Polygon([[min(lon_list), max(lat_list)],
                                 [max(lon_list), max(lat_list)],
                                 [max(lon_list), min(lat_list)],
                                 [min(lon_list), min(lat_list)]])

        ee_geom = ee.Geometry(shapely.geometry.mapping(rectangle_aoi))

        sentinel1 = ee.Image(ee.ImageCollection(sentinel_ee_name)
                             .filterBounds(ee_geom)
                             .filterDate(ee.Date(start_date), ee.Date(end_date))
                             .first()
                             .select("VV", "VH")
                             .toFloat()
                             )
        sentinel1_data = sentinel1.clip(ee_geom)
        task1 = ee.batch.Export.image.toDrive(image=sentinel1_data,
                                              description=fn_temp.format(idx),
                                              folder=data_folder,
                                              scale=10,
                                              fileFormat='GeoTIFF',
                                              crs='EPSG:4326'
                                              )
        task1.start()


def clip_data_ee_ref(aoi_fn,
                     ref_ee_name="ESA/WorldCover/v100",
                     fn_temp="Land_cover_aoi_{0}",
                     data_folder="ee_data"
                     ):
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    world_cover = ee.Image(ee.ImageCollection(ref_ee_name).first())
    shp_file = ogr.Open(aoi_fn, GA_ReadOnly)
    layer = shp_file.GetLayer()
    idx = 0
    num_features = layer.GetFeatureCount()
    for idx in range(num_features):
        feat = layer.GetFeature(idx)
        geom = feat.GetGeometryRef()
        geom_json = json.loads(geom.ExportToJson())
        lon_list, lat_list = zip(*geom_json['coordinates'][0])
        rectangle_aoi = Polygon([[min(lon_list), max(lat_list)],
                                 [max(lon_list), max(lat_list)],
                                 [max(lon_list), min(lat_list)],
                                 [min(lon_list), min(lat_list)]])

        ee_geom = ee.Geometry(shapely.geometry.mapping(rectangle_aoi))
        land_cover = world_cover.clip(ee_geom)
        task1 = ee.batch.Export.image.toDrive(image=land_cover,
                                              description=fn_temp.format(idx),
                                              folder=data_folder,
                                              scale=10,
                                              fileFormat='GeoTIFF',
                                              crs='EPSG:4326'
                                              )
        task1.start()


def build_tiled_vrt(data_dir,
                    data_fn_prefix,
                    vrt_fn):
    fn_list = os.listdir(data_dir)
    fn_list = list(filter(lambda x: x.endswith(".tif")
                          and x.startswith(data_fn_prefix), fn_list))
    fn_list_full = [os.path.join(data_dir,fn) for fn in fn_list]
    vrt_opts = gdal.BuildVRTOptions(separate=False,
                                    resolution='highest')
    gdal.BuildVRT(vrt_fn, fn_list_full, options=vrt_opts)
    print("generated :{0}".format(vrt_fn))


class raster_sink:
    def __init__(self, raster_fn, band_size, geotrans, crs):
        self.raster_fn = raster_fn
        self.band_size = band_size
        self.geotrans = geotrans
        self.crs = crs

    def create_file(self):
        self.driver = gdal.GetDriverByName('GTiff')
        if not os.path.exists(self.raster_fn):
            self.raster = self.driver.Create(
                self.raster_fn,
                self.band_size[0],
                self.band_size[1],
                1,
                gdal.GDT_Float32,
                options= ['COMPRESS=LZW'])
            self.raster.SetGeoTransform(self.geotrans)
            self.raster.SetProjection(self.crs)
            print("Created file {0}".format(self.raster_fn))
        else:
            self.raster = gdal.Open(self.raster_fn, GA_Update)
            print("Updating file {0}".format(self.raster_fn))
        self.band = self.raster.GetRasterBand(1)
        self.band.SetNoDataValue(0)

    def close(self):
        self.band.SetNoDataValue(0)
        self.band.FlushCache()
        self.band = None
        self.raster = None
        self.driver = None
        print("Closed file :{0}".format(self.raster_fn))

    def write_segment(self, segment):
        if segment.predicted_class> 0 and segment.class_id != segment.predicted_class:
            #segment.class_id = segment.predicted_class
            pass
        data = segment.combine_levels()
        data_band = self.band.ReadAsArray(
            segment.ul_pixel[0], segment.ul_pixel[1], segment.data_shape[0], segment.data_shape[1])
        data = np.array(data.T)
        data_band = np.array(data_band)
        data_updated = data + data_band*(data == 0)

        self.band.WriteArray(
            data_updated, segment.ul_pixel[0], segment.ul_pixel[1])
        self.band.FlushCache()

    def run_dmn(self, input_q):
        self.create_file()
        segments = input_q.get()
        while segments is not None:
            for segment in segments:
                self.write_segment(segment)
            segments = input_q.get()
        self.close()

def convert_4326(fn):
    gdal_file = gdal.Open(fn,GA_ReadOnly)
    spatial_ref = osr.SpatialReference(wkt=gdal_file.GetProjectionRef())
    spatial_ref.AutoIdentifyEPSG()
    if spatial_ref.GetAuthorityCode(None) != '4326':
        warp_opts = gdal.WarpOptions(format="GTiff",
                                     dstSRS="EPSG:4326",
                                     )
        gdal.Warp(fn, fn, options=warp_opts)
        print("converted to EPSG:4326  {0}".format(fn))

def resample_data(src_fn,target_fn):
    target_file = gdal.Open(target_fn,GA_ReadOnly)
    src_file    = gdal.Open(src_fn,GA_ReadOnly)
    target_geotrans = target_file.GetGeoTransform()
    src_geotrans    = src_file.GetGeoTransform()
    if target_geotrans[1] == src_geotrans[1] and target_geotrans[5] == src_geotrans[5]:
        return
    warp_opts = gdal.WarpOptions(format="GTiff",
                                     xRes = abs(target_geotrans[1]) ,
                                     yRes = abs(target_geotrans[5]),
                                     resampleAlg = 'near'
                                     )
    gdal.Warp(src_fn, src_fn, options=warp_opts)
    print("Resampled ref data:{0}".format(src_fn))



if __name__ == "__main__":
    aoi_fn = r"data/Vectors/AOI2.shp"
    iter1 = raster_shp_iteraror(input_vrt_fn, aoi_fn)
    for item1 in iter1:
        pass
