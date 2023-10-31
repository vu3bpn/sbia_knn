#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 21:05:13 2021

@author: Bipin Chandran 
"""
import json
import numpy as np

def pixel_to_cord(p0, transform):
    x = p0[0]*transform[1] + transform[0]
    y = p0[1]*transform[5] + transform[3]
    return (x, y)


def cord_to_pixel(cord, geo_trans):
    x1 = (cord[0] - geo_trans[0])/geo_trans[1]
    y1 = (cord[1] - geo_trans[3])/geo_trans[5]
    return [int(x1), int(y1)]

def get_ul_size(geom,geotransform):
    coord_mat = np.array(geom['coordinates']).squeeze()
    min_lon, min_lat = np.min(coord_mat, axis=0)
    max_lon, max_lat = np.max(coord_mat, axis=0)
    pixel_list = list(map(lambda x: cord_to_pixel(
        x, geotransform), list(coord_mat)))
    ul_pixel = cord_to_pixel((min_lon, max_lat), geotransform)
    lr_pixel = cord_to_pixel((max_lon, min_lat), geotransform)
    feat_size = np.array(lr_pixel) - np.array(ul_pixel)
    return ul_pixel,lr_pixel,feat_size
