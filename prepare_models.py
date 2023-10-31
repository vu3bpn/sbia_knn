#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:59:49 2021

@author: Bipin Chandran 
"""

from osgeo import ogr, gdal
from osgeo.gdalconst import *
from utilities import *
from data_conf import *
import json
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from shapely import geometry
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.ensemble import RandomForestClassifier
import pickle

from warnings import simplefilter
simplefilter(action='ignore',category = FutureWarning)

plot_vect = True


def read_vectors(
    vect_fn=labeled_vect_fn,
    input_fn=input_vrt_fn,
    fields=['id', 'class'],
    buffer=0
):
    vect_file = ogr.Open(vect_fn, GA_ReadOnly)
    layer1 = vect_file.GetLayer()
    layer_idx_list = list(range(layer1.GetFeatureCount()))
    input_file = gdal.Open(input_fn)
    geotransform = input_file.GetGeoTransform()
    projection = input_file.GetProjectionRef()
    data_bands = list(map(lambda x: input_file.GetRasterBand(
        x+1), range(input_file.RasterCount)))
    feat_vect_list = []
    for layer_idx1 in layer_idx_list:
        feat1 = layer1.GetFeature(layer_idx1)
        geom1 = feat1.GetGeometryRef()
        feat_vect = {}
        for field1 in fields:
            feat_vect[field1] = feat1.GetField(field1)
        geom1_dict = json.loads(geom1.ExportToJson())
        coord_mat = np.array(geom1_dict['coordinates']).squeeze()
        min_lon, min_lat = np.min(coord_mat, axis=0)
        max_lon, max_lat = np.max(coord_mat, axis=0)
        min_lon = min_lon - buffer
        max_lon = max_lon + buffer
        min_lat = min_lat - buffer
        max_lat = max_lat + buffer

        pixel_list = list(map(lambda x: cord_to_pixel(
            x, geotransform), list(coord_mat)))
        shp1 = geometry.Polygon(pixel_list)

        ul_pixel = cord_to_pixel((min_lon, max_lat), geotransform)
        lr_pixel = cord_to_pixel((max_lon, min_lat), geotransform)
        feat_size = np.array(lr_pixel) - np.array(ul_pixel)

        data_mat_list = []
        for band1 in data_bands:
            data_mat = band1.ReadAsArray(int(ul_pixel[0]), int(
                ul_pixel[1]), int(feat_size[0]), int(feat_size[1])).T
            data_mat_list.append(data_mat)
        data_mat_stack = np.array(data_mat_list)
        x_0, y_0 = ul_pixel
        vect_list = []
        bg_vect_list = []
        for x_2, y_2 in itertools.product(range(ul_pixel[0], lr_pixel[0]), range(ul_pixel[1], lr_pixel[1])):
            x_3 = int(x_2-x_0)
            y_3 = int(y_2-y_0)
            if shp1.intersects(geometry.Point(x_2, y_2)):
                vect_list.append(data_mat_stack[:, x_3, y_3])
            else:
                bg_vect_list.append(data_mat_stack[:, x_3, y_3])

        feat_vect["vectors"] = vect_list
        feat_vect["bg_vectors"] = bg_vect_list
        feat_vect_list.append(feat_vect)

    feat_vect_dict = {}
    for feat_vect in feat_vect_list:
        if feat_vect['class'] in feat_vect_dict:
            feat_vect_dict[feat_vect['class']]['vectors'].extend(
                feat_vect['vectors'])
            feat_vect_dict[feat_vect['class']]['bg_vectors'].extend(
                feat_vect['bg_vectors'])
        else:
            feat_vect_dict[feat_vect['class']] = {
                'vectors': feat_vect['vectors'],
                'bg_vectors':feat_vect['bg_vectors']}
    return feat_vect_dict

def plot_feat_vect(vect,label,title):
    x, y = zip(*vect)
    #plt.figure()
    #plt.title(title)
    plt.scatter(x, y, c=label, alpha=0.5, marker="+")
    bulk_x,bulk_y = [],[]
    for idx in range(len(x)):
        if label[idx] == 0:
            bulk_x.append(x[idx])
            bulk_y.append(y[idx])
    min_x = min(bulk_x)
    min_y = min(bulk_y)
    max_x = max(bulk_x)
    max_y = max(bulk_y)
    
    title1 = title.split('.')[0].split('_')[-1]
    plt.plot([min_x,min_x,max_x,max_x,min_x],[min_y,max_y,max_y,min_y,min_y],c='blue')
    plt.annotate(title1, ((min_x+max_x)/2,(min_y+max_y)/2))


def prepare_model(vect0,
                  model_fn,
                  levels_population_ratio,
                  num_vect=max_vectors_per_class,
                  ):
    
    if len(vect0) > num_vect:
        vect1 = random.sample(vect0, k=num_vect)
        vect2 = random.sample(vect0, k=num_vect)
    else:
        vect1 = vect0
        vect2 = vect0
    max_vect = np.max(np.array(vect1), axis=0)
    min_vect = np.min(np.array(vect1), axis=0)
    mean_vect = np.mean(np.array(vect1), axis=0)

    noise_gain = (max_vect-min_vect)*1.5
    n_vect, vect_len = np.array(vect1).shape
    n_noise_vect = n_vect*2

    noise_vect = np.random.random((n_noise_vect, vect_len))

    noise_vect = (noise_vect-0.5) * \
        np.repeat(np.expand_dims(noise_gain, 0), n_noise_vect, axis=0) +\
        np.repeat(np.expand_dims(mean_vect, 0), n_noise_vect, axis=0)

    noise_vect = list(noise_vect)
    feat_vect = vect1+noise_vect
    target_vect = list(np.ones(len(vect1))) + \
        list(np.zeros(len(noise_vect)))

    classifier_bulk = KNeighborsClassifier(n_neighbors=n_neighbours_bulk)
    classifier_bulk.fit(feat_vect, target_vect)
    predict_proba = classifier_bulk.predict_proba(vect2)

    vect_idx_list = list(range(len(vect2)))
    vect_idx_list = sorted(
        vect_idx_list, key=lambda x: predict_proba[x, 1], reverse=True)
    sorted_vect = [vect2[x] for x in vect_idx_list]

    levels_vect = []
    levels_pop_frac = np.array(
        levels_population_ratio)/sum(levels_population_ratio)
    for level in range(len(levels_pop_frac)):
        levels_vect += list(
            np.ones(int(levels_pop_frac[level]*len(sorted_vect)))*level)

    if len(sorted_vect) > len(levels_vect):
        levels_vect += list(np.ones(len(sorted_vect) -
                            len(levels_vect))*level)
    else:
        levels_vect = levels_vect[:len(sorted_vect)]

    classifier_level = KNeighborsClassifier(n_neighbors=n_neighbours_level)
    classifier_level.fit(sorted_vect, levels_vect)

    pickle.dump(classifier_level, open(
        os.path.join(model_dir, model_fn), 'wb'))
    print("Model size :{0}".format(len(sorted_vect)))
    print("saved model :{0}".format(model_fn))

    if plot_vect:
        sorted_vect.reverse()
        levels_vect.reverse()
        plot_feat_vect(sorted_vect, levels_vect, model_fn)
        
def prepare_model_bg(vect_dict,
                     model_fn,
                     levels_population_ratio = [1,8,5],
                     max_vectors=max_vectors_per_class
                     ):
    vect0 = vect_dict['vectors']
    vect_bg = vect_dict['bg_vectors']

    num_vect = min(max_vectors,len(vect0))
    num_vect_bg = min(max_vectors,len(vect_bg))

    level_pop_tot = sum(levels_population_ratio)
    level1_pop = int(max_vectors*levels_population_ratio[0]/level_pop_tot)
    level2_pop = int(max_vectors*levels_population_ratio[1]/level_pop_tot)
    level3_pop = int(max_vectors*levels_population_ratio[2]/level_pop_tot)

    vect1 = random.sample(vect0, k=num_vect)
    vect1_bg = random.sample(vect_bg, k=num_vect_bg)

    vect2 = vect0
    vect2_bg = vect_bg

    feat_vect = []
    feat_vect.extend(vect1)
    feat_vect.extend(vect1_bg)
    target_vect = []
    target_vect.extend(list(np.ones(num_vect)))
    target_vect.extend(list(np.zeros(num_vect_bg)))

    classifier_bulk = KNeighborsClassifier(n_neighbors=n_neighbours_bulk)
    classifier_bulk.fit(feat_vect, target_vect)

    predict_proba = classifier_bulk.predict_proba(vect2)
    vect_idx_list = list(range(len(vect2)))
    proba_idx1 = list(classifier_bulk.classes_).index(1)

    vect_idx_list = list(filter(lambda x: predict_proba[x,proba_idx1]>0.51 ,vect_idx_list))
    num_vect2     = min(level2_pop,len(vect_idx_list)-level1_pop)


    vect_idx_list = sorted(
        vect_idx_list, key=lambda x: predict_proba[x,proba_idx1], reverse=True)

    vect_idx_list_0 = vect_idx_list[:level1_pop]

    vect_idx_list_1 = random.sample(vect_idx_list[level1_pop:],k=num_vect2)

    vect_idx_list = vect_idx_list_0+vect_idx_list_1

    sorted_core_vect = [vect2[x] for x in vect_idx_list]
    level_vect = list(np.zeros(len(vect_idx_list_0)))
    level_vect.extend(list(np.ones(len(vect_idx_list_1 ))))

    num_vect3 = min(level3_pop,len(vect2_bg))
    vect_bg_idx_list = list(range(len(vect2_bg)))
    vect_bg_idx_list = random.sample(vect_bg_idx_list, k=num_vect3)
    level3_vect = [vect2_bg[x] for x in vect_bg_idx_list]

    feat_vect2 = sorted_core_vect + level3_vect
    level_vect.extend(list(2*np.ones(num_vect3)))

    classifier_level = KNeighborsClassifier(n_neighbors=n_neighbours_level)
    classifier_level.fit(feat_vect2, level_vect)

    pickle.dump(classifier_level, open(
        os.path.join(model_dir, model_fn), 'wb'))
    print("model size :{0} min vect {1}".format(len(feat_vect2),min(len(vect_idx_list_0),len(vect_idx_list_1),len(level3_vect))))
    print("saved model :{0}".format(model_fn))
    if plot_vect:
        feat_vect2.reverse()
        level_vect.reverse()
        plot_feat_vect(feat_vect2, level_vect, model_fn)
        vect_fn = os.path.join(data_base_dir,
                               'debug',
                               vector_fn_temp.format(model_fn,experiment_id)
                               )
        model_vectors = {'vect':feat_vect2,'labels':level_vect}
        pickle.dump( model_vectors,
                    open(vect_fn,'wb')
                    )
        print(f"generated {vect_fn}")



def prepare_urban_model(vect0,
                        model_fn,
                        levels_population_ratio,
                        num_vect=max_vectors_per_class,
                        sigma0_in_db=False
                        ):
    if num_vect < len(vect0):
        vect1 = list(random.sample(vect0, k=num_vect))
    else:
        vect1 = list(vect0)
    sorted_vect = sorted(vect1, key=lambda x: np.sum(np.power(x, 2)))
    if not sigma0_in_db:
        sorted_vect.reverse()

    levels_pop_frac = np.array(
        levels_population_ratio)/sum(levels_population_ratio)
    levels_vect = []
    for level in range(len(levels_pop_frac)):
        levels_vect += list(
            np.ones(int(levels_pop_frac[level]*len(sorted_vect)))*level)

    if len(sorted_vect) > len(levels_vect):
        levels_vect += list(np.ones(len(sorted_vect) -
                            len(levels_vect))*level)
    else:
        levels_vect = levels_vect[:len(sorted_vect)]

    classifier_level = KNeighborsClassifier(n_neighbors=n_neighbours_level)
    classifier_level.fit(sorted_vect, levels_vect)
    pickle.dump(classifier_level, open(
        os.path.join(model_dir, model_fn), 'wb'))
    print("Model size :{0}".format(len(sorted_vect)))
    print("Saved model :{0}".format(model_fn))
    if plot_vect:
        sorted_vect.reverse()
        levels_vect.reverse()
        plot_feat_vect(sorted_vect, levels_vect, model_fn)


def prepare_water_model(vect0,
                        model_fn,
                        levels_population_ratio,
                        num_vect=max_vectors_per_class,
                        sigma0_in_db=False
                        ):
    if num_vect < len(vect0):
        vect1 = list(random.sample(vect0, k=num_vect))
    else:
        vect1 = list(vect0)
    sorted_vect = sorted(vect1, key=lambda x: np.sum(np.power(x, 2)))
    if sigma0_in_db:
        sorted_vect.reverse()

    levels_pop_frac = np.array(
        levels_population_ratio)/sum(levels_population_ratio)
    levels_vect = []
    for level in range(len(levels_pop_frac)):
        levels_vect += list(
            np.ones(int(levels_pop_frac[level]*len(sorted_vect)))*level)

    if len(sorted_vect) > len(levels_vect):
        levels_vect += list(np.ones(len(sorted_vect) -
                            len(levels_vect))*level)
    else:
        levels_vect = levels_vect[:len(sorted_vect)]

    classifier_level = KNeighborsClassifier(n_neighbors=n_neighbours_level)
    classifier_level.fit(sorted_vect, levels_vect)
    pickle.dump(classifier_level, open(
        os.path.join(model_dir, model_fn), 'wb'))
    print("Model size :{0}".format(len(sorted_vect)))
    print("Saved model :{0}".format(model_fn))
    if plot_vect:
        sorted_vect.reverse()
        levels_vect.reverse()
        plot_feat_vect(sorted_vect, levels_vect, model_fn)



def prepare_models(vect_dict,
                   fn_temp=model_fn_temp):
    for feat1 in vect_dict:
        vect0 = vect_dict[feat1]['vectors']
        model_fn = fn_temp.format(class_name=feature_mapping[feat1])
        if feat1 == 2:
            prepare_urban_model(vect0,
                                model_fn,
                                levels_population_ratio=population_ratio_dict[feat1])
        elif feat1 == 1:
            prepare_water_model(vect0,
                                model_fn,
                                levels_population_ratio=population_ratio_dict[feat1])
        else:
            vect0 = vect_dict[feat1]
            prepare_model_bg(vect0,
                             model_fn,
                             levels_population_ratio=population_ratio_dict[feat1])


def prepare_segment_classifier(vect_dict,
                               model_fn,
                               num_vect=max_vectors_per_class,
                               n_neighbors=n_neighbours_segment):
    vect_list = []
    feat_list = []
    for feat in vect_dict:
        if len(vect_dict[feat]['vectors']) > num_vect:
            vect_list.extend(random.sample(
                vect_dict[feat]['vectors'], k=num_vect))
            feat_list.extend(list(np.ones(num_vect)*feat))
        else:
            vect_list.extend(vect_dict[feat]['vectors'])
            feat_list.extend(
                list(np.ones(len(vect_dict[feat]['vectors']))*feat))
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(vect_list, feat_list)
    pickle.dump(model, open(
        os.path.join(model_dir, model_fn), 'wb'))
    print("saved model :{0}".format(model_fn))


if __name__ == "__main1__":
    '''prepare data'''
    clip_data_ee_sentinel(labeled_vect_fn,
                          data_folder="ee_data_train",
                          fn_temp="Sentinel_train_aoi_{0}"
                          )
if __name__ == "__main1__":
    vect_dict = read_vectors(vect_fn=labeled_vect_fn,
                             input_fn=validation_train_data_vrt_fn)
    prepare_models(vect_dict)
    prepare_segment_classifier(vect_dict,
                               classifier_model_fn,
                               num_vect=500)


if __name__ == "__main1__":
    num_vect = 5000
    vect_dict = read_vectors()
    prepare_models(vect_dict)
