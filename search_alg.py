#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:38:49 2021

@author: Bipin Chandran 
"""

import pickle
import numpy as np
import os
from data_conf import *
from data_utilities import *
import matplotlib.pyplot as plt
from collections import Counter


def neighbours(pos):
    return [(pos[0], pos[1]+1),
            (pos[0], pos[1]-1),
            (pos[0]+1, pos[1]),
            (pos[0]-1, pos[1]),
            ]


class search_result:
    def __init__(self):
        self.neighbours_list = []
        self.result_list = []
        self.working_list = []
        self.level = -1


class segment_result:
    def __init__(self,
                 ul_pixel=(0, 0),
                 data_shape=(100, 100),
                 class_id=1
                 ):
        self.level_results = []
        self.ul_pixel = ul_pixel
        self.data_shape = data_shape
        self.class_id = class_id
        self.predicted_class = 0

    def __iadd__(self, result):
        result.neighbours_list = []
        self.level_results.append(result)
        return self

    def combine_levels(self):
        combined_result = np.zeros(self.data_shape)
        for result in self.level_results:
            for p1 in result.result_list:
                combined_result[p1[0], p1[1]] = self.class_id  # result.level+1
        return combined_result

    def length(self):
        return sum([len(x.result_list) for x in self.level_results])


class level_search():
    def __init__(self, model, level3_size=15):
        self.model = model
        self.level3_size = level3_size
        self.result = search_result()

    def get_neighbours(self, pos):
        res, neigh = [], []
        if self.mask[pos[0], pos[1]] == 0:
            self.mask[pos[0], pos[1]] = 1
            neighbours_list = neighbours(pos)
            for n1 in neighbours_list:
                if n1[0] > 0 and n1[0] < self.data_shape[0] and n1[1] > 0 and n1[1] < self.data_shape[1]:
                    l1 = self.model.predict([self.data[:, n1[0], n1[1]]])
                    if l1 == self.result.level:
                        res.append(n1)
                    else:
                        neigh.append(n1)
        return res, neigh

    def bfs_iter(self):
        while len(self.result.working_list) > 0:
            self.pos = self.result.working_list.pop(0)
            if self.mask[self.pos[0], self.pos[1]] == 0:
                self.result.result_list.append(self.pos)
                res, neigh = self.get_neighbours(self.pos)
                self.result.working_list.extend(res)
                self.result.neighbours_list.extend(neigh)
                if len(self.result.result_list) > self.level3_size and self.result.level == 2:
                    self.result = search_result()

    def search(self, p1):
        if self.mask[p1[0], p1[1]] == 0:
            sample = self.data[:, p1[0], p1[1]]
            self.result.level = self.model.predict([sample])[0]
            self.result.working_list.append(p1)
            self.bfs_iter()

    def load_data(self, data, mask=None):
        self.data = data
        if mask == None:
            self.mask = np.zeros(self.data.shape[1:])
        else:
            self.mask = mask
        self.data_shape = data.shape[1:]


class segment_search:
    def __init__(self,
                 model_fn,
                 feature_id,
                 level3_size):
        self.model_fn = model_fn
        self.level_model = pickle.load(open(model_fn, 'rb'))
        self.level3_size = level3_size
        self.level_search = level_search(self.level_model,
                                         self.level3_size)
        self.feature_id = feature_id
        self.result = segment_result(class_id=self.feature_id)
        self.working_list = []
        self.feature_name = feature_mapping[self.feature_id]

    def is_travelled(self, p1):
        return self.level_search.mask[p1[0], p1[1]] != 0

    def search(self, p1):
        self.working_list.append(p1)
        while len(self.working_list) > 0:
            p1 = self.working_list.pop(0)
            self.level_search.search(p1)
            if self.level_search.result.level >= 0:
                self.working_list.extend(
                    self.level_search.result.neighbours_list)
                self.result += self.level_search.result
            self.level_search.result = search_result()

    def load_data(self, data, ul_pixel=(0, 0)):
        self.level_search.load_data(data)
        self.ul_pixel = ul_pixel
        pass

    def get_level1_points(self, data):
        classes = self.level_model.predict(data.reshape(
            (data.shape[0], -1)).T).reshape(data.shape[1:])
        return list(np.argwhere(classes == 0))

    def search_all(self, data, ul_pixel):
        self.load_data(data, ul_pixel)
        l1_points = self.get_level1_points(data)
        segments_list = []
        for p1 in l1_points:
            if not self.is_travelled(p1):
                self.result = segment_result(
                    ul_pixel=ul_pixel, data_shape=data.shape[1:], class_id=self.feature_id)
                self.search(p1)
                segments_list.append(self.result)
        return segments_list

    def run_dmn(self, input_q, output_q):
        data_in = input_q.get()
        while data_in is not None:
            data, ul_pixel = data_in
            segments_list = self.search_all(data, ul_pixel)
            output_q.put(segments_list)
            '''
            for segment in segments_list:
                output_q.put(segment)
            '''
            data_in = input_q.get()
        print("Stopped search :{0}".format(self.feature_name))
        # output_q.put(None)


class segment_classifier:
    def __init__(self,
                 feature_model_fn,
                 data_fn):
        self.feature_model_fn = feature_model_fn
        self.data_fn = data_fn

    def load_models(self):
        self.feature_model = pickle.load(open(self.feature_model_fn, 'rb'))
        self.data_file = gdal.Open(self.data_fn, GA_ReadOnly)
        self.data_band_list = [self.data_file.GetRasterBand(
            band_id+1) for band_id in range(self.data_file.RasterCount)]

    def classify_segment(self, segment, data_mat):
        vectors = []
        for levels in segment.level_results:
            for p1 in levels.result_list:
                vectors.append(p1)
        vect_list = [data_mat[:, int(x), int(y)] for x, y in vectors]
        predicted_class = self.feature_model.predict(vect_list)
        counts = Counter(predicted_class)
        maj_class, cnt = counts.most_common()[0]
        segment.predicted_class = maj_class
        return segment

    def classify_segment_list(self, seg_list):
        seg_list_out = []
        if len(seg_list) > 0:
            segment = seg_list[0]
            ul_pixel = segment.ul_pixel
            data_shape = segment.data_shape
            data_list = [band.ReadAsArray(int(ul_pixel[0]), int(ul_pixel[1]), int(
                data_shape[0]), int(data_shape[1])).T for band in self.data_band_list]
            data_mat = np.array(data_list)
            seg_list_out = [self.classify_segment(
                segment, data_mat) for segment in seg_list]
        return seg_list_out

    def run_dmn(self, input_q, output_q):
        self.load_models()
        segments = input_q.get()
        while segments is not None:
            classified_seg = self.classify_segment_list(segments)
            output_q.put(classified_seg)
            segments = input_q.get()


if __name__ == "__main__":
    model_fn1 = os.path.join(
        model_dir, model_fn_temp.format(class_name='Water'))
    aoi_fn = os.path.join(data_base_dir, r"Vectors/AOI3.shp")
    iter1 = raster_shp_iteraror(input_vrt_fn, aoi_fn)
    segment_search = segment_search(model_fn1, 1, speckle_size)

    seg_classify = segment_classifier(os.path.join(
        model_dir, classifier_model_fn), input_vrt_fn)

    result_sink = raster_sink(raster_fn=output_fn,
                              band_size=(iter1.raster.RasterXSize,
                                         iter1.raster.RasterYSize),
                              geotrans=iter1.geotrans,
                              crs=iter1.crs
                              )

    segments_list = []
    for data, ul_pixel in iter1:
        print("\n*", end='')
        segment_search.load_data(data)
        l1_points = segment_search.get_level1_points(data)
        for p1 in l1_points:
            if not segment_search.is_travelled(p1):
                segment_search.result = segment_result(
                    ul_pixel=ul_pixel, data_shape=data.shape[1:])
                segment_search.search(p1)
                segments_list.append(segment_search.result)
                print('.', end='')

    result_sink.create_file()
    for segment in segments_list:
        if segment.length() > min_segment_size:
            segment = seg_classify.classify_segment_list([segment])
            result_sink.write_segment(segment)
    result_sink.close()
