#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 09:42:56 2022

@author: Bipin Chandran 
"""
import multiprocessing
import os
from search_alg import *
from data_conf import *

if __name__ == "__main__":
    '''validation popeline'''
    process_per_class = 5
    if os.path.exists(output_fn):
        os.remove(output_fn)

    data_q_1 = multiprocessing.Queue()
    data_q_2 = multiprocessing.Queue()
    data_q_3 = multiprocessing.Queue()
    segment_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()

    data_q_list = [data_q_1,
                   data_q_2,
                   data_q_3
                   ]

    model_fn1 = os.path.join(
        model_dir, model_fn_temp.format(class_name='Water'))
    model_fn2 = os.path.join(
        model_dir, model_fn_temp.format(class_name='Urban'))

    iter1 = raster_shp_iteraror(validation_data_vrt_fn, validation_aoi_fn)
    water_search = segment_search(model_fn1, 1, speckle_size)
    urban_search = segment_search(model_fn2, 2, speckle_size)
    tree_search = segment_search(model_fn2, 6, speckle_size)

    seg_classify = segment_classifier(os.path.join(
        model_dir, classifier_model_fn), validation_data_vrt_fn)
    result_sink = raster_sink(raster_fn=output_fn,
                              band_size=(iter1.raster.RasterXSize,
                                         iter1.raster.RasterYSize),
                              geotrans=iter1.geotrans,
                              crs=iter1.crs
                              )
    processing_task_list = []

    for idx in range(process_per_class):
        water_task = multiprocessing.Process(
            target=water_search.run_dmn, args=(data_q_1, segment_q))
        water_task.start()
        processing_task_list.append(water_task)

        urban_task = multiprocessing.Process(
            target=urban_search.run_dmn, args=(data_q_2, segment_q))
        urban_task.start()
        processing_task_list.append(urban_task)

        tree_task = multiprocessing.Process(
            target=tree_search.run_dmn, args=(data_q_3, segment_q))
        tree_task.start()
        processing_task_list.append(tree_task)

    segment_classifier_task = multiprocessing.Process(
        target=seg_classify.run_dmn, args=(segment_q, result_q))
    segment_classifier_task.start()

    result_task = multiprocessing.Process(
        target=result_sink.run_dmn, args=(result_q,))
    result_task.start()

    for data, ul_pixel in iter1:
        for data_q in data_q_list:
            data_q.put((data, ul_pixel))

    for data_q in data_q_list:
        for _ in range(process_per_class):
            data_q.put(None)

    for task in processing_task_list:
        task.join()

    segment_q.put(None)
    segment_classifier_task.join()
    result_q.put(None)
    result_task.join()
    print("Done")


if __name__ == "__main1__":
    data_q = multiprocessing.Queue()
    data_q_2 = multiprocessing.Queue()
    segment_q = multiprocessing.Queue()

    model_fn1 = os.path.join(
        model_dir, model_fn_temp.format(class_name='Water'))

    aoi_fn = r"data/Vectors/AOI3.shp"

    iter1 = raster_shp_iteraror(input_vrt_fn, aoi_fn)
    water_search = segment_search(model_fn1, 1, speckle_size)

    result_sink = raster_sink(raster_fn=output_fn,
                              band_size=(iter1.raster.RasterXSize,
                                         iter1.raster.RasterYSize),
                              geotrans=iter1.geotrans,
                              crs=iter1.crs
                              )

    water_task = multiprocessing.Process(
        target=water_search.run_dmn, args=(data_q_1, segment_q))
    water_task.start()

    result_task = multiprocessing.Process(
        target=result_sink.run_dmn, args=(segment_q,))
    result_task.start()

    for data, ul_pixel in iter1:
        data_q.put((data, ul_pixel))
    data_q.put(None)
    water_task.join()
    segment_q.put(None)
    result_task.join()
    print("Done")
