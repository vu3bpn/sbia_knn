#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:19:45 2022

@author: Bipin Chandran 
"""

'''
Speckle analysis for SAR images
Analysis of speckle on uniform targets
'''

from osgeo import gdal, osr, ogr
from osgeo.gdalconst import *
import itertools
import numpy as np
from data_utilities import *
from data_conf import *
from prepare_models import *
from multiprocessing import Pool
import seaborn as sns
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.metrics import f1_score,accuracy_score
from search_alg import *


spk_aoi_fn          = os.path.join(vect_dir,"spk_aoi4.shp")
feat_vect_fn        = os.path.join(vect_dir,"spk_feat2.shp")
val_ref_fn          = os.path.join(vect_dir,"spk_val4.shp")

prefix_list         = ["lee_3_1_",'lee_3_2_',"lee_5_1_","gamma_5_1_","frost_5_1_","refined_lee_"]

model_fn_3_1        = os.path.join(model_dir,"Segment_classifier_3_1.model")
model_fn_3_2        = os.path.join(model_dir,"Segment_classifier_3_2.model")
model_fn_5_1        = os.path.join(model_dir,"Segment_classifier_5_1.model")
model_fn_raw        = os.path.join(model_dir,"Segment_classifier_raw.model")
model_fn_frost      = os.path.join(model_dir,"Segment_classifier_frost_5.model")
model_fn_gamma      = os.path.join(model_dir,"Segment_classifier_gamma_5.model")
model_fn_refined_lee      = os.path.join(model_dir,"Segment_classifier_refined_lee.model")

spk_input_vrt_3_1   = os.path.join(vrt_dir,"lee_3_1_"+"filtered_Sigma0.vrt")
spk_input_vrt_3_2   = os.path.join(vrt_dir,"lee_3_2_"+"filtered_Sigma0.vrt")
spk_input_vrt_5_1   = os.path.join(vrt_dir,"lee_5_1_"+"filtered_Sigma0.vrt")
spk_input_vrt_frost   = os.path.join(vrt_dir,"frost_5_1_"+"filtered_Sigma0.vrt")
spk_input_vrt_gamma   = os.path.join(vrt_dir,"gamma_5_1_"+"filtered_Sigma0.vrt")
spk_input_vrt_raw   = os.path.join(vrt_dir,"un_"+"filtered_Sigma0.vrt")
spk_input_vrt_refined_lee   = os.path.join(vrt_dir,"refined_lee_"+"filtered_Sigma0.vrt")

output_fn1 = os.path.join(output_dir,"Search_result_{0}.tif".format(experiment_name))
output_fn2 = os.path.join(output_dir,"knn_result_3_1_{0}.tif".format(experiment_name))
output_fn3 = os.path.join(output_dir,"knn_result_3_2_{0}.tif".format(experiment_name))
output_fn4 = os.path.join(output_dir,"knn_result_raw_{0}.tif".format(experiment_name))
output_fn5 = os.path.join(output_dir,"knn_result_5_1_{0}.tif".format(experiment_name))
output_fn6 = os.path.join(output_dir,"knn_result_frost_5_{0}.tif".format(experiment_name))
output_fn7 = os.path.join(output_dir,"knn_result_gamma_5_{0}.tif".format(experiment_name))
output_fn8 = os.path.join(output_dir,"knn_result_refined_lee_{0}.tif".format(experiment_name))
output_fn9 = os.path.join(output_dir,"pre_filt_search_result_{0}.tif".format(experiment_name))
output_fn10 = os.path.join(output_dir,"pre_filt2_search_result_{0}.tif".format(experiment_name))


#random.seed(experiment_id)
random.seed(2)

if __name__ == "__main1__":
    clip_data_ee(spk_aoi_fn)

if __name__ == "__main__":
    '''Prepare input VRTs'''
    VH_fn_list = list(map(lambda x: os.path.join(
        processed_data_dir, x.strip(".zip"), "Sigma0_VH.img"), data_fn_list))
    VV_fn_list = list(map(lambda x: os.path.join(
        processed_data_dir, x.strip(".zip"), "Sigma0_VV.img"), data_fn_list))
    vrt_opts = gdal.BuildVRTOptions(separate=False,
                                    srcNodata=0,
                                    resolution='highest')
    gdal.BuildVRT(spk_input_vv_vrt,
                  VV_fn_list,
                  options=vrt_opts)
    gdal.BuildVRT(spk_input_vh_vrt, VH_fn_list, options=vrt_opts)
    vrt_opts = gdal.BuildVRTOptions(separate=True,
                                    srcNodata=0,
                                    resolution='highest')
    gdal.BuildVRT(spk_input_vrt_raw, [spk_input_vv_vrt,
                                  spk_input_vh_vrt], options=vrt_opts)
    print("Generated :{0}".format(spk_input_vrt_raw))

if __name__ == "__main__":
    '''Prepare filtered VRTs'''
    for prefix in prefix_list:
        VH_fn_list = list(map(lambda x: os.path.join(
            spk_out_dir, x.strip(".zip"), prefix+"Sigma0_VH.img"), data_fn_list))
        VV_fn_list = list(map(lambda x: os.path.join(
            spk_out_dir, x.strip(".zip"), prefix+"Sigma0_VV.img"), data_fn_list))
        vrt_opts = gdal.BuildVRTOptions(separate=False,
                                        srcNodata=0,
                                        resolution='highest')
        spk_filt_input_vv_vrt = os.path.join(vrt_dir,prefix+"Sigma0_VV.vrt")
        spk_filt_input_vh_vrt = os.path.join(vrt_dir,prefix+"Sigma0_VH.vrt")
        spk_input_vrt1 = os.path.join(vrt_dir,prefix+"filtered_Sigma0.vrt")

        gdal.BuildVRT(spk_filt_input_vv_vrt,
                      VV_fn_list,
                      options=vrt_opts)
        gdal.BuildVRT(spk_filt_input_vh_vrt,
                      VH_fn_list,
                      options=vrt_opts)
        vrt_opts = gdal.BuildVRTOptions(separate=True,
                                        srcNodata=0,
                                        resolution='highest')
        gdal.BuildVRT(spk_input_vrt1, [spk_filt_input_vv_vrt,
                                      spk_filt_input_vh_vrt], options=vrt_opts)
        print("Generated :{0}".format(spk_input_vrt1))

if __name__ == "__main__":
    '''preparing data vectors'''
    vect_dict_raw = read_vectors(vect_fn=feat_vect_fn,
                                 input_fn=spk_input_vrt_raw,
                                 fields=['id', 'class'],
                                 buffer=0.003
                                 )
    vect_dict_3_1 = read_vectors(vect_fn=feat_vect_fn,
                                 input_fn=spk_input_vrt_3_1,
                                 fields=['id', 'class'],
                                 
                                 )
    vect_dict_3_2 = read_vectors(vect_fn=feat_vect_fn,
                                 input_fn=spk_input_vrt_3_2,
                                 fields=['id', 'class'],
                                 
                                 )
    vect_dict_5_1 = read_vectors(vect_fn=feat_vect_fn,
                                 input_fn=spk_input_vrt_5_1,
                                 fields=['id', 'class'],
                                 buffer=0.003
                                 )
    vect_dict_frost_5 = read_vectors(vect_fn=feat_vect_fn,
                                 input_fn=spk_input_vrt_frost,
                                 fields=['id', 'class'],
                                 
                                 )
    vect_dict_gamma_5 = read_vectors(vect_fn=feat_vect_fn,
                                 input_fn=spk_input_vrt_gamma,
                                 fields=['id', 'class'],
                                 
                                 )
    vect_dict_refined_lee = read_vectors(vect_fn=feat_vect_fn,
                                 input_fn=spk_input_vrt_refined_lee,
                                 fields=['id', 'class'],
                                 
                                 )
if __name__ == "__main__":
    '''preparing search models'''
    prepare_models(vect_dict_raw)
  

if __name__ == "__main__":
    '''preparing segment classifier models'''
    prepare_segment_classifier(vect_dict_3_1, model_fn_3_1)
    prepare_segment_classifier(vect_dict_3_2, model_fn_3_2)
    prepare_segment_classifier(vect_dict_5_1, model_fn_5_1)
    prepare_segment_classifier(vect_dict_raw, model_fn_raw)
    prepare_segment_classifier(vect_dict_frost_5, model_fn_frost)
    prepare_segment_classifier(vect_dict_gamma_5, model_fn_gamma)
    prepare_segment_classifier(vect_dict_refined_lee, model_fn_refined_lee)


def search_job(feat1):
    print(".")
    iter_un_filt = raster_shp_iteraror(spk_input_vrt_raw,spk_aoi_fn)
    model_fn = os.path.join(model_dir, model_fn_temp.format(class_name=feature_mapping[feat1]))
    search_1 = segment_search(model_fn,feat1,speckle_size)
    seg_list = []
    for data, ul_pixel in iter_un_filt:
        seg_list.extend(search_1.search_all(data, ul_pixel))
    seg_list = list(filter(lambda x: x.length()>min_segment_size,seg_list))
    print("completed search: {0}".format(feature_mapping[feat1]))
    return seg_list

if __name__ == "__main__":
    '''load and evaluate search models in parallel'''
    
    feat_list = [5,4,3]
    with Pool(4) as p:
        seg_list_p = p.map(search_job, feat_list)
    seg_list1 =[]
    for l1 in seg_list_p:
        seg_list1.extend(l1)
    print("{0} segments searched".format(len(seg_list1)))

if __name__ == "__main__":
    '''write search results '''
    iter_un_filt = raster_shp_iteraror(spk_input_vrt_raw,spk_aoi_fn)
    if os.path.exists(output_fn1):
        os.remove(output_fn1)
    result_sink1 = raster_sink(raster_fn=output_fn1,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink1.create_file()
    for seg1 in seg_list1:
        result_sink1.write_segment(seg1)
    result_sink1.close()
    
def search_job_filt(feat1):
    print(".")
    iter_un_filt = raster_shp_iteraror(spk_input_vrt_5_1,spk_aoi_fn)
    model_fn = os.path.join(model_dir, model_fn_temp.format(class_name=feature_mapping[feat1]))
    search_1 = segment_search(model_fn,feat1,speckle_size)
    seg_list = []
    for data, ul_pixel in iter_un_filt:
        seg_list.extend(search_1.search_all(data, ul_pixel))
    seg_list = list(filter(lambda x: x.length()>min_segment_size,seg_list))
    print("completed search: {0}".format(feature_mapping[feat1]))
    return seg_list
    
if __name__ == "__main__":
    '''search with pre filtered data'''
    feat_list = [5,4,3]
    with Pool(4) as p:
        seg_list_p = p.map(search_job_filt, feat_list)
    seg_list9 =[]
    for l1 in seg_list_p:
        seg_list9.extend(l1)
    print("{0} segments searched".format(len(seg_list9)))
        
    iter_filt = raster_shp_iteraror(spk_input_vrt_5_1,spk_aoi_fn)
    if os.path.exists(output_fn9):
        os.remove(output_fn9)
    result_sink9 = raster_sink(raster_fn=output_fn9,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink9.create_file()
    for seg1 in seg_list9:
        result_sink9.write_segment(seg1)
    result_sink9.close()
    
    
    
def search_job_filt2(feat1):
    print(".")
    iter_un_filt = raster_shp_iteraror(spk_input_vrt_5_1,spk_aoi_fn)
    model_fn = os.path.join(model_dir, model2_fn_temp.format(class_name=feature_mapping[feat1]))
    search_1 = segment_search(model_fn,feat1,speckle_size)
    seg_list = []
    for data, ul_pixel in iter_un_filt:
        seg_list.extend(search_1.search_all(data, ul_pixel))
    seg_list = list(filter(lambda x: x.length()>min_segment_size,seg_list))
    print("completed search: {0}".format(feature_mapping[feat1]))
    return seg_list
    
if __name__ == "__main__":
    '''search with pre filtered data 2'''    
    '''preparing search models'''
    prepare_models(vect_dict_5_1,
                   fn_temp = model2_fn_temp
                   )
    feat_list = [5,4,3]
    with Pool(4) as p:
        seg_list_p = p.map(search_job_filt2, feat_list)
    seg_list10 = []
    for l1 in seg_list_p:
        seg_list10.extend(l1)
    print("{0} segments searched".format(len(seg_list10)))
        
    iter_filt = raster_shp_iteraror(spk_input_vrt_5_1,spk_aoi_fn)
    if os.path.exists(output_fn10):
        os.remove(output_fn10)
    result_sink10 = raster_sink(raster_fn=output_fn10,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink10.create_file()
    for seg1 in seg_list10:
        result_sink10.write_segment(seg1)
    result_sink10.close()
    
    
    
    
    

def eval_knn_model(model_fn,data,ul_pixel):
    model = pickle.load(open(model_fn, 'rb'))
    data_shape = data.shape
    data_out = model.predict(data.reshape(data_shape[0],-1).T).reshape(data_shape[1:])
    result_classes = np.unique(data_out)
    seg_list = []
    for level in result_classes:
        seg_res = segment_result(ul_pixel=ul_pixel,
                                 data_shape=data_shape[1:],
                                 class_id=level)
        knn_res = search_result()
        knn_res.level = 0
        knn_res.result_list = list(np.argwhere(data_out==level))
        seg_res.level_results.append(knn_res)
        seg_list.append(seg_res)    
    return seg_list


if __name__ == "__main__":
    '''evaluate knn models '''
    iter_un_filt = raster_shp_iteraror(spk_input_vrt_raw,spk_aoi_fn)
    iter_3_1    = raster_shp_iteraror(spk_input_vrt_3_1, spk_aoi_fn)
    iter_3_2    = raster_shp_iteraror(spk_input_vrt_3_2, spk_aoi_fn)
    iter_5_1    = raster_shp_iteraror(spk_input_vrt_5_1, spk_aoi_fn)
    iter_frost    = raster_shp_iteraror(spk_input_vrt_frost, spk_aoi_fn)
    iter_gamma    = raster_shp_iteraror(spk_input_vrt_gamma, spk_aoi_fn)
    iter_refined_lee    = raster_shp_iteraror(spk_input_vrt_refined_lee, spk_aoi_fn)

    seg_list2 =  []
    for data,ul_pixel in iter_3_1:
        seg_list2.extend(eval_knn_model(model_fn_3_1, data, ul_pixel))

    seg_list3 =  []
    for data,ul_pixel in iter_3_2:
        seg_list3.extend(eval_knn_model(model_fn_3_2, data, ul_pixel))

    seg_list4 =  []
    for data,ul_pixel in iter_un_filt:
        seg_list4.extend(eval_knn_model(model_fn_raw, data, ul_pixel))

    seg_list5 =  []
    for data,ul_pixel in iter_5_1:
        seg_list5.extend(eval_knn_model(model_fn_5_1, data, ul_pixel))

    seg_list6 =  []
    for data,ul_pixel in iter_frost:
        seg_list6.extend(eval_knn_model(model_fn_frost, data, ul_pixel))

    seg_list7 =  []
    for data,ul_pixel in iter_gamma:
        seg_list7.extend(eval_knn_model(model_fn_gamma, data, ul_pixel))
        
    seg_list8 =  []
    for data,ul_pixel in iter_refined_lee:
        seg_list8.extend(eval_knn_model(model_fn_refined_lee, data, ul_pixel))


if __name__ == "__main__":
    '''write results of filtered knn'''
    if os.path.exists(output_fn2):
        os.remove(output_fn2)
    result_sink2 = raster_sink(raster_fn=output_fn2,
                              band_size=(iter_3_1.raster.RasterXSize,
                                         iter_3_1.raster.RasterYSize),
                              geotrans=iter_3_1.geotrans,
                              crs=iter_3_1.crs
                              )
    result_sink2.create_file()
    for seg1 in seg_list2:
        result_sink2.write_segment(seg1)
    result_sink2.close()


    if os.path.exists(output_fn3):
        os.remove(output_fn3)
    result_sink3 = raster_sink(raster_fn=output_fn3,
                              band_size=(iter_3_2.raster.RasterXSize,
                                         iter_3_2.raster.RasterYSize),
                              geotrans=iter_3_2.geotrans,
                              crs=iter_3_2.crs
                              )
    result_sink3.create_file()
    for seg1 in seg_list3:
        result_sink3.write_segment(seg1)
    result_sink3.close()


    if os.path.exists(output_fn4):
        os.remove(output_fn4)
    result_sink4 = raster_sink(raster_fn=output_fn4,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink4.create_file()
    for seg1 in seg_list4:
        result_sink4.write_segment(seg1)
    result_sink4.close()


    if os.path.exists(output_fn5):
        os.remove(output_fn5)
    result_sink5 = raster_sink(raster_fn=output_fn5,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink5.create_file()
    for seg1 in seg_list5:
        result_sink5.write_segment(seg1)
    result_sink5.close()

    if os.path.exists(output_fn6):
        os.remove(output_fn6)
    result_sink6 = raster_sink(raster_fn=output_fn6,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink6.create_file()
    for seg1 in seg_list6:
        result_sink6.write_segment(seg1)
    result_sink6.close()

    if os.path.exists(output_fn7):
        os.remove(output_fn7)
    result_sink7 = raster_sink(raster_fn=output_fn7,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink7.create_file()
    for seg1 in seg_list7:
        result_sink7.write_segment(seg1)
    result_sink7.close()
    
    if os.path.exists(output_fn8):
        os.remove(output_fn8)
    result_sink8 = raster_sink(raster_fn=output_fn8,
                              band_size=(iter_un_filt.raster.RasterXSize,
                                         iter_un_filt.raster.RasterYSize),
                              geotrans=iter_un_filt.geotrans,
                              crs=iter_un_filt.crs
                              )
    result_sink8.create_file()
    for seg1 in seg_list8:
        result_sink8.write_segment(seg1)
    result_sink8.close()


#%% validation
def plot_confusion(val_vect,
                   plot_fn,
                   ticks=[0.5,1.5,2.5,3.5],
                   labels=["None","crop1","crop2","crop3"],
                   label_vals = [4,5,3]):
    predicted= []
    expected =[]
    for feat1 in val_vect:
        vect_list = val_vect[feat1]['vectors']
        predicted.extend(vect_list)
        expected.extend(list(np.ones(len(vect_list))*feat1))
    conf_mat = confusion_matrix(expected,
                                predicted,
                                normalize='true',
                                )
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_mat,annot=True,cmap='Blues')
    plt.xticks(ticks=ticks,labels=labels,rotation=0)
    plt.yticks(ticks=ticks,labels=labels,rotation=0)
    plot_fn_full = os.path.join(output_dir,"plots",plot_fn)
    plt.savefig(plot_fn_full,dpi=300,bbox_inches='tight')
    precision = precision_score(expected, 
                                predicted,
                                average='macro',
                                labels=label_vals)
    recall = recall_score(expected, 
                          predicted,
                          average='macro',
                          labels=label_vals)
    f1score = f1_score(expected, 
                       predicted,
                       average='macro',
                       labels=label_vals)
    accuracy = accuracy_score(expected, predicted)
    return precision,recall,f1score,accuracy


if __name__ == "__main__":
    '''evaluate accuracy and confusion matrix'''

    val_vect1 = read_vectors(val_ref_fn,output_fn1)
    plot_fn1 = "confusion_matrix_search.png"
    score1 = plot_confusion(val_vect1,plot_fn1)

    val_vect2 = read_vectors(val_ref_fn,output_fn2)
    plot_fn2 = "confusion_matrix_knn_3_1.png"
    score2 = plot_confusion(val_vect2,
                   plot_fn2,
                   ticks=[0.5,1.5,2.5],
                   labels=["crop1","crop2","crop3"])

    val_vect3 = read_vectors(val_ref_fn,output_fn3)
    plot_fn3 = "confusion_matrix_knn_3_2.png"
    score3 = plot_confusion(val_vect3,
                   plot_fn3,
                   ticks=[0.5,1.5,2.5],
                   labels=["crop1","crop2","crop3"])

    val_vect4 = read_vectors(val_ref_fn,output_fn4)
    plot_fn4 = "confusion_matrix_knn_raw.png"
    score4 = plot_confusion(val_vect4,
                   plot_fn4,
                   ticks=[0.5,1.5,2.5],
                   labels=["crop1","crop2","crop3"])

    val_vect5 = read_vectors(val_ref_fn,output_fn5)
    plot_fn5 = "confusion_matrix_knn_5_1.png"
    score5 = plot_confusion(val_vect5,
                   plot_fn5,
                   ticks=[0.5,1.5,2.5],
                   labels=["crop1","crop2","crop3"])

    val_vect6 = read_vectors(val_ref_fn,output_fn6)
    plot_fn6 = "confusion_matrix_frost_5.png"
    score6 = plot_confusion(val_vect6,
                   plot_fn6,
                   ticks=[0.5,1.5,2.5],
                   labels=["crop1","crop2","crop3"])

    val_vect7 = read_vectors(val_ref_fn,output_fn7)
    plot_fn7 = "confusion_matrix_knn_gamma.png"
    score7 = plot_confusion(val_vect7,
                   plot_fn7,
                   ticks=[0.5,1.5,2.5],
                   labels=["crop1","crop2","crop3"])
    
    val_vect8 = read_vectors(val_ref_fn,output_fn8)
    plot_fn8 = "confusion_matrix_knn_refine_lee.png"
    score8 = plot_confusion(val_vect8,
                   plot_fn8,
                   ticks=[0.5,1.5,2.5],
                   labels=["crop1","crop2","crop3"])
    
    val_vect9 = read_vectors(val_ref_fn,output_fn9)
    plot_fn9 = "confusion_matrix_pre_filt_search.png"
    score9 = plot_confusion(val_vect9,
                   plot_fn9,
                   )
    
    val_vect10 = read_vectors(val_ref_fn,output_fn10)
    plot_fn10 = "confusion_matrix_pre_filt2_search.png"
    score10 = plot_confusion(val_vect10,
                   plot_fn10,
                   )

if __name__ == "__main__":
    table_fn = "Accuracy.tex"
    table_file = open(table_fn, 'w')
    algorithm_names = ["SBIA",
                       "Without filter",
                       "Lee filter (3x3) ",
                       "Lee filter (5x5) ",
                       "Frost filter (5x5 ) ",
                       "Gamma-MAP filter (5x5 )",
                       "Refined Lee",
                       "SBIA with Lee filter"
                       ]
    score_list = [score1,
                  score4,
                  score2,
                  score5,
                  score6,
                  score7,
                  score8,
                  score10
                  ]
    num_col = 8
    measure_list = ["Algorithm","Precision", "Recall", "F1 Score", "Accuracy"]
    table_file.write("\\begin{tabular}{" + "| c "*num_col + "|} \\hline \n ")
    table_file.write(" & ".join(measure_list)+"\\\\ \\hline \n ")
    for idx1 in range(8):
        row_list =[]
        row_list.append(algorithm_names[idx1])
        for idx2 in range(4):
            row_list.append(" {0:0.3} ".format(score_list[idx1][idx2]))
        table_file.write(" & ".join(row_list))
        table_file.write("\\\\ \n")

    table_file.write("\\hline \n\\end{tabular}\n")
    table_file.flush()
    table_file.close()


    
