#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:13:46 2021

@author: Bipin Chandran 
"""

import matplotlib.pyplot as plt
import random
import pickle
import seaborn as sns
from data_conf import *
import pandas
random.seed(3)


def plot_vectors(model_fn_temp,
                 plot_fn):
    feat_names = ['Crop1','Crop2','Crop3']
    frames = []
    index1 = 0
    for feat1 in feat_names:
        model_fn = model_fn_temp.format(feat=feat1)
        vect_fn = os.path.join(data_base_dir,
                               'debug',
                               vector_fn_temp.format(model_fn,experiment_id)
                               )
        
        model_vectors = pickle.load(open(vect_fn,'rb'))
        vv,vh = zip(*model_vectors['vect'])
        data_dict = {"VV" : vv, "VH":vh , "Label": model_vectors['labels']}
        df1 = pandas.DataFrame(data_dict,index=range(index1,index1+len(vv)))
        index1 = index1+len(vv)
        df1["Feature"] = feat1
        frames.append(df1)
    vect_df = pandas.concat(frames)
    vect_df = vect_df[vect_df['VV'] < 0.4]
    vect_df = vect_df[vect_df['VH'] < 0.1]
    vect_df = vect_df.sample(n=500,random_state=3)
    vect_df['Label'] = vect_df['Label']+1
    vect_df = vect_df.sort_values('Label',ascending=False)
    vect_df['Label'] = vect_df['Label'].apply(int)
    vect_df['Label'] = vect_df['Label'].apply(str)
    vect_df['Label'] = "$l_"+vect_df['Label']+"$"
    vect_df['VV $(\sigma_0)$'] = vect_df['VV']
    vect_df['VH $(\sigma_0)$'] = vect_df['VH']
    
    speckle_palette = {'$l_1$':'blue','$l_2$':'green','$l_3$':'yellow'}
    sns.set(rc={'figure.figsize':(6,6),
                'figure.facecolor':'white',
                'axes.facecolor':'white',
                'grid.color':'0.8'}
            )
    scat = sns.scatterplot(data=vect_df,
                    x='VV $(\sigma_0)$',
                    y='VH $(\sigma_0)$', 
                    hue='Label',
                    style='Feature',
                    alpha=0.75,
                    palette=speckle_palette
                    )
    plt.savefig(plot_fn,dpi=500,bbox_inches='tight')
    plt.figure()
  
if __name__ == "__main__":
    model_fn_list = ["Filt_Level_model_{feat}.model",
                     "Level_model_{feat}.model"
                     ]
    plot_fn_list = [os.path.join(output_dir,"plots","Vector_distribution_pre_filt.png"),
                    os.path.join(output_dir,"plots","Vector_distribution.png")
                    ]
    for model_fn,plot_fn in zip(model_fn_list,plot_fn_list):
        plot_vectors(model_fn,plot_fn)

if __name__ == "__main1__":
    vect_dict = read_vectors(vect_fn=labeled_vect_fn,
                             input_fn=validation_train_data_vrt_fn)

    for feat_id in vect_dict:
        feature_name = feature_mapping[feat_id]
        model_fn = model_fn_temp.format(class_name=feature_name)
        model_fn = os.path.join(model_dir,model_fn)
        model = pickle.load(open(model_fn,'rb'))

        vect1 = vect_dict[feat_id]['vectors']
        vect1 = random.sample(vect1, k=1000)
        levels = model.predict(vect1)

        x,y = zip(*vect1)
        plt.figure(feature_name)
        plt.scatter(x,y,c=levels,alpha=0.5,marker="+")
        plt.colorbar()


if __name__ == "__main1__":
    feat_vect_list = get_vect_list()
    for fold in range(len(feat_vect_list)):
        for feat_id in feat_vect_list[fold]:
            feature_name = feature_mapping[feat_id]
            '''
            model_fn = validation_model_fn_temp.format_map({'fold':fold,
                                                            'class_name':feature_name})
            '''
            model_fn = model_fn_temp.format(class_name=feature_name)
            model_fn = os.path.join(model_dir,model_fn)
            model = pickle.load(open(model_fn,'rb'))

            vect1 = feat_vect_list[fold][feat_id]['vectors']
            vect1 = random.sample(vect1, k=1000)
            levels = model.predict(vect1)

            x,y = zip(*vect1)
            plt.figure()
            plt.scatter(x,y,c=levels,alpha=0.5)
            plt.colorbar()


if __name__ == "__main1__":
    '''plot feature vectors'''
    num_vect = 500
    vect_dict = read_vectors()
    for feat1 in vect_dict:
        vect1 = list(vect_dict[feat1]['vectors'])
        if len(vect1) > num_vect:
            vect1 = random.sample(vect1, k=num_vect)
        std_vect = np.std(np.array(vect1), axis=0)
        mean_vect = np.mean(np.array(vect1), axis=0)
        max_vect = np.max(np.array(vect1), axis=0)

        noise_gain = max_vect
        noise_offset = mean_vect

        n_vect, vect_len = np.array(vect1).shape
        n_noise_vect = n_vect

        noise_vect = np.random.random((n_noise_vect, vect_len))
        noise_vect = noise_vect * \
            np.repeat(np.expand_dims(noise_gain, 0), n_noise_vect, axis=0)

        x, y = zip(*vect1)
        x_n, y_n = zip(*list(noise_vect))

        model_fn = model_fn_temp .format(class_name=feature_mapping[feat1])
        level_model = pickle.load(
            open(os.path.join(model_dir, model_fn), 'rb'))
        levels = level_model.predict(vect1)

        plt.figure()
        plt.scatter(x, y, c=levels, marker='+', alpha=0.5)
        plt.title(feature_mapping[feat1])
        plt.colorbar

if __name__ == "__main1__":
    model_fn1 = os.path.join(
        model_dir, model_fn_temp.format(class_name='Water'))
    aoi_fn = r"SBIA/data/Vectors/AOI2.shp"
    iter1 = raster_shp_iteraror(input_vrt_fn, aoi_fn)
    search_list = []
    for data1 in iter1:
        plt.figure()
        plt.imshow(data1[0, :, :], vmax=0.1)
        search1 = multi_level_search(model_fn=model_fn1,
                                     feature_name="Water",
                                     level3_size=speckle_size)
        search1.search_All(data1)
        search_list.append(search1)

if __name__ == "__main1__":
    for result1 in search_list:
        plt.figure()
        plt.imshow(result1.result_mat, interpolation='nearest')
        plt.colorbar()
        plt.figure()
        plt.imshow(result1.level_search.mask, interpolation='nearest')


if __name__ == "__main1__":
    for result1 in search_list:
        for segment_result in result1.segments:
            for res1 in segment_result:
                if len(res1.result_list) > 0:
                    print(res1.level)

if __name__ == "__main1__":
    model_fn = os.path.join(
        model_dir, model_fn_temp.format(class_name='Water'))
    aoi_fn = r"SBIA/data/Vectors/AOI2.shp"
    iter1 = raster_shp_iteraror(input_vrt_fn, aoi_fn)
    search_list = []

    for data in iter1:
        model = pickle.load(open(model_fn, 'rb'))
        level_search1 = search_level(model)
        classes = model.predict(data.reshape(
            (data.shape[0], -1)).T).reshape(data.shape[1:])
        level_0_pts = list(np.argwhere(classes == 0))

        plt.figure('data')
        plt.imshow(data[0, :, :], vmax=0.1)
        plt.figure('classes')
        plt.imshow(classes)

        p1 = level_0_pts.pop(1)
        level_results = []
        working_list = []
        segment = segment()
        level_search1.start_search(data, p1)
        if level_search1.result.level >= 0:
            working_list.extend(level_search1.result.neighbours_list)
            print(p1, level_search1.result.level, len(
                level_search1.result.result_list), len(level_search1.result.neighbours_list))
            segment += level_search1.result
        cnt = 0
        plt.scatter(p1[0], p1[1], marker='+')
        level_search1.result = search_result()

        while len(working_list) > 0 and cnt < 1000:
            p1 = working_list.pop(0)
            level_search1.search(p1)
            if level_search1.result.level >= 0:
                working_list.extend(level_search1.result.neighbours_list)
                print(p1, level_search1.result.level, len(
                    level_search1.result.result_list), len(level_search1.result.neighbours_list))
                segment += level_search1.result
            cnt += 1
            plt.scatter(
                p1[0], p1[1], c=level_search1.result.level+3, marker='+')
            level_search1.result = search_result()

        plt.figure('mask')
        plt.imshow(level_search1.mask)
        for seg1 in level_results:
            print(seg1.level, len(seg1.result_list))
