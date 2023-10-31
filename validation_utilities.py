#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 09:59:33 2022

@author: Bipin Chandran 
"""
import numpy as np
from data_conf import *
from data_utilities import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

if __name__ == "__main1__":
    clip_data_ee_sentinel()

if __name__ == "__main__":
    #ref_fn = validation_ref_vrt_fn
    ref_fn = os.path.join(validation_data_dir, "Land_cover_region_1.tif")
    result_fn = output_fn
    aoi_fn = validation_aoi_fn
    selected_feat = [1, 2]
    accuracy_dict_full = {}
    result_iter = raster_shp_iteraror(result_fn, aoi_fn)
    ref_iter = raster_shp_iteraror(ref_fn, aoi_fn)
    aoi_idx = 0
    for res_data_ul, ref_data_ul in zip(result_iter, ref_iter):
        aoi_idx += 1
        print("AOI :{0}".format(aoi_idx))
        res_data, res_ul = res_data_ul
        ref_data, ref_ul = ref_data_ul
        ref_data = ref_data.astype(float)
        res_data = res_data.astype(float)
        if res_data.shape != ref_data.shape:
            data_shape = min(res_data.shape, ref_data.shape)
            res_data = res_data[:data_shape[0], :data_shape[1], :data_shape[2]]
            ref_data = ref_data[:data_shape[0], :data_shape[1], :data_shape[2]]

        ref_data_remap = np.zeros_like(ref_data)
        for ref_class in validation_ref_class_mapping:
            feat_class = validation_ref_class_mapping[ref_class]
            ref_data_remap += (ref_data == ref_class)*feat_class

        selected_res = np.zeros_like(res_data)
        selected_ref = np.zeros_like(ref_data)

        for feat in selected_feat:
            selected_res += (res_data == feat)*feat
            selected_ref += (ref_data_remap == feat)*feat

        accuracy_dict = {}
        tot_accuracy_list = []
        aoi_TP = 0
        aoi_FP = 0
        aoi_FN = 0
        for feat in selected_feat:
            y_true = (selected_ref == feat)
            y_pred = (selected_res == feat)

            true_total = (selected_ref == feat) == (selected_res == feat)
            true_pos = (selected_ref == feat) * (selected_res == feat)
            false_pos = (selected_res == feat) * (selected_ref != feat)
            false_neg = (selected_res != feat) * (selected_ref == feat)

            total_size = len(true_pos.flatten())
            true_pos = sum(true_pos.flatten())
            false_pos = sum(false_pos.flatten())
            false_neg = sum(false_neg.flatten())

            aoi_TP += true_pos
            aoi_FP += false_pos
            aoi_FN += false_neg

            precision = 0.0
            recall = 0.0
            f1 = 0.0
            accuracy = 0.0

            if true_pos > 0:
                precision = true_pos/(true_pos+false_pos)
                recall = true_pos/(true_pos+false_neg)
                f1 = 2*precision*recall/(precision+recall)
                accuracy = sum(true_total.flatten())/total_size

            tot_accuracy_list.append(accuracy)
            accuracy_dict[feature_mapping[feat]] = {"precision": precision,
                                                    "recall": recall,
                                                    "f1": f1,
                                                    "accuracy": accuracy}
            print("{0} \t: {1:.3} \t{2:.3} \t{3:.3} \t{4:.3}".format(
                feature_mapping[feat], precision, recall, f1, accuracy))

        aoi_precision = 0.0
        aoi_recall = 0.0
        aoi_f1 = 0.0
        aoi_accuracy = 0.0

        if aoi_TP > 0:
            aoi_precision = aoi_TP/(aoi_TP+aoi_FP)
            aoi_recall = aoi_TP/(aoi_TP+aoi_FN)
            aoi_f1 = 2*aoi_precision*aoi_recall/(aoi_precision+aoi_recall)
            aoi_accuracy = np.mean(tot_accuracy_list)

        accuracy_dict['Total'] = {"precision": aoi_precision,
                                  "recall": aoi_recall,
                                  "f1": aoi_f1,
                                  "accuracy": aoi_accuracy}
        accuracy_dict_full[aoi_idx] = accuracy_dict

        feature_cmap = ListedColormap(
            ['white', 'blue', 'red', 'yellow', 'lime', 'green', 'darkgreen', 'cyan'])
        plt.figure()
        plt.subplot(121)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.title('Predicted')
        plt.imshow(selected_res.squeeze(), cmap=feature_cmap, vmax=6)
        plt.subplot(122)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(selected_ref.squeeze(), cmap=feature_cmap, vmax=6)
        plt.title("Reference")
        plt.savefig(os.path.join(output_dir, 'plots',
                    "Predicted_res_AOI_{0}.tif".format(aoi_idx)), dpi=300)

if __name__ == "__main__":
    table_fn = "Accuracy.tex"
    table_file = open(table_fn, 'w')
    num_col = 6
    aoi_list = [8, 9, 10]
    heading_list = ["precision", "recall", "f1", "accuracy"]
    feat_name_list = ['Water', 'Urban', "Total"]
    table_file.write("\\begin{tabular}{" + "c "*num_col + "}\n")
    table_file .write(" & ".join(
        [" AOI ", " feature "]+heading_list) + "\\\\\n")

    for aoi in aoi_list:
        table_file .write(" \\multirow{3}*{"+" {0} ".format(aoi)+"}")
        for feat1 in feat_name_list:
            val_list = [feat1]
            for head1 in heading_list:
                val_list.append("{0:0.3}".format(
                    accuracy_dict_full[aoi][feat1][head1]))
            table_file .write("\n & " + " \t& ".join(val_list)+"\\\\\n")

    table_file.write("\n\\end{tabular}\n")
    table_file.flush()
    table_file.close()
