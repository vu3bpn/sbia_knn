#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:40:20 2021

@author: Bipin Chandran 
"""

data_base_dir = "SPK_work"
fast_data_dir = r"fast_data/SBIA"
result_dir = os.path.join(data_base_dir,'results')
data_dir = r"uniform_regions"
dem_file = r"DEM/srtm_1arcsec_merged.tif"
processed_data_dir = os.path.join(data_base_dir, "TC_data")
gpt_cmd = "/home/bipin/snap/bin/gpt"
preprocess_logfile = "Preprocessing_log.txt"


input_xml_fn = 'CAL_TC_auto.xml'
output_xml_fn = 'CAL_TC_out.xml'


input_filter_xml_fn = 'Speckle_filter.xml'
output_filter_xml_fn = 'Speckle_filter_out.xml'


data_fn_list = [
                "S1A_IW_GRDH_1SDV_20210531T235709_20210531T235734_038139_048057_E349.zip",
                ]
vrt_dir = os.path.join(data_base_dir, "VRT_files")


# %% Model preparation
vect_dir = os.path.join(data_base_dir, "Vectors")
labeled_vect_fn = os.path.join(vect_dir,"FEAT_VECT4.shp")


# %% SBIA configurations
output_dir = os.path.join(fast_data_dir, "Output")
model_dir = os.path.join(fast_data_dir, "models")
experiment_id = 1
experiment_name = f"Exp{experiment_id}"

feature_mapping =   {1: "Water",
                      2: "Urban",
                      3: "Crop1",
                      4: "Crop2",
                      5: "Crop3",
                      6: "Barren",
                      7: "Trees",
                      8: "Crop6",
                      9: "Crop7"
                      }

input_vrt_fn = os.path.join(
    vrt_dir, "Input_vrt_{0}.vrt".format(experiment_name))
filtered_vrt_fn = os.path.join(
    vrt_dir, "Filtered_vrt_{0}.vrt".format(experiment_name))
n_neighbours_bulk = 200
n_neighbours_segment = 15
n_neighbours_level = 15
max_vectors_per_class = 1500
levels_population_ratio = [1,2,4]

population_ratio_dict = {1: [1, 10, 1],
                         2: [1, 8, 5],
                         3: [1, 8, 8],
                         4: [1, 8, 8],
                         5: [1, 8, 8],
                         6: [1, 4, 6],
                         7: [1, 8, 5],
                         8: [1, 8, 5],
                         9: [1, 8, 5],
                         }


model_fn_temp = "Level_model_{class_name}.model"
model2_fn_temp = "Filt_Level_model_{class_name}.model"
speckle_size = 200
min_segment_size = 200
output_fn = os.path.join(
    output_dir, "Segmented_results_{0}.tif".format(experiment_name))

# %%validation configs
validation_aoi_fn = os.path.join(data_base_dir, "Vectors/AOI3.shp")
validation_data_dir = os.path.join(data_base_dir, "Drive_download", "ee_Data")
validation_train_data_dir = os.path.join(
    data_base_dir, "Drive_download", "ee_data_train")


validation_train_data_vrt_fn = input_vrt_fn
validation_data_vrt_fn = input_vrt_fn
validation_ref_vrt_fn = os.path.join(vrt_dir, "Validation_ref_data.vrt")

sentinel_prefix = "Sentinel_aoi"
ref_prefix = "Land_cover_aoi"
validation_model_fn_temp = "Level_model_fold_{fold}_{class_name}.model"
validation_ref_class_mapping = {80: 1, 50: 2, 10: 6}
classifier_model_fn = "segment_classifier.model"


# % Speckle filter
spk_input_vrt = os.path.join(vrt_dir, "spk_input.vrt")
spk_input_vv_vrt = os.path.join(vrt_dir, "spk_input_VV.vrt")
spk_input_vh_vrt = os.path.join(vrt_dir, "spk_input_VH.vrt")
spk_out_dir = os.path.join(data_base_dir,"Spk_filtered")

tc_out_dir = processed_data_dir
vector_fn_temp = "Feat_vect_{0}_{1}.pickle"
results_list_fn = os.path.join(result_dir,"grid_search.json")
results_table_fn = "accuracy_table.tex"
