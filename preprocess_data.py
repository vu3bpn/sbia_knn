#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:26:31 2021

@author: Bipin Chandran 
"""

import xml.etree.ElementTree as et
import os
import time
from data_conf import *
from osgeo import gdal

def speckle_filter_otb(input_dir,
                       output_dir,
                       filter_name = 'lee',
                       rad = 3,
                       looks = 2,
                       prefix= "lee_3_2_"
                       ):
    fname_list = ["Sigma0_VH.img","Sigma0_VV.img"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for fn1 in fname_list:
        out_fn = os.path.join(output_dir,prefix+fn1)
        in_fn = os.path.join(input_dir,fn1)
        if os.path.exists(in_fn):
            if not os.path.exists(out_fn):
                if filter_name == 'frost':
                    cmd1 = "otbcli_Despeckle -in {infn} -filter {filt} -filter.{filt}.rad {rad} -out {outfn}".format(infn=in_fn,filt=filter_name,rad=rad,looks=looks,outfn=out_fn)
                else:
                    cmd1 = "otbcli_Despeckle -in {infn} -filter {filt}  -filter.{filt}.rad {rad} -filter.{filt}.nblooks {looks} -out {outfn}".format(infn=in_fn,filt=filter_name,rad=rad,looks=looks,outfn=out_fn)
                print(cmd1)
                os.system(cmd1)
            else:
                print("Skipping file :{0}".format(out_fn))
        else:
            print("Input file not present :{0}".format(in_fn))



def speckle_filter_data(input_fname,
                        input_base_dir,
                        output_dir,
                        out_prefix="Spk"):
    properties = et.parse(input_filter_xml_fn)
    f1 = properties.find("node[@id='Read']/parameters/file")
    f2 = properties.find("node[@id='Write']/parameters/file")
    f3 = properties.find("node[@id='Speckle-Filter']/parameters/sourceBands")
    f1.text = os.path.join(input_base_dir, input_fname)
    f2.text = os.path.join(output_dir, out_prefix)
    f3.text = input_fname.strip(".img")
    properties.write(output_filter_xml_fn)
    command1 = "{1}  {0}  -J-Xms512M -J-Xmx11G -x -c 1G -q 6 -e ".format(
        output_filter_xml_fn, gpt_cmd)
    if os.path.exists(os.path.join(f2.text, input_fname)):
        print("Skipping Processed file {0}".format(f1.text))
    else:
        print("Filtering : {0} ".format(input_fname))
        os.system(command1)


def prepare_data(
    input_fname,
    input_base_dir,
    output_dir,
    dem_file
):
    '''Preprocess the input data using SNAP GPT'''
    t1 = time.time()
    properties = et.parse(input_xml_fn)
    nodes = properties.findall('node')
    f1 = properties.find("node[@id='Read']/parameters/file")
    f1.text = os.path.join(input_base_dir, input_fname)
    f2 = properties.find("node[@id='Write']/parameters/file")
    f2.text = os.path.join(output_dir, input_fname.strip(".zip"))
    proj = properties.find(
        "node[@id='Terrain-Correction']/parameters/mapProjection")
    proj.text = proj.text.replace('\r\n', ' ')
   
    properties.write(output_xml_fn)
    command1 = "{1}  {0}  -J-Xms512M -J-Xmx11G -x -c 1G -q 6 -e ".format(
        output_xml_fn, gpt_cmd)
    if os.path.exists(f2.text):
        print("Skipping Processed file {0}".format(f1.text))
    else:
        print("Processing : {0} \nTime :{1}".format(
            input_fname, time.ctime(t1)))
        os.system(command1)
        t2 = time.time()
        open(preprocess_logfile, 'a').write("t0 :{0} \tt1 :{1} \tEpoh : {2} \tdata :{3}\n".format(
            time.ctime(t1), time.ctime(t2), t2-t1, input_fname))


def build_vrt_filtered_sigma(vrt_fn,
                             data_fn_list,
                             out_prefix="Spk"):
    tiff_list_full = []
    for fn1 in data_fn_list:
        fn1_full = os.path.join(processed_data_dir, fn1.strip(
            ".zip"), out_prefix, "Sigma0_VH.img")
        fn2_full = os.path.join(processed_data_dir, fn1.strip(
            ".zip"), out_prefix, "Sigma0_VV.img")
        tiff_list_full.append(fn1_full)
        tiff_list_full.append(fn2_full)
    vrt_opts = gdal.BuildVRTOptions(separate=True,
                                    resolution='highest')
    gdal.BuildVRT(vrt_fn, tiff_list_full, options=vrt_opts)
    print("Generated {0}".format(vrt_fn))



def build_vrt_sigma(vrt_fn,
                    data_fn_list,
                    data_dir = output_dir):
    tiff_list_full = []
    for fn1 in data_fn_list:
        fn1_full = os.path.join(
            data_dir, fn1.strip(".zip"), "Sigma0_VH.img")
        fn2_full = os.path.join(
            data_dir, fn1.strip(".zip"), "Sigma0_VV.img")
        tiff_list_full.append(fn1_full)
        tiff_list_full.append(fn2_full)
    vrt_opts = gdal.BuildVRTOptions(separate=True,
                                    resolution='highest')
    gdal.BuildVRT(vrt_fn, tiff_list_full, options=vrt_opts)
    print("Generated {0}".format(vrt_fn))


if __name__ == "__main1__":
    properties = et.parse(input_filter_xml_fn)

if __name__ == "__main1__":
    '''prepare data'''
    input_base_dir = data_dir
    output_dir = tc_out_dir
    for input_fname in data_fn_list:
        prepare_data(input_fname, input_base_dir, output_dir, dem_file)

if __name__ == "__main__":
    '''filter data'''
    for data_fn  in data_fn_list:
        input_dir = os.path.join(tc_out_dir,data_fn.strip(".zip"))
        output_dir = os.path.join(spk_out_dir,data_fn.strip(".zip"))
        speckle_filter_otb(input_dir,
                           output_dir,
                           rad = 3,
                           looks = 2,
                           prefix= "lee_3_2_")
        speckle_filter_otb(input_dir,
                           output_dir,
                           rad = 5,
                           looks = 1,
                           prefix= "lee_5_1_")
        speckle_filter_otb(input_dir,
                           output_dir,
                           rad = 5,
                           looks = 1,
                           prefix= "frost_5_1_",
                           filter_name = 'frost',)
        speckle_filter_otb(input_dir,
                           output_dir,
                           rad = 5,
                           looks = 1,
                           prefix= "gamma_5_1_",
                           filter_name = 'gammamap',)


if __name__ == "__main1__" :
    '''Speckle filtering'''
    band_fnames = ["Sigma0_VH.img", "Sigma0_VV.img"]
    for input_fname in data_fn_list:
        input_base_dir = os.path.join(
            tc_out_dir, input_fname.strip(".zip"))
        output_dir = os.path.join(spk_out_dir,input_fname.strip(".zip"))
        for band_name1 in band_fnames:
            speckle_filter_data(band_name1, input_base_dir, output_dir)

if __name__ == "__main__":
    '''build vrt'''
    build_vrt_sigma(input_vrt_fn,
                    data_fn_list,
                    data_dir =tc_out_dir)
    #build_vrt_filtered_sigma(filtered_vrt_fn, data_fn_list)
