import os

import numpy as np
import pandas as pd

import SimpleITK as sitk
from skimage import measure
import six
import scipy

import matplotlib.pyplot as plt
import matplotlib

from radiomics_utils import lungmask
from radiomics import featureextractor, getTestCase

ids = []
dataset = "20230928COPD"
# save_file = np.empty(shape=[1,1302])
id_name = 0
for file_name,subname, name_list in os.walk("../../data/before/"+dataset):
# for dir,subname, name_list in os.walk("./test_data/"+dataset):
    if len(name_list) > 128:
        try:    
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_name)
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_name, series_IDs[0])
            # print(series_file_names)
            # print(file)
            print(file_name)
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_file_names)
            image3D_raw = series_reader.Execute()
            label, mask = lungmask(image3D_raw)
            image3D_raw = sitk.Cast(sitk.RescaleIntensity(image3D_raw,), sitk.sitkUInt8)
            overlay = sitk.LabelOverlay(image3D_raw, label)
            nda = sitk.GetArrayViewFromImage(overlay)
            id_name = id_name + 1
            id = dataset+"_"+file_name.split("/")[4]+"_"+str(id_name)
            print(id)
            
            matplotlib.image.imsave("./mask_overlay/"+ id+'.pdf', nda[100])
            
        except Exception as e:
            print(e)
