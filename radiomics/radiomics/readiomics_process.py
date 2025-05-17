import os

import numpy as np
import pandas as pd

import SimpleITK as sitk
from skimage import measure
import six
import scipy

from radiomics_utils import lungmask
from radiomics import featureextractor, getTestCase

def feature_extraction(file_name):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_name)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_name, series_IDs[0])
    # print(series_file_names)
    # print(file)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D_raw = series_reader.Execute()
    label, mask = lungmask(image3D_raw)
    extractor = featureextractor.RadiomicsFeatureExtractor("./settings.yaml")
    result = extractor.execute(image3D_raw, label, label=1)
    feature_cur = []
    feature_name = []
    # result = extractor.execute(imagePath, maskPath, label=255)
    for key, value in six.iteritems(result):
        # print('\t', key, ':', value)
        feature_name.append(key)
        feature_cur.append(value)

    name = feature_name[37:]
    name = np.array(name)
    return feature_cur[37:],name

id_name = 0
ids = []
dataset = "COPD"
save_file = np.empty(shape=[1,1302])
for dir,subname, name_list in os.walk("../../data/before/"+dataset):
# for dir,subname, name_list in os.walk("./"+dataset):
    if len(name_list) > 128:
        try:
            feature, name = feature_extraction(dir)
            feature = np.array(feature)
            feature = feature.reshape([1,-1])
            id = dataset+"_"+dir.split("/")[4]+"_"+str(id_name)
            print(id)
            ids.append(id)
            id_name = id_name + 1
            save_file = np.append(save_file,feature,axis=0)
        except Exception as e:
            print(e)

save_file = np.delete(save_file,0,0)
#save_file = save_file.transpose()
#print(save_file.shape)
id_num = len(ids)
ids = np.array(ids)
name_df = pd.DataFrame(save_file)
name_df.index = ids
name_df.columns = name
# writer = pd.ExcelWriter(dataset+'_Radiomics-features.xlsx')
name_df.to_csv(dataset+'_Radiomics-features.csv')
