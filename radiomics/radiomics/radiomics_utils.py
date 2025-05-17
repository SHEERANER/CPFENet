import os

import numpy as np
import pandas as pd

import SimpleITK as sitk
from skimage import measure
import scipy
import six

from radiomics import featureextractor, getTestCase

def lungmask(vol):
    size = sitk.Image(vol).GetSize()
    spacing = sitk.Image(vol).GetSpacing()
    #将体数据转为numpy数组
    volarray = sitk.GetArrayFromImage(vol)

    #根据CT值，将数据二值化（一般来说-300以下是空气的CT值）
    volarray[volarray>=-300]=1
    volarray[volarray<=- 300]=0
    #生成阈值图像
    threshold = sitk.GetImageFromArray(volarray)
    threshold.SetSpacing(spacing)

    #利用种子生成算法，填充空气
    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList([(0,0,0),(size[0]-1,size[1]-1,0)])
    
    #得到body的mask，此时body部分是0，所以反转一下
    bodymask = ConnectedThresholdImageFilter.Execute(threshold)
    bodymask = sitk.ShiftScale(bodymask,-1,-1)
    
    #用bodymask减去threshold，得到初步的lung的mask
    temp = sitk.GetImageFromArray(sitk.GetArrayFromImage(bodymask)-sitk.GetArrayFromImage(threshold))
    temp.SetSpacing(spacing)
    #利用形态学来去掉一定的肺部的小区域
    
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(2)
    bm.SetForegroundValue(1)
    lungmask = bm.Execute(temp)
    
    #利用measure来计算连通域
    lungmaskarray = sitk.GetArrayFromImage(lungmask)
    label = measure.label(lungmaskarray,connectivity=2)
    props = measure.regionprops(label)

    #计算每个连通域的体素的个数
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]

    #最大连通域的体素个数，也就是肺部
    maxnum = max(numPix)

    for i in range(len(numPix)):
    #如果当前连通区域不是最大值所在的区域，则当前区域的值全部置为0，否则为1
        if numPix[i]==maxnum:
            max_index = i
            break   
    
    label_mask = (label==max_index+1).astype("int16")
    mask1 = scipy.ndimage.binary_erosion(label_mask, structure=np.ones((4,4,4)))
    mask2 = scipy.ndimage.binary_dilation(mask1, structure=np.ones((10,10,10))).astype("int16")
    # label_mask = label_mask.astype("int16")
    l = sitk.GetImageFromArray(mask2)
    l.SetSpacing(spacing)
    l.SetOrigin(vol.GetOrigin())

    return l, mask2