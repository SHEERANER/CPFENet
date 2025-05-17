import os
import numpy as np
import torch
import SimpleITK as sitk
import pydicom

def ImageResample(sitk_image, new_spacing=[1.0, 1.0, 1.0], is_label=False):
    '''
    Resample SimpleITK image to new spacing

    Args:
        sitk_image (SimpleITK.Image): Image to be resampled
        new_spacing (list): New spacing for the image
        is_label (bool): Whether the image is a label image

    Returns:
        SimpleITK.Image: Resampled image
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)
    newimage = resample.Execute(sitk_image)
    return newimage

def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    '''
    Perform window transformation on CT array

    Args:
        ct_array (numpy.ndarray): CT array to be transformed
        windowWidth (float): Window width
        windowCenter (float): Window center
        normal (bool): Whether to normalize the output to [0, 1]

    Returns:
        numpy.ndarray: Transformed CT array
    '''
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('float32')
    return newimg

def normalise_zero_one(image):
    '''
    Normalise image to fit [0, 1] range

    Args:
        image (numpy.ndarray): Image to be normalised

    Returns:
        numpy.ndarray: Normalised image
    '''
    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret

def read_dicom_pydicom(file, size, min_images=100):
    '''
    Read DICOM files using pydicom

    Args:
        file (str): Path to DICOM directory
        size (tuple): Target size for the image
        min_images (int): Minimum number of images required

    Returns:
        list: List containing the 3D image tensor and patient information
    '''
    WIDTH = size[1]
    HEIGHT = size[2]
    DCM_list = []
    len_file = len(os.listdir(file))
    if len_file < min_images:
        return None
    else:
        dcm_ref = pydicom.dcmread(os.path.join(file, "IM0"))

        info = {}
        info["PatientID"] = dcm_ref.PatientID  # Patient ID
        info["PatientName"] = dcm_ref.PatientName  # Patient name
        info["PatientAge"] = dcm_ref.PatientAge  # Patient age
        info['PatientSex'] = dcm_ref.PatientSex  # Patient gender
        info['StudyID'] = dcm_ref.StudyID  # Study ID
        info['StudyDate'] = dcm_ref.StudyDate  # Study date
        info['StudyTime'] = dcm_ref.StudyTime  # Study time
        info['InstitutionName'] = dcm_ref.InstitutionName  # Institution name
        info['Manufacturer'] = dcm_ref.Manufacturer  # Manufacturer

        for i in range(len_file):
            file_name = os.path.join(file, "IM " + str(i))
            dcm = pydicom.dcmread(file_name).pixel_array.astype(np.int32)
            DCM_list.append(transform.resize(dcm, (WIDTH, HEIGHT)))
        print(file)
        DCM_list = np.array(DCM_list)
        DCM_list = transform.resize(DCM_list, size)
        img3d = torch.tensor(DCM_list)
        return [img3d, info]

def read_dicom_sitk(file, size):
    '''
    Read DICOM files using SimpleITK

    Args:
        file (str): Path to DICOM directory
        size (tuple): Target size for the image

    Returns:
        list: List containing the 3D image tensor and patient information
    '''
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D_raw = series_reader.Execute()

    image3D = ImageResample(image3D_raw)
    image3D = window_transform(sitk.GetArrayFromImage(image3D), 1300, -450, False)
    image3D = normalise_zero_one(image3D)
    image3D = transform.resize(image3D, size)
    img3D = torch.tensor(image3D)
    info = None
    return [img3D, info]
