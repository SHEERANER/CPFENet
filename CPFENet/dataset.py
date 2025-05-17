import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pydicom
from skimage import transform

class CTDataset(Dataset):
    '''
    Dataset class for CT images

    Args:
        data_dir (str): Path to dataset directory
        input_w (int): the width of the 3D images
        input_d (int): the depth of the 3D images
        input_h (int): the height of the 3D images
        threeD (bool): Whether to treat images as 3D
    '''

    def __init__(self, data_dir, input_w=128, input_d=128, input_h=128, threeD=True):
        self.dir = data_dir
        self.threeD = threeD
        img_list = []
        ref_list = []
        label_list = []
        dir_list = []
        for dir, subname, name_list in os.walk(data_dir):
            if "CPFE" in dir:
                label = 2
            elif "COPD" in dir:
                label = 1
            else:
                label = 0
            if len(name_list) > 100:
                print(dir)
                dicom = read_dicom_sitk(dir, (input_h, input_w, input_d))
                if dicom is not None:
                    if threeD:
                        dicom_tensor = dicom[0].float().unsqueeze(0)
                    else:
                        dicom_tensor = dicom[0].float()
                    img_list.append(dicom_tensor)
                    ref_list.append(dicom[1])
                    label_list.append(label)
                    dir_list.append(dir)
        self.img_list = img_list
        self.ref_list = ref_list
        self.label_list = label_list
        self.dir_list = dir_list

    def transform_shape(self, shape):
        '''
        Transform the shape of images

        Args:
            shape (tuple): Target shape
        '''
        new_images = []
        imgs = self.img_list
        for img in imgs:
            img_numpy = img.squeeze().numpy()
            img_numpy = transform.resize(img_numpy, shape)
            img_tensor = torch.tensor(img_numpy)
            if self.threeD:
                img_tensor = img_tensor.float().unsqueeze(0)
            new_images.append(img_tensor)
        self.img_list = new_images

    def transform_dimension(self, threeD=False):
        '''
        Transform the dimension of images

        Args:
            threeD (bool): Whether to treat images as 3D
        '''
        new_images = []
        imgs = self.img_list
        if self.threeD != threeD and threeD == False:
            for img in imgs:
                new_image = img.squeeze()
                new_images.append(new_image)
            self.img_list = new_images
        elif self.threeD != threeD and threeD == True:
            for img in imgs:
                new_image = img.permute((1, 2, 0)).unsqueeze(0)
                new_images.append(new_image)
            self.img_list = new_images

    def __getitem__(self, i):
        return self.img_list[i].float(), self.label_list[i]

    def __len__(self):
        return len(self.label_list)
    
class CTDataset_COPD(Dataset):
    """
    Custom Dataset class for loading and preprocessing CT images.

    Args:
        data_dir (str): Path to the dataset directory.
        threeD (bool): Whether to treat images as 3D.

    Attributes:
        dir (str): Path to the dataset directory.
        threeD (bool): Whether to treat images as 3D.
        img_list (list): List of processed image tensors.
        ref_list (list): List of patient information (not used in this implementation).
        label_list (list): List of labels corresponding to the images.
    """

    def __init__(self, data_dir, threeD=True):
        self.dir = data_dir
        self.threeD = threeD
        self.img_list = []
        self.ref_list = []
        self.label_list = []

        for root, _, files in os.walk(data_dir):
            if "CPFE" in root or "COPD" in root:
                label = 1
            else:
                label = 0

            if len(files) > 100 and ("CPFE" in root or "COPD" in root):
                dicom_data = read_dicom_sitk(root, (128, 128, 128))
                if dicom_data is not None:
                    image_tensor = dicom_data[0].float()
                    if threeD:
                        image_tensor = image_tensor.permute((1, 2, 0)).unsqueeze(0)
                    self.img_list.append(image_tensor)
                    self.ref_list.append(dicom_data[1])
                    self.label_list.append(label)

    def transform_shape(self, target_shape):
        """
        Transform the shape of the images in the dataset.

        Args:
            target_shape (tuple): Target shape for the images.
        """
        new_images = []
        for img in self.img_list:
            img_numpy = img.squeeze().numpy()
            img_resized = transform.resize(img_numpy, target_shape)
            img_tensor = torch.tensor(img_resized)
            if self.threeD:
                img_tensor = img_tensor.float().permute((1, 2, 0)).unsqueeze(0)
            new_images.append(img_tensor)
        self.img_list = new_images

    def transform_dimension(self, threeD=True):
        """
        Transform the dimensionality of the images in the dataset.

        Args:
            threeD (bool): Whether to treat images as 3D.
        """
        new_images = []
        for img in self.img_list:
            if self.threeD != threeD and not threeD:
                new_img = img.squeeze().permute((2, 0, 1))
            elif self.threeD != threeD and threeD:
                new_img = img.permute((1, 2, 0)).unsqueeze(0)
            new_images.append(new_img)
        self.img_list = new_images
        self.threeD = threeD

    def __getitem__(self, index):
        return self.img_list[index].float(), self.label_list[index]

    def __len__(self):
        return len(self.label_list)