import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import scipy.io as sio
import codecs
import os
import os.path




def _dataset_info(txt_labels,folder_dataset):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_name = folder_dataset+row[0]
        file_names.append(file_name)
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list,folder_dataset):
    names, labels = _dataset_info(txt_list,folder_dataset)
    return names, labels

class CustomDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None,returns=None,is_train=None,ss_classes=None,n_classes=None,only_4_rotations=None,n_classes_target=None):
        self.data_path = ""
        self.names = names
        self.labels = labels
        self.N = len(self.names)
        self._image_transformer = img_transformer
        self.is_train = is_train
        self.returns = returns
        self.ss_classes = ss_classes
        self.n_classes = n_classes
        self.only_4_rotations = only_4_rotations
        self.n_classes_target = n_classes_target
        
    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')

        if self.returns==3:
            data,data_ss,label_ss = self._image_transformer(img,self.labels[index], self.is_train,self.ss_classes,self.n_classes,self.only_4_rotations,self.n_classes_target)
            return data,data_ss,label_ss 
        elif self.returns==4:
            data,data_ss,label,label_ss =self._image_transformer(img,self.labels[index], self.is_train,self.ss_classes,self.n_classes,self.only_4_rotations,self.n_classes_target)
            return data,data_ss,label,label_ss
        elif self.returns==5:
            data,data_ss,label,label_ss,label_ss_center = self._image_transformer(img,self.labels[index], self.is_train,self.ss_classes,self.n_classes,self.only_4_rotations,self.n_classes_target)
            return data,data_ss,label,label_ss,label_ss_center
        elif self.returns==6:
            data,data_ss,label,label_ss,label_ss_center,label_object_center = self._image_transformer(img,self.labels[index], self.is_train,self.ss_classes,self.n_classes,self.only_4_rotations,self.n_classes_target)  
            return data,data_ss,label,label_ss,label_ss_center,label_object_center
        elif self.returns==2:
            data,label = self._image_transformer(img,self.labels[index], self.is_train,self.ss_classes,self.n_classes,self.only_4_rotations,self.n_classes_target)
            return data,label

    def __len__(self):
        return len(self.names)