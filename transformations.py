from data import *
from utilities import *
from networks import *
import numpy as np
from random import sample, random
from PIL import Image
import torchvision.transforms as transforms
import sys
import torchvision
from itertools import chain
from skimage.transform import resize
from center_loss import CenterLoss
import math
from sklearn.metrics import roc_auc_score
import random
from sklearn import preprocessing
from itertools import cycle     
from scipy.misc import imread, imresize


def transform_target_train(data, label, is_train,ss_classes,n_classes,only_4_rotations,n_classes_target):
    data = imresize(data, (256,256))
    original_image = tl.prepro.crop(data, 224, 224, is_random=is_train)                   
    original_image = np.transpose(original_image, [2, 0, 1])
    original_image = np.asarray(original_image, np.float32) / 255.0            
    label = one_hot(n_classes_target, label)  
    return original_image, label    
        
def transform_source_ss(data, label, is_train,ss_classes,n_classes,only_4_rotations,n_classes_target):
    ss_transformation = np.random.randint(ss_classes)
    data = imresize(data, (256,256))
    original_image = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
            
    if ss_transformation==0:
        ss_data=data
    if ss_transformation==1:
        ss_data=np.rot90(data,k=1)              
    if ss_transformation==2:
        ss_data=np.rot90(data,k=2)
    if ss_transformation==3:
        ss_data=np.rot90(data,k=3)
                                
    if only_4_rotations:
        ss_label = one_hot(ss_classes,ss_transformation)
        label_ss_center = ss_transformation
    else:
        ss_label = one_hot(ss_classes*n_classes,(ss_classes*label)+ss_transformation)
        label_ss_center = (ss_classes*label)+ss_transformation

    ss_data = np.transpose(ss_data, [2, 0, 1])
    ss_data = np.asarray(ss_data, np.float32) / 255.0
                        
    original_image = np.transpose(original_image, [2, 0, 1])
    original_image = np.asarray(original_image, np.float32) / 255.0
    label_object_center = label
    label = one_hot(n_classes+1, label)           

    return original_image,ss_data,label,ss_label,label_ss_center,label_object_center 

        
def transform_source_ss_step2(data, label, is_train,ss_classes,n_classes,only_4_rotations,n_classes_target):
    data = imresize(data, (256,256))
    ss_transformation = np.random.randint(ss_classes)
            
    original_image = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
            
    if ss_transformation==0:
        ss_data=data
    if ss_transformation==1:
        ss_data=np.rot90(data,k=1)              
    if ss_transformation==2:
        ss_data=np.rot90(data,k=2)
    if ss_transformation==3:
        ss_data=np.rot90(data,k=3)
                
    ss_label = one_hot(ss_classes,ss_transformation)
            
    ss_data = np.transpose(ss_data, [2, 0, 1])
    ss_data = np.asarray(ss_data, np.float32) / 255.0
                        
    original_image = np.transpose(original_image, [2, 0, 1])
    original_image = np.asarray(original_image, np.float32) / 255.0
    label_object_center = label
    label = one_hot(n_classes+1, label)   
    label_ss_center = ss_transformation

    return original_image,ss_data,label,ss_label,label_ss_center,label_object_center


def transform_target_ss_step2(data, label, is_train,ss_classes,n_classes,only_4_rotations,n_classes_target):
    data = imresize(data, (256,256))
    ss_transformation = np.random.randint(ss_classes)
            
    original_image = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
            
    if ss_transformation==0:
        ss_data=data
    if ss_transformation==1:
        ss_data=np.rot90(data,k=1)              
    if ss_transformation==2:
        ss_data=np.rot90(data,k=2)
    if ss_transformation==3:
        ss_data=np.rot90(data,k=3)
                
    ss_label = one_hot(ss_classes,ss_transformation)
            
    ss_data = np.transpose(ss_data, [2, 0, 1])
    ss_data = np.asarray(ss_data, np.float32) / 255.0
                        
    original_image = np.transpose(original_image, [2, 0, 1])
    original_image = np.asarray(original_image, np.float32) / 255.0                

    return original_image,ss_data,ss_label 
  
def transform_target_test(data, label, is_train,ss_classes,n_classes,only_4_rotations,n_classes_target):
    data = imresize(data, (256,256))
    label = one_hot(n_classes_target, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

def transform_target_test_mnist(data, label, is_train,ss_classes,n_classes,only_4_rotations,n_classes_target,mean,std):

    label = one_hot(n_classes_target, label)
    data = imresize(data, (32,32))
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    
    mean = mean.repeat(3)
    mean = np.asarray(mean, dtype=np.float32).reshape((3, 1, 1))
    std = std.repeat(3)
    std = np.asarray(std, dtype=np.float32).reshape((3, 1, 1))
    
    data = (data - mean)/std    
    
    return data, label

def transform_target_test_for_scores(data, label, is_train,ss_classes,n_classes,only_4_rotations,n_classes_target):
    data = imresize(data, (256,256))
    return data, label
