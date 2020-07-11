from data import *
from transformations import *
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

def create_txt_target(type_subset,subset_index_high,source,target,folder_txt_files,folder_txt_files_saving,n_classes):
    path = folder_txt_files+target+'_test.txt'
    new_file = folder_txt_files_saving+source+'_'+target+'_test_'+type_subset+'.txt'
    cont = 0
    f = open(path, 'r') 
    list_images = f.readlines()
            
    w = open(new_file, 'w') 

    for i in list_images:
        for k in subset_index_high:
            if cont==k:
                if type_subset is 'low': 
                    words = i.split(' ')
                    w.write(words[0]+' '+str(n_classes)+'\n')
                else:
                    w.write(i)
        cont=cont+1
    if type_subset is 'low':
        path = folder_txt_files+source+'_train_all.txt'
        f = open(path, 'r') 
        list_images = f.readlines()
        for j in list_images:                    
            w.write(j)
   
                    
def compute_scores_all_target(target_test,feature_extractor,discriminator_p,net,vgg,n_classes,ss_classes,device,source,target,folder_txt_files,folder_txt_files_saving):
            
            all_target_labels = []
            all_target_predictions = []
            len_target = len(target_test)

            with torch.no_grad():
                len_target = len(target_test)
                scores = torch.zeros(len_target)
                scores_entropy_ss = torch.zeros(len_target)

                if vgg:
                    target_original = torch.zeros(len_target,4096)
                else:
                    target_original = torch.zeros(len_target,2048)

                for (i, (im_target,label_target)) in enumerate(target_test):
                    all_target_labels.append(label_target[0].item())
                    k_list = torch.zeros(n_classes)
                    logit = 0

                    for j in range(ss_classes):
                        ss_data_orig = tl.prepro.crop(im_target[0], 224, 224, is_random=False)
                        if j==0 or j==1 or j==2 or j==3:
                            ss_data=np.rot90(ss_data_orig,k=j)                                  

                        if j==0:
                            ss_data_orig = np.transpose(ss_data_orig, [2, 0, 1])
                            ss_data_orig = np.asarray(ss_data_orig, np.float32) / 255.0                    
                            ss_data_orig = torch.from_numpy(ss_data_orig).to(device)
                            net.eval()   
                            (ft1, _, _, class_label) = net.forward(ss_data_orig)     
                            target_original[i] = ft1
                            net.train()                                                      

                        ss_data = np.transpose(ss_data, [2, 0, 1])
                        ss_data = np.asarray(ss_data, np.float32) / 255.0                    
                        ss_data = torch.from_numpy(ss_data).to(device)
                        
                        feature_extractor.eval()
                        ft_ss= feature_extractor.forward(ss_data)
                        feature_extractor.train()
                        
                        double_input = torch.cat((ft1.cpu(), ft_ss.cpu()), 1)
                        ft_ss=double_input
                        ft_ss = ft_ss.to(device)

                        discriminator_p.eval()        
                        p0,features = discriminator_p.forward(ft_ss)
                        discriminator_p.train()

                        p0 = nn.Softmax(dim=-1)(p0)

                        scores_entropy_ss[i] = scores_entropy_ss[i]+EntropyLoss(p0)

                        for k in range(n_classes):
                            logit = p0[0][(ss_classes*k)+j]
                            k_list[k]=k_list[k]+logit
                                                                
                    k_list=k_list/ss_classes 
                    #normality score
                    scores[i] = max(k_list) 
                    #entropy ss score
                    scores_entropy_ss[i] = scores_entropy_ss[i]/ss_classes

                all_target_labels = np.asarray(all_target_labels)

                #normalization entropy ss score
                scores_entropy_ss =  (scores_entropy_ss-min(scores_entropy_ss))/(max(scores_entropy_ss)-min(scores_entropy_ss))
                scores_entropy_ss = 1-scores_entropy_ss                

                #score = max(entropy,normality)
                score_sum = np.maximum(scores_entropy_ss,scores)

                scores_ordered_entropy_ss_sum = (score_sum).argsort().numpy()   
                scores_entropy_ss_ordered_sum = score_sum[scores_ordered_entropy_ss_sum]
                label_target_sorted_entropy_ss_sum = all_target_labels[scores_ordered_entropy_ss_sum]               
               
                scores_for_mean = np.asarray(scores_entropy_ss_ordered_sum)
                mean_scores = sum(scores_for_mean)/len(scores_for_mean)
                number = int((str(mean_scores).split('.'))[1][0])+1
                threshold = float('0.'+str(number))
                num_low=0
                for value in scores_for_mean:
                    if value>threshold:
                        num_low = num_low+1
                select_high=num_low
                select_low =num_low
                
                create_txt_target('high',np.asarray(scores_ordered_entropy_ss_sum)[-select_high:],source,target,folder_txt_files,folder_txt_files_saving,n_classes)
                create_txt_target('low',np.asarray(scores_ordered_entropy_ss_sum)[0:select_low],source,target,folder_txt_files,folder_txt_files_saving,n_classes)
                
                
                return select_low