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
from compute_score import compute_scores_all_target


def skip(data, label, is_train):
    return False


class Trainer:
    def __init__(self, args, device,rand):
        self.args = args
        self.device = device
        self.source = self.args.source
        self.target = self.args.target
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate
        self.epochs_step1 = self.args.epochs_step1
        self.epochs_step2 = self.args.epochs_step2
        self.n_classes = self.args.n_classes
        self.n_classes_target = self.args.n_classes_target
        self.ss_classes = self.args.ss_classes
        self.cls_weight_source = self.args.cls_weight_source
        self.ss_weight_target = self.args.ss_weight_target
        self.ss_weight_source = self.args.ss_weight_source
        self.entropy_weight= self.args.entropy_weight
        self.folder_dataset = self.args.folder_dataset
        self.folder_name = self.args.folder_name
        self.folder_txt_files = self.args.folder_txt_files
        self.folder_txt_files_saving  = self.args.folder_txt_files_saving
        self.folder_log = self.args.folder_log
        self.divison_learning_rate_backbone = self.args.divison_learning_rate_backbone
        self.only_4_rotations = self.args.only_4_rotations
        self.use_weight_net_first_part = self.args.use_weight_net_first_part
        self.weight_class_unknown = self.args.weight_class_unknown         
        self.weight_center_loss = self.args.weight_center_loss
        self.use_VGG = self.args.use_VGG
        self.n_workers = self.args.n_workers


    def _do_train(self):
        
        # STEP 1 -------------------------------------------------------------------------------------

        #data-------------------------------------------------------------------------------------
        
        torch.backends.cudnn.benchmark
        
        if self.use_VGG:
            feature_extractor = VGGFc(self.device,model_name='vgg19')
        else:
            feature_extractor = ResNetFc(self.device,model_name='resnet50')
        
         #### source on which perform training of cls and self-sup task            
        images,labels = get_split_dataset_info(self.folder_txt_files+self.source+'_train_all.txt',self.folder_dataset)
        ds_source_ss = CustomDataset(images,labels,img_transformer=transform_source_ss,returns=6,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
        source_train_ss = torch.utils.data.DataLoader(ds_source_ss, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)

        images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
        ds_target_train = CustomDataset(images,labels,img_transformer=transform_target_train,returns=2,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
        target_train = torch.utils.data.DataLoader(ds_target_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)

         #### target on which compute the scores to select highest batch (integrate to the learning of ss task) and lower batch (integrate to the learning of cls task for the class unknown)
        images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
        ds_target_test_for_scores = CustomDataset(images,labels,img_transformer=transform_target_test_for_scores,returns=2,is_train=False,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
        target_test_for_scores = torch.utils.data.DataLoader(ds_target_test_for_scores, batch_size=1, shuffle=False, num_workers=self.n_workers, pin_memory=True, drop_last=False)


        #### target for the final evaluation
        images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
        ds_target_test = CustomDataset(images,labels,img_transformer=transform_target_test,returns=2,is_train=False,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
        target_test = torch.utils.data.DataLoader(ds_target_test, batch_size=1, shuffle=False, num_workers=self.n_workers, pin_memory=True, drop_last=False)

        # network -----------------------------------------------------------------------------------------------
        if self.only_4_rotations:
            discriminator_p = Discriminator(n = 1,n_s = self.ss_classes,vgg=self.use_VGG)
        else:
            discriminator_p = Discriminator(n = self.n_classes,n_s = self.ss_classes,vgg=self.use_VGG)

        cls = CLS(feature_extractor.output_num(), self.n_classes+1, bottle_neck_dim=256,vgg=self.use_VGG)

        discriminator_p.to(self.device)
        feature_extractor.to(self.device)        
        cls.to(self.device)                 

        net = nn.Sequential(feature_extractor, cls).to(self.device)
            
        center_loss = CenterLoss(num_classes=self.ss_classes*self.n_classes, feat_dim=256*self.n_classes, use_gpu=True,device=self.device)
        if self.use_VGG:
            center_loss_object = CenterLoss(num_classes=self.n_classes, feat_dim=4096, use_gpu=True,device=self.device)
        else:
            center_loss_object = CenterLoss(num_classes=self.n_classes, feat_dim=2048, use_gpu=True,device=self.device)

        # scheduler, optimizer ---------------------------------------------------------
        max_iter = int(self.epochs_step1*len(source_train_ss))
        scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
                            
        params = list(discriminator_p.parameters()) 
        if self.weight_center_loss>0:
            params = params+ list(center_loss.parameters())

        optimizer_discriminator_p = OptimWithSheduler(optim.SGD(params, lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)

        if not self.use_VGG:
            for name,param in feature_extractor.named_parameters():
                words= name.split('.')
                if words[1] =='layer4':
                    param.requires_grad = True
                else:
                    param.requires_grad = False  

            params_cls = list(cls.parameters())
            optimizer_cls = OptimWithSheduler(optim.SGD([{'params': params_cls},{'params': feature_extractor.parameters(), 'lr': (self.learning_rate/self.divison_learning_rate_backbone)}], lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)   
  
        else:
            for name,param in feature_extractor.named_parameters():
                words= name.split('.')
                if words[1] =='classifier':
                    param.requires_grad = True
                else:
                    param.requires_grad = False                          
            params_cls = list(cls.parameters())                            
            optimizer_cls = OptimWithSheduler(optim.SGD([{'params': params_cls},{'params': feature_extractor.parameters(), 'lr': (self.learning_rate/self.divison_learning_rate_backbone)}], lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)


        log = Logger(self.folder_log+'/step', clear=True)
        target_train = cycle(target_train)
        
        k=0
        print('\n')
        print('Separation known/unknown phase------------------------------------------------------------------------------------------')
        print('\n')

        while k <self.epochs_step1:
            print('Epoch: ',k)
            for (i, (im_source,im_source_ss,label_source,label_source_ss,label_source_ss_center,label_source_center_object)) in enumerate(source_train_ss):

                (im_target,_) = next(target_train)

                global loss_object_class
                global acc_train
                global loss_rotation
                global acc_train_rot
                global loss_center

                im_source = im_source.to(self.device)
                im_target = im_target.to(self.device)
                im_source_ss = im_source_ss.to(self.device)
                label_source = label_source.to(self.device)
                label_source_ss = label_source_ss.to(self.device)
                label_source_ss_center = label_source_ss_center.to(self.device)
                label_source_center_object = label_source_center_object.to(self.device)

                (_, _, _, predict_prob_source) = net.forward(im_source)
                (_, _, _, _) = net.forward(im_target)

                fs1_ss = feature_extractor.forward(im_source_ss)

                fs1_original = feature_extractor.forward(im_source)
                _ = feature_extractor.forward(im_target)

                double_input = torch.cat((fs1_original, fs1_ss), 1)
                fs1_ss=double_input  

                p0,p0_center = discriminator_p.forward(fs1_ss) 
                p0 = nn.Softmax(dim=-1)(p0)                
                # =========================loss function
                ce = CrossEntropyLoss(label_source, predict_prob_source)
                d1 = CrossEntropyLoss(label_source_ss,p0)
                center,_ = center_loss(p0_center, label_source_ss_center)

                with OptimizerManager([optimizer_cls, optimizer_discriminator_p]):
                    loss_object_class = self.cls_weight_source*ce
                    loss_rotation = self.ss_weight_source*d1
                    loss_center = self.weight_center_loss*center
                    loss = loss_object_class + loss_rotation +loss_center
                    loss.backward()
                    
                    if self.weight_center_loss>0:
                        for param in center_loss.parameters():
                            param.grad.data *= (1./self.weight_center_loss)


                log.step += 1

            k += 1
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32)).to(self.device)
            counter_ss =  AccuracyCounter()
            counter_ss.addOntBatch(variable_to_numpy(p0), variable_to_numpy(label_source_ss))
            acc_train_rot = torch.from_numpy(np.asarray([counter_ss.reportAccuracy()], dtype=np.float32)).to(self.device)
            track_scalars(log, ['loss_object_class', 'acc_train', 'loss_rotation','acc_train_rot','loss_center'],globals())
            
        select_low = compute_scores_all_target(target_test_for_scores,feature_extractor,discriminator_p,net,self.use_VGG,self.n_classes,self.ss_classes,self.device,self.source,self.target,self.folder_txt_files,self.folder_txt_files_saving)
                    
# ========================= Add target samples to cls and discriminator_p classifiers in function of the score
    #data--------------------------------------------------------------------------------------------------------------- 
        self.only_4_rotations = True

        images,labels = get_split_dataset_info(self.folder_txt_files_saving+self.source+'_'+self.target+'_test_high.txt',self.folder_dataset)
        ds_target_high = CustomDataset(images,labels,img_transformer=transform_target_ss_step2,returns=3,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
        target_train_high = torch.utils.data.DataLoader(ds_target_high, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)
                                

        images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
        ds_target = CustomDataset(images,labels,img_transformer=transform_target_ss_step2,returns=3,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
        target_train = torch.utils.data.DataLoader(ds_target, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)
                
        images,labels = get_split_dataset_info(self.folder_txt_files_saving+self.source+'_'+self.target+'_test_low.txt',self.folder_dataset)
        ds_target_low = CustomDataset(images,labels,img_transformer=transform_source_ss_step2,returns=6,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
        target_train_low = torch.utils.data.DataLoader(ds_target_low, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)

            # network --------------------------------------------------------------------------------------------------------------------------
        discriminator_p = Discriminator(n = 1,n_s = self.ss_classes,vgg=self.use_VGG) 
        discriminator_p.to(self.device)

        if not self.use_weight_net_first_part:
            if self.use_VGG:
                feature_extractor = VGGFc(self.device,model_name='vgg19')
            else:
                feature_extractor = ResNetFc(self.device,model_name='resnet50')
            cls = CLS(feature_extractor.output_num(), self.n_classes+1, bottle_neck_dim=256,vgg=self.use_VGG)
            feature_extractor.to(self.device)        
            cls.to(self.device)
            net = nn.Sequential(feature_extractor, cls).to(self.device)
                
        if len(target_train_low) >= len(target_train_high):
            length = len(target_train_low)
        else:
            length = len(target_train_high)

        max_iter = int(self.epochs_step2*length)
                
        scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
        params = list(discriminator_p.parameters())              
                
        optimizer_discriminator_p = OptimWithSheduler(optim.SGD(params, lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)

        if not self.use_VGG:
            for name,param in feature_extractor.named_parameters():
                words= name.split('.')
                if words[1] =='layer4':
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            params_cls = list(cls.parameters())
            optimizer_cls = OptimWithSheduler(optim.SGD([{'params': params_cls},{'params': feature_extractor.parameters(), 'lr': (self.learning_rate/self.divison_learning_rate_backbone)}], lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)
                
        else:
            for name,param in feature_extractor.named_parameters():
                words= name.split('.')
                if words[1] =='classifier':
                    param.requires_grad = True
                else:
                    param.requires_grad = False                             
            params_cls = list(cls.parameters())
            optimizer_cls = OptimWithSheduler(optim.SGD([{'params': params_cls},{'params': feature_extractor.parameters(), 'lr': (self.learning_rate/self.divison_learning_rate_backbone)}], lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)

        k=0
        print('\n')
        print('Adaptation phase--------------------------------------------------------------------------------------------------------')
        print('\n')
        ss_weight_target = self.ss_weight_target            
        weight_class_unknown = 1/(select_low*(self.n_classes/(len(source_train_ss)*self.batch_size)))
        
        while k <self.epochs_step2:
            print('Epoch: ',k)
            iteration = cycle(target_train)
    
            if len(target_train_low) > len(target_train_high):
                num_iterations =  len(target_train_low)
                num_iterations_smaller = len(target_train_high)
                target_train_low_iter = iter(target_train_low)
                target_train_high_iter = cycle(target_train_high)
            else:
                num_iterations = len(target_train_high)
                num_iterations_smaller = len(target_train_low)
                target_train_low_iter = cycle(target_train_low)
                target_train_high_iter = iter(target_train_high)

            for i in range(num_iterations):

                global entropy_loss

                (im_target_entropy,_,_) = next(iteration)
                (im_source,im_source_ss,label_source,label_source_ss,_,_) = next(target_train_low_iter)
                (im_target,im_target_ss,label_target_ss) = next(target_train_high_iter)

                im_source = im_source.to(self.device)
                im_source_ss = im_source_ss.to(self.device)
                label_source = label_source.to(self.device)
                label_source_ss = label_source_ss.to(self.device)
                im_target = im_target.to(self.device)
                im_target_ss = im_target_ss.to(self.device)
                label_target_ss = label_target_ss.to(self.device)
                im_target_entropy = im_target_entropy.to(self.device)
                

                ft1_ss = feature_extractor.forward(im_target_ss)
                ft1_original = feature_extractor.forward(im_target)
                double_input_t = torch.cat((ft1_original, ft1_ss), 1)
                ft1_ss=double_input_t

                (_, _, _, predict_prob_source) = net.forward(im_source)

                (_ ,_, _, _) = net.forward(im_target_entropy)
                (_, _, _, predict_prob_target) = net.forward(im_target)

                p0_t,_ = discriminator_p.forward(ft1_ss)
                p0_t = nn.Softmax(dim=-1)(p0_t)

                    # =========================loss function
                class_weight = np.ones((self.n_classes+1),dtype=np.dtype('f'))
                class_weight[self.n_classes]= weight_class_unknown*self.weight_class_unknown
                class_weight = (torch.from_numpy(class_weight)).to(self.device)
                ce = CrossEntropyLoss(label_source, predict_prob_source,class_weight)

                entropy = EntropyLoss(predict_prob_target)
                d1_t = CrossEntropyLoss(label_target_ss,p0_t)

                with OptimizerManager([optimizer_cls, optimizer_discriminator_p]):
                    loss_object_class = self.cls_weight_source*ce
                    loss_rotation = ss_weight_target*d1_t
                    entropy_loss = self.entropy_weight*entropy

                    loss = loss_object_class + loss_rotation + entropy_loss
                    loss.backward()
                    log.step += 1

            k += 1
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32)).to(self.device)

            counter_ss =  AccuracyCounter()
            counter_ss.addOntBatch(variable_to_numpy(p0_t), variable_to_numpy(label_target_ss))
            acc_train_rot = torch.from_numpy(np.asarray([counter_ss.reportAccuracy()], dtype=np.float32)).to(self.device)
            track_scalars(log, ['loss_object_class', 'acc_train', 'loss_rotation', 'acc_train_rot','entropy_loss'], globals())

            global predict_prob
            global label
            global predict_index        

            # =================================evaluation
            if k%10==0 or k==(self.epochs_step2):
                with TrainingModeManager([feature_extractor, cls], train=False) as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
                    for (i, (im, label)) in enumerate(target_test):
                        with torch.no_grad():
                            im = im.to(self.device)
                            label = label.to(self.device)
                            (ss, fs,_,  predict_prob) = net.forward(im)
                            predict_prob,label = [variable_to_numpy(x) for x in (predict_prob,label)]
                            label = np.argmax(label, axis=-1).reshape(-1, 1)
                            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
                            accumulator.updateData(globals())

                for x in accumulator.keys():
                    globals()[x] = accumulator[x]
                y_true = label.flatten()
                y_pred = predict_index.flatten()
                m = extended_confusion_matrix(y_true, y_pred, true_labels=range(self.n_classes_target), pred_labels=range(self.n_classes+1))

                cm = m
                cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
                acc_os_star = sum([cm[i][i] for i in range(self.n_classes)]) / (self.n_classes)
                unkn = sum([cm[i][self.n_classes] for i in range(self.n_classes,self.n_classes_target)]) / (self.n_classes_target - (self.n_classes))
                acc_os = (acc_os_star * (self.n_classes) + unkn) / (self.n_classes+1)
                hos = (2*acc_os_star*unkn)/(acc_os_star+unkn)
                print('os',acc_os)
                print('os*', acc_os_star)
                print('unkn',unkn)
                print('hos',hos)

                net.train()
                
            #torch.save(net.state_dict(),self.folder_name+'/model.pkl')