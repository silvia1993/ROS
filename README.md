# ROS (Rotation Open Set)

PyTorch official implementation of "On the Effectiveness of Image Rotation for Open Set Domain Adaptation" in European Conference on Computer Vision 2020 (ECCV2020) 

![Test Image 1](image.jpg)

## Experiments
In order to replicate the results shown in the paper (Tables 1,2) please follow these istructions:

1. Download Office-31 and Office-Home datasets:

    - Office-31: (You can find this dataset also in the folder ROS as office.zip)
      if you want to download it use this link:
      https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
      save the folder as "office" in ROS folder.
      At the end you should have:
      office/amazon
      office/webcam
      office/dslr
    
    - Office-Home: 
      https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view
      save the folder as "office-home" in ROS folder.
      At the end you should have:
      office-home/OfficeHomeDataset_10072016/art
      office-home/OfficeHomeDataset_10072016/clipart
      office-home/OfficeHomeDataset_10072016/real_world
      office-home/OfficeHomeDataset_10072016/product
  
2. Use the python version: Python 3.6.8 

   Install all the libreries requested with the command:
   
   pip3 install -r requirements_ROS.txt 

3. Go into the folder ROS and:

    3a. In order to replicate the experiments of Office31 dataset with ResNet-50 (Table 1) run: 
    
        train_resnet50_office31.sh replacing 
        "/.../" with "/path_in_which_you_save_ROS/"
    
    3b. In order to replicate the experiments of Office31 dataset with ResNet-50 (in Table 2) run: 
    
        train_resnet50_officehome.sh replacing 
        "/.../" with "/path_in_which_you_save_ROS/"        
        
    3c. In order to replicate the experiments of Office31 dataset with VggNet (in Table 1) run: 
    
        train_vgg_office31.sh replacing 
        "/.../" with "/path_in_which_you_save_ROS/"
    
    
You can also replicate the results obtained for STA_max,STA_sum,OSBP and UAN (Tables 1,2) following the instruction of the GitHub repositories proposed by the authors:

- STA: https://github.com/thuml/Separate_to_Adapt

- OSBP: https://github.com/ksaito-ut/OPDA_BP

- UAN: https://github.com/thuml/Universal-Domain-Adaptation

## Citation

To cite, please use the following reference: 
