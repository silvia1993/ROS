
##########################train of Office-31 dataset using VGGNet#########################################################


python3 train.py --source amazon_0-9 --target webcam_0-9_20-30  --epochs_step1 100 --epochs_step2 200 --n_classes 10 --n_classes_target 21 --weight_class_unknown 1.5 --weight_center_loss 0.1 --use_VGG --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source webcam_0-9 --target dslr_0-9_20-30  --epochs_step1 100 --epochs_step2 200 --n_classes 10 --n_classes_target 21 --weight_class_unknown 1.5 --weight_center_loss 0.1 --use_VGG --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source webcam_0-9 --target amazon_0-9_20-30  --epochs_step1 100 --epochs_step2 200 --n_classes 10 --n_classes_target 21 --weight_class_unknown 1.5 --weight_center_loss 0.1 --use_VGG --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source dslr_0-9 --target amazon_0-9_20-30  --epochs_step1 100 --epochs_step2 200 --n_classes 10 --n_classes_target 21 --weight_class_unknown 1.5 --weight_center_loss 0.1 --use_VGG --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source dslr_0-9 --target webcam_0-9_20-30  --epochs_step1 100 --epochs_step2 200 --n_classes 10 --n_classes_target 21 --weight_class_unknown 1.5 --weight_center_loss 0.1 --use_VGG --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source amazon_0-9 --target dslr_0-9_20-30  --epochs_step1 100 --epochs_step2 200 --n_classes 10 --n_classes_target 21 --weight_class_unknown 1.5 --weight_center_loss 0.1 --use_VGG --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

