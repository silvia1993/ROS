
##########################train of Office-31 dataset using ResNet-50#########################################################


python3 train.py --source dslr_0-9 --target webcam_0-9_20-30 --epochs_step1 80 --epochs_step2 80 --n_classes 10 --n_classes_target 21  --use_weight_net_first_part --weight_class_unknown 2 --weight_center_loss 0.1 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source webcam_0-9 --target dslr_0-9_20-30 --epochs_step1 80 --epochs_step2 80 --n_classes 10 --n_classes_target 21  --use_weight_net_first_part --weight_class_unknown 2 --weight_center_loss 0.1 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source dslr_0-9 --target amazon_0-9_20-30 --epochs_step1 80 --epochs_step2 80 --n_classes 10 --n_classes_target 21  --use_weight_net_first_part --weight_class_unknown 2 --weight_center_loss 0.1 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source webcam_0-9 --target amazon_0-9_20-30 --epochs_step1 80 --epochs_step2 80 --n_classes 10 --n_classes_target 21  --use_weight_net_first_part --weight_class_unknown 2 --weight_center_loss 0.1 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source amazon_0-9 --target dslr_0-9_20-30 --epochs_step1 80 --epochs_step2 80 --n_classes 10 --n_classes_target 21  --use_weight_net_first_part --weight_class_unknown 2 --weight_center_loss 0.1 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source amazon_0-9 --target webcam_0-9_20-30 --epochs_step1 80 --epochs_step2 80 --n_classes 10 --n_classes_target 21  --use_weight_net_first_part --weight_class_unknown 2 --weight_center_loss 0.1 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/