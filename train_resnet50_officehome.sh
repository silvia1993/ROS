
##########################train of Office-Home dataset using ResNet50#########################################################

      
python3 train.py --source product_0-24 --target clipart_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source product_0-24 --target art_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source product_0-24 --target real_world_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/



python3 train.py --source art_0-24 --target clipart_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source art_0-24 --target product_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source art_0-24 --target real_world_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/



python3 train.py --source clipart_0-24 --target art_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source clipart_0-24 --target product_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source clipart_0-24 --target real_world_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2  --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/



python3 train.py --source real_world_0-24 --target art_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source real_world_0-24 --target product_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/

python3 train.py --source real_world_0-24 --target clipart_0-64 --epochs_step1 150 --epochs_step2 45 --use_weight_net_first_part  --weight_center_loss 0.001 --weight_class_unknown 2 --folder_txt_files /.../ROS/data/ --folder_dataset /.../ROS/