import sys
import argparse
from steps_separation_adaptation import Trainer
import numpy as np
import torch
import os


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #domains
    parser.add_argument("--source", help="Source")
    parser.add_argument("--target", help="Target")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--divison_learning_rate_backbone", type=float, default=10.0, help="Scaling factor of the learning rate used for the part pf the backbone not freezed")
    
    #epochs step1 and step2
    parser.add_argument("--epochs_step1", type=int, default=80, help="Epochs of step1")
    parser.add_argument("--epochs_step2", type=int, default=80,help="Epochs of step2")

    #number of classes: known, unknown and the classes of self-sup task
    parser.add_argument("--n_classes", type=int, default=25, help="Number of classes of source domain -- known classes")
    parser.add_argument("--n_classes_target", type=int, default=65,help="Number of classes of target domain -- known+unknown classes")
    parser.add_argument("--ss_classes", "-rc", type=int, default=4, help="Number of classes for the self-supervised task")

    #weights used during training
    parser.add_argument("--ss_weight_source", type=float, default=1.0, help="Weight of the source domain for the ss task (it acts in step1)")
    parser.add_argument("--ss_weight_target", type=float, default=3.0, help="Weight of the target domain for the ss task (it acts in step2)")
    parser.add_argument("--cls_weight_source", type=float, default=1.0, help="Weight for the cls task (it acts in step1 and step2)")
    parser.add_argument("--entropy_weight", type=float, default=0.1, help="Weight for the ss task (it acts in step2)")
    parser.add_argument("--weight_center_loss", type=float, default=0.0, help="Weight of the center loss for the ss task (it acts in step1)")
    parser.add_argument("--weight_class_unknown", type=float, default=1.0, help="Power of learning of the unknown class (it acts in step2)")

    #path of the folders used
    parser.add_argument("--folder_dataset",default=None, help="Path to the dataset")
    parser.add_argument("--folder_txt_files", default='/.../ROS/data/',help="Path to the txt files of the dataset")
    parser.add_argument("--folder_txt_files_saving", default='/.../ROS/data/',help="Path where to save the new txt files")
    parser.add_argument("--folder_log", default=None, help="Path of the log folder")

    #to select gpu/num of workers
    parser.add_argument("--gpu", type=int, default=0, help="gpu chosen for the training")
    parser.add_argument("--n_workers", type=int, default=4, help="num of worker used")
    
    parser.add_argument("--use_VGG", action='store_true', default=False, help="If use VGG")
    parser.add_argument("--use_weight_net_first_part", action='store_true', default=False, help="If use the weight computed in the step1 for step2")
    parser.add_argument("--only_4_rotations", action='store_true', default=False,help="If not use rotation for class")
    return parser.parse_args()


args = get_args()

orig_stdout = sys.stdout
rand = np.random.randint(200000)

words = args.folder_txt_files.split('/')
args.folder_log = words[0]+'/'+words[1]+'/'+words[2]+'/'+'ROS/outputs/logs/' + str(rand)
args.folder_name = words[0]+'/'+words[1]+'/'+words[2]+'/'+'ROS/outputs/' + str(rand)
args.folder_txt_files_saving = args.folder_txt_files + str(rand)

gpu = str(args.gpu)
device = torch.device("cuda:"+gpu)

if not os.path.exists(args.folder_name):
    os.makedirs(args.folder_name)
    
print('\n')    
print('TRAIN START!')
print('\n')
print('THE OUTPUT IS SAVED IN A TXT FILE HERE -------------------------------------------> ', args.folder_name)
print('\n')

f = open(args.folder_name + '/out.txt', 'w')
sys.stdout = f
print("\n%s to %s - %d ss classes" % (args.source, args.target, args.ss_classes))

trainer = Trainer(args, device, rand)
trainer._do_train()

print(args)
sys.stdout = orig_stdout
f.close()
