from model import faster_rcnn
import numpy as np
from class_name import className
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_val',type=str,default='voc_data/train_val.npy',help='The path of train_val.npy')
parser.add_argument('--pre_path',type=str,default='VOC2007_data/JPEGImages/',help='Pre path of images')
parser.add_argument('--pre_train',type=str,default='VGG_imagenet.npz',help='The pre_trained model path')
parser.add_argument('--max_step',type=int,default=70000,help='The max train steps')
parser.add_argument('--save_step',type=int,default=5000,help='Save the model every * steps')
parser.add_argument('--base_eta',type=float,default=0.0005,help='The basic learning rate')
parser.add_argument('--decay_steps',type=int,default=50000,help='current learning rate = base_eta * 0.1 ^ (current_steps/decay_steps)')
args = parser.parse_args()

train_data = np.load(args.train_val)
img_path_list = []
gt_boxes_list = []
for item in train_data:
	img_path_list.append(args.pre_path+item['filename'])
	objects = item['object']
	gt_bboxes = item['gt_bbox']
	temp_list = []
	for ob,gt_bbox in zip(objects,gt_bboxes):
		label = className.index(ob)
		temp_list.append([gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3],label])
	gt_boxes_list.append(temp_list)

fr = faster_rcnn()
MAX_STEP = args.max_step
pre_train = args.pre_train

fr.train_model(MAX_STEP=MAX_STEP,img_path_list=img_path_list,gt_boxes_list=gt_boxes_list,pre_train=pre_train,
	save_step=args.save_step,base_eta=args.base_eta,decay_steps=args.decay_steps)

