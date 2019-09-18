from model import faster_rcnn
import numpy as np
from class_name import className,className_CN
from PIL import Image,ImageDraw,ImageFont
import sys,os
from glob import glob
import argparse
from voc_eval import compute_ap
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default='voc_data/test.npy',help='The path of test.npy')
parser.add_argument('--model_path',type=str,default='faster_rcnn.npz',help='The path of model')
parser.add_argument('--num_test',type=int,default=10000,help='The number of images to compute mAP')
parser.add_argument('--pre_path',type=str,default='VOC2007_data/JPEGImages/',help='Pre path of images')
parser.add_argument('--annopath',type=str,default='VOC2007_data/Annotations/{}.xml',help='annopath xml file')
parser.add_argument('--cachedir',type=str,default='anno_cache',help='chache file for annotations')
parser.add_argument('--detpath',type=str,default='det',help='includes 21 txt files for each class')
args = parser.parse_args()


test_data = np.load(args.test_file)
img_path_list = []
gt_boxes_list = []
for item in test_data:
	img_path_list.append(args.pre_path+item['filename'])
	objects = item['object']
	gt_bboxes = item['gt_bbox']
	temp_list = []
	for ob,gt_bbox in zip(objects,gt_bboxes):
		label = className.index(ob)
		temp_list.append([gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3],label])
	gt_boxes_list.append(temp_list)

num_test = min(args.num_test,len(img_path_list))
img_path_list = img_path_list[:num_test]
gt_boxes_list = gt_boxes_list[:num_test]
print('Total images:',len(img_path_list))

fr = faster_rcnn()
# bboxes_list [x1,y1,x2,y2,label,prob]
bboxes_list = fr.test_model(img_path_list,args.model_path)

if os.path.exists(args.detpath) is False:
	os.mkdir(args.detpath)

file_map = {}
for item in className:
	if item == 'background':
		continue
	file_map[item] = open(os.path.join(args.detpath,item),'w')

imagenames = []
for idx,bboxes in enumerate(bboxes_list):
	img_id = os.path.split(img_path_list[idx])[-1].split('.')[0]
	imagenames.append(img_id)
	for item in bboxes:
		confidence = item[-1]
		clsname = className[int(item[-2])] 
		file_map[clsname].write('{} {} {} {} {} {}\n'.format(img_id,confidence,item[0],item[1],item[2],item[3]))

for item in className:
	if item == 'background':
		continue
	file_map[item].close()

res = [];mAP = 0
for item in className:
	if item == 'background':
		continue
	detpath = args.detpath + '/{}'
	annopath = args.annopath
	_,_,ap = compute_ap(detpath,annopath,imagenames,item,args.cachedir)
	mAP = mAP + ap/(len(className)-1)
	ap = int(ap * 1000)/1000
	res.append([item,ap])
res.append(['mAP',int(mAP*1000)/1000])

res = pd.DataFrame(res)
res.columns = ['class','AP']
res.to_csv('mAP.csv',index=False)








