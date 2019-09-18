from model import faster_rcnn
import numpy as np
from class_name import className,className_CN
from PIL import Image,ImageDraw,ImageFont
import sys,os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default='test',help='The file whcih includes the test images')
parser.add_argument('--model_path',type=str,default='faster_rcnn.npz',help='The path of model')
parser.add_argument('--thresh',type=float,default=0.6,help='The thresh filter the pred bounding box')
parser.add_argument('--output_file',type=str,default='output',help='The file save the detected images')
parser.add_argument('--img_format',type=str,default='jpg',help='The format of test images')
parser.add_argument('--language',type=str,default='english',help='The language of label, only [chinese,english]')
parser.add_argument('--font_file',type=str,default='arial.ttf',help='If you choose chinese or you use run this code on mac, you may need download the font file')
args = parser.parse_args()

# The rectangle color for every class
colors = [(0,0,0),(0,191,255),(0,0,0),(139,69,19),(0,255,255),(127,255,212),(255,165,0),(131,111,255),(255,110,180),
(127,255,0),(128,128,255),(128,255,128),(255,128,128),(64,128,128),(128,64,128),(255,0,0),(128,128,192),(32,64,128),(128,64,32),
(32,64,32),(32,128,32)]


def draw_panel(draw,boxes,shape,label,prob,width,back_color,font_color,language,font_file,text_size=18):
	x1,y1,x2,y2 = boxes
	draw.line([x1,y1,x2,y1],fill=back_color,width=width)
	draw.line([x1,y1,x1,y2],fill=back_color,width=width)
	draw.line([x1,y2,x2,y2],fill=back_color,width=width)
	draw.line([x2,y1,x2,y2],fill=back_color,width=width)

	if language == 'chinese':
		position = [crop_box[0]-1,crop_box[1]-10,crop_box[0]+len(className_CN[label])*text_size + 52,crop_box[1]+10]
	else:
		position = [crop_box[0]-1,crop_box[1]-10,crop_box[0]+len(className[label])*int(text_size/2) + 52,crop_box[1]+10]

	if position[0] < 0:
		off_set = 0 - position[0]
		position[0] += off_set; position[2] += off_set
	if position[1] < 0:
		off_set = 0 - position[1]
		position[1] += off_set; position[3] += off_set
	if position[2] >= shape[0]:
		off_set = position[2] - shape[0]
		position[0] -= off_set; position[2] -= off_set
	if position[3] >= shape[1]:
		off_set = position[3] - shape[1]
		position[1] -= off_set; position[3] -= off_set

	draw.rectangle(position,fill=back_color)
	font = ImageFont.truetype(font_file, text_size)
	if language == 'chinese':
		draw.text((position[0]+2,position[1]-3),u'{} {}'.format(className_CN[label],prob),fill=font_color,font=font)
	else:
		draw.text((position[0]+2,position[1]-3),'{} {}'.format(className[label],prob),fill=font_color,font=font)


img_path_list = glob(os.path.join(args.test_file,'*.{}'.format(args.img_format)))
if os.path.exists(args.output_file) is False:
	os.mkdir(args.output_file)

fr = faster_rcnn()
bboxes_list = fr.test_model(img_path_list,args.model_path)

for idx,bboxes in enumerate(bboxes_list):
	img_path = img_path_list[idx]
	img = Image.open(img_path)
	draw = ImageDraw.Draw(img)
	for box in bboxes:
		crop_box = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
		label = int(box[4]); prob = int(box[5] * 100)/100
		if prob < args.thresh:
			continue
		draw_panel(draw=draw,shape=img.size,boxes=crop_box,label=label,prob=prob, width=3,back_color=colors[label],
			font_color=(255,255,255),language=args.language,font_file=args.font_file)
	img.save(os.path.join(args.output_file,'{}_detected.png'.format(os.path.split(img_path)[-1].split('.')[0])))
