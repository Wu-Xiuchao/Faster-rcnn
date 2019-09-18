import xml.sax
import numpy as np
import sys,os
from glob import glob
import argparse

class XmlHandler( xml.sax.ContentHandler ):
	def __init__(self):
		self.CurrentData = ""
		self.name = ""
		self.filename = ""
		self.xmin = ""
		self.ymin = ""
		self.xmax = ""
		self.ymax = ""
		self.bbox_list = []
		self.parse_result = {'filename':"",'object':[],'gt_bbox':[]}
		self.get = True
 
	def startElement(self, tag, attributes):
		self.CurrentData = tag
		if tag == 'part':
			self.get = False

	def endElement(self, tag):
		if tag == "object":
			self.parse_result['object'].append(self.name)
			self.parse_result['filename'] = self.filename
			self.parse_result['gt_bbox'].append(self.bbox_list)
			self.bbox_list = []

		if tag == "bndbox" and  self.get == True:
			self.bbox_list = [float(self.xmin),float(self.ymin),float(self.xmax),float(self.ymax)]

		if tag == 'part':
			self.get = True

		self.CurrentData = ""
 
	def characters(self, content):
		if self.get == False:
			return 
		if self.CurrentData == "name":
			self.name = content
		elif self.CurrentData == "filename":
			self.filename = content
		elif self.CurrentData == "xmin":
			self.xmin = content
		elif self.CurrentData == 'ymin':
			self.ymin = content
		elif self.CurrentData == 'xmax':
			self.xmax = content
		elif self.CurrentData == 'ymax':
			self.ymax = content

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path',type=str,default='VOC2007_data/Annotations',help='the file includes xml files')
parser.add_argument('--split_path',type=str,default='VOC2007_data/ImageSets/Main',help='the file includes train.txt val.txt text.txt')
parser.add_argument('--save_path',type=str,default='voc_data',help='the file save the output files')
args = parser.parse_args()

xml_list = glob(os.path.join(args.xml_path,'*.xml'))
sorted(xml_list)

Img_info = []
# create XMLReader
parser = xml.sax.make_parser()
# turn off namepsaces
parser.setFeature(xml.sax.handler.feature_namespaces, 0)

for xml_file in xml_list:
	Handler = XmlHandler()
	parser.setContentHandler(Handler)
	parser.parse(xml_file)
	Img_info.append(Handler.parse_result)

annotation = np.array(Img_info)
print('Finished parsing xml file')

with open(os.path.join(args.split_path,'train.txt'),'r') as file:
	train_data = file.readlines()
train_data = [item.strip('\n') for item in train_data]

with open(os.path.join(args.split_path,'val.txt'),'r') as file:
	val_data = file.readlines()
val_data = [item.strip('\n') for item in val_data]

with open(os.path.join(args.split_path,'test.txt'),'r') as file:
	test_data = file.readlines()
test_data = [item.strip('\n') for item in test_data]

train_val = []; test = []
for item in annotation:
	filename = item['filename'].split('.')[0];
	if filename in train_data or filename in val_data:
		train_val.append(item)
	else:
		test.append(item)

print('train_val: {}  test: {}'.format(len(train_val),len(test)))

if os.path.exists(args.save_path) is False:
	os.mkdir(args.save_path)
np.save(os.path.join(args.save_path,'train_val.npy'),np.array(train_val))
np.save(os.path.join(args.save_path,'test.npy'),np.array(test))
print('Finished save the ouput file in {}'.format(args.save_path))
