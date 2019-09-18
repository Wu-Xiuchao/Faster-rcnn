import numpy as np
import sys,os
import argparse
import xml.etree.ElementTree as ET
import pickle
import numpy as np

def Iou(box,boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) 
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) 
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text # 拍摄角度
        obj_struct['truncated'] = int(obj.find('truncated').text) # 目标被挡住了
        obj_struct['difficult'] = int(obj.find('difficult').text) # 目标很难识别
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

"""
detpath:
        这是一个目录，目录包含21个类文件txt，每个文件的格式如下 [图片名,置信度,bbox]
annopath:
        就是xml那个文件目录
imagenames:
        就是图片的名字list
classname:
        要计算的类名字
cachedir:
        就是如果第一次计算好之后，后面可以直接用这个目录下的文件

"""
def compute_ap(detpath,annopath,imagenames,classname,cachedir,ovthresh=0.5,use_07_metric=False):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir,'annots.pkl')

    if not os.path.isfile(cachefile):
        recs = {}
        for i,imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('\rReading annotation for {}/{}'.format(i+1,len(imagenames)),end='')
        with open(cachefile,'wb') as f:
            pickle.dump(recs,f)
    else:
        with open(cachefile,'rb') as f:
            recs = pickle.load(f)

    class_recs = {} # 这个是ground_truth
    npos = 0 # 真正存在正类数量
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # 把一张图中存在classname的目标存起来
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox':bbox,'difficult':difficult,'det':det}

    detfile = detpath.format(classname) # 每一个类存一个文件
    with open(detfile,'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        #[image_ids, confidence, BB]一行
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines] # 就是图片名集合
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        sorted_ind = np.argsort(-confidence)
        sorted_score = np.sort(-confidence)
        BB = BB[sorted_ind,:]
        image_ids = [image_ids[x] for x in sorted_ind]

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d,:].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                overlaps = Iou(bb,BBGT)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1
                        R['det'][jmax] = 1 # 这个可能是标记的
                    else:
                        fp[d] = 1 # 冗余框
            else:
                fp[d] = 1

        fp = np.cumsum(fp) # 梯形累计和
        tp = np.cumsum(tp)
        rec = tp / float(npos) # 召回率
        prec = tp / np.maximum(tp+fp,np.finfo(np.float64).eps) # 防止除0
        ap = voc_ap(rec,prec,use_07_metric)
    else:
        rec = -1
        prec = -1
        ap = -1
    return rec,prec,ap