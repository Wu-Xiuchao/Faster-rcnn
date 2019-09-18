import tensorflow as tf 
import numpy as np 
from scipy.misc import imread, imresize
import math,cv2,os
import random
import matplotlib.pyplot as plt

class faster_rcnn(object):
	def __init__(self):
		self.paras = [] # The all variables that need to be saved
		self.train_variables = [] # The variables that need to be trained
		# anchor type
		self.anchor_type = np.array([[ -83.,  -39.,  100.,   56.],[-175.,  -87.,  192.,  104.],[-359., -183.,  376.,  200.],
									 [ -55.,  -55.,   72.,   72.],[-119., -119.,  136.,  136.],[-247., -247.,  264.,  264.],
									 [ -35.,  -79.,   52.,   96.],[ -79., -167.,   96.,  184.],[-167., -343.,  184.,  360.]])
	"""train model end2end
	paras:
		MAX_STEP: num steps of training 
		img_path_list: the list includes all the path of train-images
		gt_boxes_list: bounding box list, the shape is [num_images,num_boxes_per_img,5],and for each box is [x1,y1,x2,y2,label]
		pre_train: pre_train model path,default is none; You should use vgg16 pretrained paras if you train this model for the first time
		save_step: Save the model ecery save_step
		log_inter: log the loss info every log_inetr steps
		base_eta: the base learning rate set for the optimizer
		decay_steps: every decay_steps step, the learning_rate will times 0.1
	"""
	def train_model(self,MAX_STEP,img_path_list,gt_boxes_list,pre_train=None,save_step=10000,log_inter=10,base_eta=1e-3,decay_steps=50000):

		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
		self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
		self.network(self.data,self.im_info,self.gt_boxes,'TRAIN')
		print('\nFinished Graph-construction\n')

		# RPN cls Loss
		rpn_cls_score = tf.reshape(self.output['rpn_cls_score_reshape'],[-1,2])
		rpn_labels = tf.reshape(self.output['rpn_labels'],[-1])

		# select rpn_cls_score whose rpn_labels that not equal to -1
		rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_labels,-1))),[-1,2]) 
		rpn_labels = tf.reshape(tf.gather(rpn_labels,tf.where(tf.not_equal(rpn_labels,-1))),[-1])
		# sparse_softmax_cross_entropy_with_logits 
		rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))

		# RPN regression loss
		rpn_bbox_pred = self.output['rpn_bbox_pred']
		rpn_bbox_targets = tf.transpose(self.output['rpn_bbox_targets'],[0,2,3,1]) # [1,36,height,width] -> [1,height,width,36]
		rpn_bbox_inside_weights = tf.transpose(self.output['rpn_bbox_inside_weights'],[0,2,3,1]) # [1,36,height,width] -> [1,height,width,36]
		rpn_bbox_outside_weights = tf.transpose(self.output['rpn_bbox_outside_weights'],[0,2,3,1]) # [1,36,height,width] -> [1,height,width,36]
		rpn_smooth_l1 = _modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
		rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

		# RCNN cls loss
		cls_score = self.output['cls_score']
		labels = tf.reshape(self.output['labels'],[-1])
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels))

		# RCNN regression loss
		bbox_pred = self.output['bbox_pred']
		bbox_targets = self.output['bbox_targets']
		bbox_inside_weights = self.output['bbox_inside_weights']
		bbox_outside_weights = self.output['bbox_outside_weights']
		smooth_l1 = _modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
		loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

		# total loss
		loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

		# optimizer
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(base_eta,global_step,decay_steps, 0.1, staircase=True)
		momentum = 0.9
		train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, var_list=self.train_variables ,global_step=global_step)

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		if pre_train is not None:
			self._load_para(pre_train,sess)

		def get_batch():
			current_index = 0
			while True:
				_path = img_path_list[current_index]
				_gt_boxes = gt_boxes_list[current_index]
				img_batch,im_info_batch,gt_boxes_batch = _loadImg(_path,_gt_boxes)
				current_index = (current_index + 1) % len(img_path_list)
				yield img_batch,im_info_batch,gt_boxes_batch

		step = 0
		loss_list = []
		for img_batch,im_info_batch,gt_boxes_batch in get_batch():
			step += 1
			rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
																			feed_dict={self.data:img_batch,self.im_info:im_info_batch,self.gt_boxes:gt_boxes_batch})
			total_loss = rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value
			loss_list.append(total_loss)

			if step % log_inter == 0:
				print('step: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f'%\
		                (step, MAX_STEP, total_loss ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value))

			if step % save_step == 0:
				self._save_para(sess,'faster_rcnn-{}.npz'.format(step))
				draw_list = [np.mean(loss_list[i:min(i+log_inter,len(loss_list))]) for i in range(0,len(loss_list),log_inter)]
				plt.clf()
				plt.figure(figsize=(20,8))
				l1, = plt.plot(draw_list,linewidth=5,label='total_loss')
				plt.xticks(fontsize=20)
				plt.yticks(fontsize=20)
				plt.xlabel('{}Step'.format(log_inter),fontsize=20)
				plt.ylabel('Loss',fontsize=20)
				plt.ylim((0, 1.6)) 
				plt.legend(loc='upper right',fontsize=20)
				plt.title('Faster-Rcnn Loss Figure (by show)',fontsize=25)
				plt.grid(True, linestyle = "-.",linewidth = 5)
				plt.savefig('FasterRcnn_Loss.eps',format='eps')
			if step >= MAX_STEP:
				self._save_para(sess,'faster_rcnn-{}.npz'.format(step))
				print('\nFinished')
				break
		sess.close()

	"""test model
	paras:
		img_path_list: includes the path you want to test
		model_path: the path of the model
	"""
	def test_model(self,img_path_list,model_path,thresh=0.05,nms_th=0.3,nms_mode='Union'):
		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
		self.network(self.data,self.im_info,net_type='TEST')
		print('\nFinished Graph-construction\n')

		sess = tf.Session()
		self._load_para(model_path,sess)

		bboxes_list = []
		for index,img_path in enumerate(img_path_list):
			base_size = 600
			img = imread(img_path,mode='RGB')
			img = img / 255. - 1
			h = img.shape[0]; w = img.shape[1]
			ts = min(w,h)/base_size
			w = int(math.ceil(w/ts)); h = int(math.ceil(h/ts))
			img = [imresize(img,(h,w))] # scipy (h,w)
			im_info = np.array([[h,w,ts]]).astype('float')

			cls_prob = self.output['cls_prob']
			bbox_pred = self.output['bbox_pred']
			rois = self.output['rois']

			cls_prob_value,bbox_pred_value,rois_value = sess.run([cls_prob,bbox_pred,rois],feed_dict={self.data:img,self.im_info:im_info})
			scores = np.array(cls_prob_value); box_deltas = np.array(bbox_pred_value); rois_value = np.array(rois_value)
			boxes = bbox_transform_inv(rois_value[:,1:5]*im_info[0][-1],box_deltas)
			boxes = clip_boxes(boxes,im_info[0][:2]*im_info[0][-1])

			bboxes = []
			for i in range(1,21):
				inds = np.where(scores[:,i] > thresh)[0]
				if len(inds) < 1:
					continue
				cls_score = scores[inds,i]
				cls_boxes = boxes[inds,i*4:i*4+4]
				keep = _nms(np.hstack([cls_boxes,cls_score[:,np.newaxis]]),nms_th,nms_mode)
				cls_score = cls_score[keep]
				cls_boxes = cls_boxes[keep,:]
				cls_dets = np.hstack([cls_boxes,np.ones((cls_boxes.shape[0],1))*i,cls_score[:,np.newaxis]])
				bboxes.extend(cls_dets)
			bboxes_list.append(bboxes)
			print('\r Finished testing {}/{}'.format(index,len(img_path_list)),end='')
		return bboxes_list	

	"""Faster rcnn network
	paras:
		data: the images [None,None,None,3]
		im_info: about (width,height,scale) [None,3]
		gt_boxes: (x1,y1,x2,y2,label)  [None,5]
		net_type: ['TRAIN','TEST']
	"""
	def network(self,data,im_info,gt_boxes=None,net_type='TRAIN'):
		#========== Shared ConvNet =========
		net = self.conv_layer(data,'conv1_1',3,3,64,trainable=False)
		net = self.conv_layer(net,'conv1_2',3,64,64,trainable=False)
		net = tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool1')

		net = self.conv_layer(net,'conv2_1',3,64,128,trainable=False)
		net = self.conv_layer(net,'conv2_2',3,128,128,trainable=False)
		net = tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')

		net = self.conv_layer(net,'conv3_1',3,128,256)
		net = self.conv_layer(net,'conv3_2',3,256,256)
		net = self.conv_layer(net,'conv3_3',3,256,256)
		net = tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool3')

		net = self.conv_layer(net,'conv4_1',3,256,512)
		net = self.conv_layer(net,'conv4_2',3,512,512)
		net = self.conv_layer(net,'conv4_3',3,512,512)
		net = tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool4')

		net = self.conv_layer(net,'conv5_1',3,512,512)
		net = self.conv_layer(net,'conv5_2',3,512,512)
		feature_map = self.conv_layer(net,'conv5_3',3,512,512)

		#========== RPN ==========
		rpn_conv = self.conv_layer(feature_map,'rpn_conv/3x3',3,512,512)
		rpn_cls_score = self.conv_layer(rpn_conv,'rpn_cls_score',1,512,9*2,padding='VALID',if_relu=False)
		rpn_bbox_pred = self.conv_layer(rpn_conv,'rpn_bbox_pred',1,512,9*4,padding='VALID',if_relu=False)

		# output
		if net_type == 'TRAIN':
			rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = self.rpn_data(rpn_cls_score,gt_boxes,im_info,data,'rpn_data')

		#========== ROI proposals ==========
		rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score,2,'rpn_cls_score_reshape') # [1,height,width,18] -> [1,9*height,width,2]
		rpn_cls_prob = self.softmax(rpn_cls_score_reshape,'rpn_cls_prob') # [1,9*height,width,2]
		rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob,18,'rpn_cls_prob_reshape') # [1,9*height,width,2] -> [1,height,width,18]

		# output
		rpn_rois = self.proposal_layer(rpn_cls_prob_reshape,im_info,rpn_bbox_pred,net_type,'proposal_layer')

		# output
		if net_type == 'TRAIN':
			rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = self.proposal_target_layer(rpn_rois,gt_boxes,'proposal_target_layer')
		elif net_type == 'TEST':
			rois = rpn_rois

		#========== RCNN ==========
		# output
		pool5 = self.roi_pool(feature_map,rois,im_info,'pool5')
		net = self.fc_layer(tf.layers.flatten(pool5),'fc6',7*7*512,4096)
		if net_type == 'TRAIN':
			net = tf.nn.dropout(net,keep_prob=0.5,name='drop6')
		net = self.fc_layer(net,'fc7',4096,4096)
		if net_type == 'TRAIN':
			net = tf.nn.dropout(net,keep_prob=0.5,name='drop7')
		cls_score = self.fc_layer(net,'cls_score',4096,21,if_relu=False)
		bbox_pred = self.fc_layer(net,'bbox_pred',4096,84,if_relu=False)

		if net_type == 'TRAIN':
			self.output = {'rpn_cls_score_reshape':rpn_cls_score_reshape,'rpn_bbox_pred':rpn_bbox_pred,'rpn_labels':rpn_labels,
					  'rpn_bbox_targets':rpn_bbox_targets,'rpn_bbox_inside_weights':rpn_bbox_inside_weights,
			 		  'rpn_bbox_outside_weights':rpn_bbox_outside_weights,'cls_score':cls_score,'bbox_pred':bbox_pred,
			 		  'labels':labels,'bbox_targets':bbox_targets,'bbox_inside_weights':bbox_inside_weights,
			 		  'bbox_outside_weights':bbox_outside_weights,'rois':rois}
		elif net_type == 'TEST':
			cls_prob = self.softmax(cls_score,'cls_prob')
			self.output = {'cls_prob':cls_prob,'bbox_pred':bbox_pred,'rois':rois}

	# save model paras
	def _save_para(self,sess,save_path):
		vals_to_save = {}
		for each in self.paras:
			value = np.array(each.eval(session=sess))
			key = each.name
			vals_to_save[key] = value
		np.savez(save_path,**vals_to_save)

	# load model paras
	def _load_para(self,file,sess):
		print('\nLoad paras...')
		para = np.load(file)
		keys = sorted(para.keys())
		print(keys)
		variables = tf.global_variables()
		# print(variables)
		for k in keys:
			for i in range(len(variables)):
				if variables[i].name == k:
					print('Finished loading {}'.format(k))
					sess.run(variables[i].assign(para[k]))
		print('Load paras successfully!\n')

	"""ROI pooling
	This code is from [https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Lib/roi_pool.py]
	"""
	def roi_pool(self,feature_map,rois,im_info,name):    
	    '''
	    Regions of Interest (ROIs) from the Region Proposal Network (RPN) are 
	    formatted as:
	    (image_id, x1, y1, x2, y2)
	    
	    Note: Since mini-batches are sampled from a single image, image_id = 0s
	    '''
	    with tf.variable_scope('roi_pool'):
	        # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
	        box_ind = tf.cast(rois[:,0],dtype=tf.int32)
	        
	        # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
	        boxes = rois[:,1:]
	        normalization = tf.cast(tf.stack([im_info[:,1],im_info[:,0],im_info[:,1],im_info[:,0]],axis=1),dtype=tf.float32)
	        boxes = tf.div(boxes,normalization)
	        boxes = tf.stack([boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2]],axis=1)  # y1, x1, y2, x2
	        
	        # ROI pool output size
	        crop_size = tf.constant([14,14])
	        
	        # ROI pool
	        pooledFeatures = tf.image.crop_and_resize(image=feature_map, boxes=boxes, box_ind=box_ind, crop_size=crop_size)
	        
	        # Max pool to (7x7)
	        pooledFeatures = tf.nn.max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	    return pooledFeatures

	def proposal_target_layer(self,rpn_rois,gt_boxes,name):
		with tf.variable_scope(name) as scope:
			rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(self.OP_proposal_target_layer,[rpn_rois,gt_boxes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
			rois = tf.reshape(rois,[-1,5] , name = 'rois') 
			labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
			bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
			bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
			bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')
		return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

	def proposal_layer(self,rpn_cls_prob_reshape,im_info,rpn_bbox_pred,net_type,name):
		if net_type == 'TRAIN':
			rpn_rois = tf.reshape(tf.py_func(self.OP_proposal_layer,[rpn_cls_prob_reshape,im_info,rpn_bbox_pred,12000,2000], [tf.float32]),[-1,5],name = name)
		else:
			rpn_rois = tf.reshape(tf.py_func(self.OP_proposal_layer,[rpn_cls_prob_reshape,im_info,rpn_bbox_pred,6000,300], [tf.float32]),[-1,5],name = name)
		return rpn_rois

	def rpn_data(self,rpn_cls_score,gt_boxes,im_info,data,name):
		with tf.variable_scope(name) as scope:
			rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(self.OP_rpn_data,[rpn_cls_score,gt_boxes,im_info,data],[tf.float32,tf.float32,tf.float32,tf.float32])
			rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
			rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
			rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
			rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')
		return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights

	def reshape_layer(self,input,d,name):
		input_shape = tf.shape(input)
		if name == 'rpn_cls_prob_reshape':
			return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
				int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
		else:
			return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
				int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
    
	def softmax(self, input, name):
		input_shape = tf.shape(input)
		if name == 'rpn_cls_prob':
			return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
		else:
			return tf.nn.softmax(input,name=name)

	def conv_layer(self,inputs,name,ksize,fileter_in,filter_out,padding='SAME',if_relu=True,trainable=True):
		with tf.variable_scope(name) as scope:
			W = tf.get_variable('W',[ksize,ksize,fileter_in,filter_out],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b = tf.get_variable('b',[filter_out],initializer=tf.constant_initializer(0.0))
			conv = tf.nn.conv2d(inputs,W,[1,1,1,1],padding=padding)
			out = tf.nn.bias_add(conv, b)
		if trainable is True:
			self.train_variables += [W,b]
		self.paras += [W,b]
		if if_relu == True:
			return tf.nn.relu(out)
		else:
			return out

	def fc_layer(self,inputs,name,units_in,units_out,if_relu=True, trainable=True):
		with tf.variable_scope(name) as scope:
			if name == 'bbox_pred':
				W = tf.get_variable('W',[units_in,units_out],initializer=tf.truncated_normal_initializer(0.0, stddev=0.001))
			else:
				W = tf.get_variable('W',[units_in,units_out],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b = tf.get_variable('b',[units_out],initializer=tf.constant_initializer(0.0))
			out = tf.nn.bias_add(tf.matmul(inputs,W),b)
		if trainable is True:
			self.train_variables += [W,b]
		self.paras += [W,b]
		if if_relu == True:
			return tf.nn.relu(out)
		else:
			return out

	def OP_rpn_data(self,rpn_cls_score,gt_boxes,im_info,data,feature_stride=16):

		_anchors = self.anchor_type.copy()
		height, width = rpn_cls_score.shape[1:3]

		im_info = im_info[0]

		shift_x = np.arange(0,width) * feature_stride
		shift_y = np.arange(0,height) * feature_stride
		shift_x,shift_y = np.meshgrid(shift_x,shift_y) 
		shifts = np.vstack([shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel()]).transpose() # [k,4]

		# generate all anchors
		A = _anchors.shape[0]
		K = shifts.shape[0]
		all_anchors = (_anchors.reshape((1,A,4))+shifts.reshape((1,K,4)).transpose((1,0,2)))
		all_anchors = all_anchors.reshape((K*A,4))
		total_anchors = int(K * A)

		# remove the cross boundary anchors
		keep_idx = np.where((all_anchors[:,0] >= 0) & (all_anchors[:,1] >= 0) & (all_anchors[:,2] < im_info[1]) & (all_anchors[:,3] < im_info[0]))[0]
		anchors = all_anchors[keep_idx,:] # shape [keep_num,4]

		# label: 1 pos , 0 neg, -1 useless
		labels = np.zeros((len(keep_idx), ), dtype=np.float32) - 1

		overlaps = np.array([Iou(anchor,gt_boxes) for anchor in anchors])
		argmax_overlaps = overlaps.argmax(axis=1) 
		max_overlaps = overlaps.max(axis=1) 
		gt_max_overlaps = overlaps.max(axis=0)
		gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

		labels[gt_argmax_overlaps] = 1
		labels[max_overlaps >= 0.7] = 1
		labels[max_overlaps < 0.3] = 0
		pos_idx = np.where(labels==1)[0]
		# subsample pos
		if len(pos_idx) > 128:
			drop_idx = np.random.choice(pos_idx,len(pos_idx)-128,replace=False)
			labels[drop_idx] = -1
		num_neg = 256 - np.sum(labels==1)
		neg_idx = np.where(labels==0)[0]
		# subsample neg
		if len(neg_idx) > num_neg:
			drop_idx = np.random.choice(neg_idx,len(neg_idx)-num_neg,replace=False)
			labels[drop_idx] = -1

		bbox_targets = np.zeros((len(keep_idx), 4), dtype=np.float32)
		bbox_targets = bbox_transform(anchors,gt_boxes[argmax_overlaps,:]).astype(np.float32,copy=False)

		bbox_inside_weights = np.zeros((len(keep_idx), 4), dtype=np.float32) # if pos
		bbox_inside_weights[labels == 1, :] = np.array([1.,1.,1.,1.]) # if is pos, then set [1,1,1,1]

		bbox_outside_weights = np.zeros((len(keep_idx), 4), dtype=np.float32) # if pos and neg, means that if it would be computed in loss

		num_examples = np.sum(labels>=0)
		positive_weights = np.ones((1,4)) * 1./num_examples
		negative_weights = np.ones((1,4)) * 1./num_examples
		bbox_outside_weights[labels==1,:] = positive_weights
		bbox_outside_weights[labels==0,:] = negative_weights

		labels = _unmap(labels, total_anchors, keep_idx, fill=-1)
		bbox_targets = _unmap(bbox_targets, total_anchors, keep_idx, fill=0)
		bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, keep_idx, fill=0)
		bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, keep_idx, fill=0)

		labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
		labels = labels.reshape((1, 1, A * height, width)) # [1,1,9*height,width]
		rpn_labels = labels

		bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2) # [1,36,height,width]
		rpn_bbox_targets = bbox_targets

		bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2) # [1,36,height,width]
		rpn_bbox_inside_weights = bbox_inside_weights

		bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2) # [1,36,height,width]
		rpn_bbox_outside_weights = bbox_outside_weights

		return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights

	def OP_proposal_layer(self,rpn_cls_prob_reshape,im_info,rpn_bbox_pred,pre_nms_topN,post_nms_topN,feature_stride=16):
		_anchors = self.anchor_type.copy()
		_num_anchors = _anchors.shape[0]
		rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2]) # [1,height,width,18] -> [1,18,height,width]
		rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2]) # [1,height,width,36] -> [1,36,height,width]

		im_info = im_info[0]

		nms_thresh = 0.7
		min_size = 16

		scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :] # former 9 is the probs of bg, later 9 is the probs of fg
		bbox_deltas = rpn_bbox_pred # targets [1,36,h,w]

		height, width = scores.shape[-2:]

		shift_x = np.arange(0,width) * feature_stride
		shift_y = np.arange(0,height) * feature_stride
		shift_x,shift_y = np.meshgrid(shift_x,shift_y) # gen coordinates
		shifts = np.vstack([shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel()]).transpose() 

		A = _anchors.shape[0]
		K = shifts.shape[0]
		anchors = (_anchors.reshape((1,A,4))+shifts.reshape((1,K,4)).transpose((1,0,2)))
		anchors = anchors.reshape((K*A,4))

		bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4)) # [1,36,h,w]->[1,h,w,36]->[h*w*9,4]
		scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1)) # [1,9,h,w]->[1,h,w,9]->[h*w*9,1]

		# to prevent overflow when computing bbox_transform_inv, Cuz dw,dh is too large
		keep = list(set(np.where( (bbox_deltas[:,2] < 20) & (bbox_deltas[:,3] < 20) )[0]))
		bbox_deltas = bbox_deltas[keep]
		scores = scores[keep]
		anchors = anchors[keep]

		proposals = bbox_transform_inv(anchors, bbox_deltas)
		proposals = clip_boxes(proposals, im_info[:2])

		# proposals is for feature_map, so it need to times im_info[2],which is scale 
		keep = _filter_boxes(proposals, min_size * im_info[2])
		proposals = proposals[keep, :]
		scores = scores[keep]

		order = scores.ravel().argsort()[::-1]
		order = order[:pre_nms_topN]
		proposals = proposals[order,:]
		scores = scores[order]

		keep = _nms(np.hstack((proposals, scores)), nms_thresh)
		keep = keep[:post_nms_topN]
		proposals = proposals[keep,:]
		scores = scores[keep]

		# [0,tx,ty,tw,th]
		batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
		blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

		return blob

	def OP_proposal_target_layer(self,rpn_rois,gt_boxes,_num_classes=21):
		all_rois = rpn_rois # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
		zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
		all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1])))) # put gt_boxes to the candidate rois

		rois_per_image = 128 # total num to train per img
		fg_rois_per_image = 32 # fg_num to train per img

		labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(all_rois, gt_boxes, fg_rois_per_image,rois_per_image, _num_classes)

		rois = rois.reshape(-1,5)
		labels = labels.reshape(-1,1)
		bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
		bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)
		bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32) 

		return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

#============================ function lib ==========================================//////

"""compute Iou
paras:
	box: single box[x1,y1,x2,y2]
	boxes: multi box shape[num,4]
"""
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

# get the [tx,ty,tw,th]
def bbox_transform(ex_rois, gt_rois):
	ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
	ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
	ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
	ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

	gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
	gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
	gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
	gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

	targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
	targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
	targets_dw = np.log(gt_widths / ex_widths)
	targets_dh = np.log(gt_heights / ex_heights)

	targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
	return targets

def _unmap(data, count, inds, fill=0):
	""" Unmap a subset of item (data) back to the original set of items (of
	size count) """
	if len(data.shape) == 1:
		ret = np.empty((count, ), dtype=np.float32)
		ret.fill(fill)
		ret[inds] = data
	else:
		ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
		ret.fill(fill)
		ret[inds, :] = data
	return ret

def _filter_boxes(boxes, min_size):
	"""Remove all boxes with any side smaller than min_size."""
	ws = boxes[:, 2] - boxes[:, 0] + 1
	hs = boxes[:, 3] - boxes[:, 1] + 1
	keep = np.where((ws >= min_size) & (hs >= min_size))[0]
	return keep

# get the final box from the pred boxes and targets(deltas)
def bbox_transform_inv(boxes, deltas):
	if boxes.shape[0] == 0:
		return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

	boxes = boxes.astype(deltas.dtype, copy=False)

	widths = boxes[:, 2] - boxes[:, 0] + 1.0
	heights = boxes[:, 3] - boxes[:, 1] + 1.0
	ctr_x = boxes[:, 0] + 0.5 * widths
	ctr_y = boxes[:, 1] + 0.5 * heights

	dx = deltas[:, 0::4]
	dy = deltas[:, 1::4]
	dw = deltas[:, 2::4]
	dh = deltas[:, 3::4]

	pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
	pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
	pred_w = np.exp(dw) * widths[:, np.newaxis]
	pred_h = np.exp(dh) * heights[:, np.newaxis]

	pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
	# x1
	pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
	# y1
	pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
	# x2
	pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
	# y2
	pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

	return pred_boxes

def clip_boxes(boxes, im_shape):
	"""
	Clip boxes to image boundaries.
	"""

	# x1 >= 0
	boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
	# y1 >= 0
	boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
	# x2 < im_shape[1]
	boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
	# y2 < im_shape[0]
	boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
	return boxes

def _nms(dets,thresh, mode="Union"):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		if mode == "Union":
		    ovr = inter / (areas[i] + areas[order[1:]] - inter)
		elif mode == "Minimum":
		    ovr = inter / np.minimum(areas[i], areas[order[1:]])
		#keep
		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]
	return keep

def _get_bbox_regression_labels(bbox_target_data, num_classes):
	"""Bounding-box regression targets (bbox_target_data) are stored in a
	compact form N x (class, tx, ty, tw, th)

	This function expands those targets into the 4-of-4*K representation used
	by the network (i.e. only one class has non-zero targets).

	Returns:
	    bbox_target (ndarray): N x 4K blob of regression targets
	    bbox_inside_weights (ndarray): N x 4K blob of loss weights
	"""
	clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
	bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
	bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
	inds = np.where(clss > 0)[0] # the foreground's index
	for ind in inds:
		item = clss[ind]
		start = 4 * item
		end = start + 4
		bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
		bbox_inside_weights[ind, start:end] = [1.,1.,1.,1.]
	return bbox_targets, bbox_inside_weights

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
	overlaps = np.array([Iou(roi[1:5],gt_boxes[:4]) for roi in all_rois])
	gt_assignment = overlaps.argmax(axis=1)
	max_overlaps = overlaps.max(axis=1)
	labels = gt_boxes[gt_assignment, 4]

	fg_inds = np.where(max_overlaps >= 0.5)[0] # foreground inds
	fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
	if fg_inds.size > 0:
		fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False) # sumsample fg

	bg_inds = np.where((max_overlaps < 0.5) & (max_overlaps >= 0.1))[0] # background inds
	bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
	bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
	if bg_inds.size > 0:
		bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

	keep_inds = np.append(fg_inds,bg_inds)
	labels = labels[keep_inds]
	labels[fg_rois_per_this_image:] = 0 # set the background to zero

	rois = all_rois[keep_inds]
	targets = bbox_transform(rois[:,1:5], gt_boxes[gt_assignment[keep_inds],:4])
	bbox_target_data = np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False) # N x [class,tx,ty,th,tw]

	bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)
	return labels, rois, bbox_targets, bbox_inside_weights

# Smooth_l1
def _modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
	"""
	    ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
	    SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
	                  |x| - 0.5 / sigma^2,    otherwise
	"""
	sigma2 = sigma * sigma

	inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

	smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
	smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
	smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
	smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
	                          tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

	outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

	return outside_mul

# load an image
def _loadImg(path,gt_bbox,base_size=600):
	gt_bbox = np.array(gt_bbox) #(shape numx5)
	img = imread(path,mode='RGB')
	img = img / 255. - 1
	h = img.shape[0]; w = img.shape[1]
	ts = min(w,h)/base_size
	w = int(math.ceil(w/ts)); h = int(math.ceil(h/ts))
	gt_bbox = np.hstack([np.ceil(gt_bbox[:,:4]/ts),gt_bbox[:,4::4]])
	gt_bbox = gt_bbox.astype('int')
	img = [imresize(img,(h,w))] 
	im_info = np.array([[h,w,ts]]).astype('float')
	return img,im_info,gt_bbox





