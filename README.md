# Faster RCNN  
**Most ways to implement faster rcnn are too complex, tens of thousands of lines of code dazzles us. So I tryed to implement faster rcnn by brief code. And meanwhile, making a more brief file architecture. Therefore, this code is really kind to the beginners.**
___
*ps: I refer to the [smallcorgi/Faster-RCNN_TF](https://github.com/rbgirshick/py-faster-rcnn ). And I only use python which means the roi-pooling part is replaced by the function of tensorflow called tf.image.crop_and_resize.*
___
So let's check the files include:  
```
class_name.py  # Include the class name list both in English and Chinese
model.py # The main file, include faster rcnn 
train_model.py # About train this model
test_model.py # About test this model
mAP.py # About compute mAP for one trained model
```