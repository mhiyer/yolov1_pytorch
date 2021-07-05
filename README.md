# yolov1_pytorch
Train and test YOLOv1 in PyTorch  
  
This code is a combination of two excellent YOLO-v1 in PyTorch repositories:  
1. https://github.com/motokimura/yolo_v1_pytorch  
2. https://github.com/abeardear/pytorch-YOLO-v1  

Specifically, the model training and testing procedure comes from (1),  
and model backbone from (2)- I modified the last few layers to enable a final feature map of (7,7) instead of (14,14).  
The model backbone is ResNet50.  
  
There are some modifications in terms of recording train and test history.  
Use train.py to train, and detect.py to perform detection with desired images.  

![alt text](https://github.com/mhiyer/yolov1_pytorch/main/result.jpg?raw=true)

