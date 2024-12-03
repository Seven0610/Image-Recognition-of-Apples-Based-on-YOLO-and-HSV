from ultralytics import YOLO
#Load a model
#model = YOLO( 'yoloy8n-cls.pt' )# load an official model
model = YOLO( 'F:\\yolov8\\ultralytics-main\\runs\classify\\train4\\weights\\best.pt')# load a custom model
#Validate the model
metrics = model.val()  # no anguments needed,dataset and settings remembered
metrics.top1  #top1 accuracy
metrics.top5  #top5 accuracy
