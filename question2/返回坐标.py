import torch
import numpy as np
# Loading model
model = torch.hub.load ( 'ultralytics/yolov8' , 'yolov8s ', pretrained=True)
# The size of the model input image
img_width, img_height = 640,640
# The input image is image and the type is numpy array
image = np.random.rand( img_height, img_width, 3)
# The image is converted to PyTorch tensor and normalized
image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
# The image is fed into the model for detection
detections = model( image.unsqueeze(0))
# The bounding boxes for each target are extracted from the detection results
for detection in detections.xyxy[0]:
      # Gets the top-left and bottom-right coordinates of the bounding box
    x_min, y_min, x_max, y_max, confidence, class_id = detection.tolist()
    # The coordinates are scaled to convert to the coordinates of the original image
    x_min = int(x_min * img_width)
    y_min = int(y_min * img_height)
    x_max = int( x_max * img_width)
    y_max = int(y_max * img_height)
    # Stores the bounding box and category information for the target
    detection_result.append({'x_min' : x_min, 'y_min': y_min, 'x_max': x_max,'y_max' : y_max, 'class_id' : class_id})
