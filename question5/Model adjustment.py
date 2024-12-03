from ultralytics import YOLO
import torch.nn as nn

# Load a YOLO model with modified resolution and output size
model = YOLO('path/to/your/yolo_config.yaml').load('path/to/your/yolo_weights.pt')

model.model[0].conv1[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.model[0].maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

num_fruit_classes = 5
model.model[-1].anchor_num = num_fruit_classes
model.model[-1].num_classes = num_fruit_classes
model.model[-1].anchor_bias = torch.nn.Parameter(torch.zeros((num_fruit_classes * 3, 2)))

model = model.load('path/to/your/yolo_weights.pt')

model.train(data='path/to/your/fruits_dataset', epochs=..., imgsz=(270, 180))
