from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # If you have a GPU, use a GPU, if you don't, use a CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device= 'cpu'
print('device:', device)
#####The apples are tested#######
model = YOLO('yolov8n.pt')
# Switching computing equipment
model.to(device)
# model.cpu()  #CPU
# model.cuda() #GPU
model.device
model.names
# The last picture in the first question is 200
ddl = 200
y = []


def process_img(i, img_path, na, y):
    # Used to count the number of apples in each picture
    img = cv2.imread(img_path)
    results = model(img)

    # Get the number of apples
    num_apples = len(results.xyxy[results.xyxy[:, -1] == 0])  # Let's assume that the category index for apples is 0

    # Add the result to the y list
    y.append(num_apples)


for i in range(ddl):
    num = i + 1
    num = str(num)
    f = 'detect_data/'
    d = '.jpg'
    name = f + num + d
    na = num + d
    # name= "detect_data\\1.jpg"
    process_img(i, name, na, y)

x = list(range(1, ddl + 1))
# create
plt.bar(x, y, color='blue')
# Add titles and labels
plt.title('number of apple')
plt.xlabel('x')
plt.ylabel('number of apple')  # Display graphics
plt.show()

# test_version...
"""
num=1
num=str(num)
a='\''
f='detect_data\\'
d='.jpg'
name=f+num+d
na=num+d
#name= 'detect_data\\1.jpg'
# process_img(name,na)
"""
print('pricture has finished')
print(y)