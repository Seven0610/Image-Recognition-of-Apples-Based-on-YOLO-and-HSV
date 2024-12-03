"""from ultralytics import YOLO
import numpy as np
#Load a model
#model = YOLO( ' voloy8n-cls.pt')# load an official model
model = YOLO('./runs/classify/train5/weights/best.pt')
#Predict with the model
results = model('F:/yolov8/ultralytics-main/dataset/test') # predict on an image
names_dict =results[0].names
probs = results[0].probs.data.tolist()
print(names_dict)
print(probs)
print(names_dict[np.argmax(probs)])"""
from ultralytics import YOLO
import os
import re
import numpy as np

# Load model
model = YOLO('runs/classify/train1/weights/best.pt')

# Specify the directory that contains the test images
test_dir = './dataset/test'

# Gets a list of all image files in the directory
image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sort image files numerically
image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

# Define a numeric dictionary for the category
class_dict = {0: 'Apple', 1: 'Carambola', 2: 'Pear', 3: 'Plum', 4: 'Tomatoes'}

# Store the category value corresponding to the highest probability of each image
predicted_classes = []

# Walk through each image and make predictions
for image_file in image_files:
    image_path = os.path.join(test_dir, image_file)
    results = model(image_path, save=True)
    
    # Extract the category name and probability of the first prediction
    probs = results[0].probs.data.tolist()

    # Find the index for the category with the highest probability
    max_prob_index = np.argmax(probs)

    # Adds the category value corresponding to the highest probability to the list
    predicted_classes.append(max_prob_index)

# Print the list of final results
print(predicted_classes)
