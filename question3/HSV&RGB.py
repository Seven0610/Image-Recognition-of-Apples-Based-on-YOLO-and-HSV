import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os

# Loading Datasets
datadir = r"F:\code\detect_data"
# 1: High Rigeness: High rigeness
# 2: Low Rigeness: Low rigeness
# 3: Medium Rigeness: Medium rigeness
# 4: represents a flower or flower (or no apple) : None
Categories = ['1', '2', '3', '4']
training_data = []

# Load Data
def load_data():
    for category in Categories:
        path = os.path.join(datadir, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                
                # RGB Model
                img_rgb = cv2.resize(img_array, (100, 100))
                image_rgb = np.array(img_rgb).flatten()
                
                # HSV Model
                img_hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
                img_hsv[:, :, 1] = 255
                img_array = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                img_hsv = cv2.resize(img_array, (100, 100))
                image_hsv = np.array(img_hsv).flatten()

                # Combining Features from RGB and HSV
                combined_features = np.concatenate((image_rgb, image_hsv))
                
                # Classifying into 4 categories (1-4)
                if class_num == 0:  # Ripe
                    label = 1
                else:  # Unripe
                    label = 4

                training_data.append([combined_features, label])
            except Exception as e:
                print(e)

load_data()

# Splitting Data into Attributes and Labels
X = np.array([i[0] for i in training_data])
Y = np.array([i[1] for i in training_data])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training the Model
model = SVC(C=1, kernel='linear', gamma='auto')
model.fit(x_train, y_train)

# Evaluating the Model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Saving the Model
with open('combined_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Inference on New Images
def predict_ripeness(image_path):
    img_array = cv2.imread(image_path)
    img_rgb = cv2.resize(img_array, (100, 100))
    image_rgb = np.array(img_rgb).flatten()
    
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = 255
    img_array = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_hsv = cv2.resize(img_array, (100, 100))
    image_hsv = np.array(img_hsv).flatten()

    combined_features = np.concatenate((image_rgb, image_hsv))

    with open('combined_model.pkl', 'rb') as model_file:
        combined_model = pickle.load(model_file)

    prediction = combined_model.predict([combined_features])

    return prediction[0]

# Example Usage
image_path = r"F:\code\detect_data\Ripe\apple.jpg"
result = predict_ripeness(image_path)

print(f"The predicted ripeness (1-4) is: {result}")
