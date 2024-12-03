# Install required libraries
pip install torch torchvision

# Clone the YOLOv8 warehouse
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Download pre-trained model weights
python download.py

# Start fine-tuning
python train.py --img-size 640 --batch-size 16 --epochs 50 --data your_data.yaml --cfg your_model.yaml --weights yolov5s.pt
