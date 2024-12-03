from ultralytics import YOLO
# Check if GPU is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Load a model
#model = YOLO( 'ultralytics/cfg/models/v8/yolov8-cls.yaml') # build a new model from YANL
#model = YOLO( 'yolov8n-cls.pt' ) # load a pretrained model ( recommended for training)
model = YOLO( 'ultralytics/cfg/models/v8/yolov8-cls.yaml' ).load('./yolov8n-cls.pt').to(device)# build from YAML and transfer weights
#Train the model
results = model.train(data='F:/yolov8/ultralytics-main/dataset',epochs=15,imgsz=64)

# threshold_loss = 0.01  # Set your desired loss threshold

# for epoch in range(100):
#     # Train for one epoch
#     results = model.train(data='F:/yolov8/ultralytics-main/dataset', epochs=1, imgsz=64)

#     # Get the loss value for the current epoch using appropriate method
#     current_loss = results.get_loss()

#     # Print the current loss
#     print(f'Epoch {epoch + 1}/{100}, Loss: {current_loss}')

#     # Check if the current loss is below the threshold
#     if current_loss < threshold_loss:
#         print(f'Loss below threshold ({threshold_loss}). Stopping training.')
#         break
