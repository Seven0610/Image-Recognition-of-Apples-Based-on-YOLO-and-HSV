import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression

def adjust_threshold(model, img_path, threshold=0.5):
    # Read image
    img = torch.zeros((1, 3, 640, 640), device='cuda')
    img[0] = torch.from_numpy(cv2.imread(img_path, cv2.IMREAD_COLOR)).permute(2, 0, 1)  # BGR to RGB

    # reason
    pred = model(img)[0]

    # Non-maximum inhibition
    pred = non_max_suppression(pred, conf_thres=threshold)

    # Here the detection results are processed, such as drawing boxes, saving results, etc

if __name__ == '__main__':
    # Loading model
    weights_path = 'path/to/yolov8/weights'
    model = attempt_load(weights_path, map_location=torch.device('cuda'))

    # Image path
    img_path = 'path/to/*.jpg'

    # Adjustment threshold
    adjusted_threshold = 0.6  # Adjustment threshold
    adjust_threshold(model, img_path, threshold=adjusted_threshold)
