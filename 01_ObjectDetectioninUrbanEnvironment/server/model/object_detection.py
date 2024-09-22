import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

# COCO dataset labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ObjectDetector:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

    def detect_objects(self, img_path):
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)

        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']

        # Filter out low-confidence detections (e.g., confidence < 0.5)
        threshold = 0.5
        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []

        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                filtered_boxes.append(box)
                filtered_labels.append(COCO_INSTANCE_CATEGORY_NAMES[label])
                filtered_scores.append(score)

        # Drawing rectangles and labels
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for box, label in zip(filtered_boxes, filtered_labels):
            draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=3)
            draw.text((box[0], box[1]), label, fill="red", font=font)

        object_counts = {label: filtered_labels.count(label) for label in set(filtered_labels)}
        num_objects = len(filtered_boxes)

        return image, num_objects, object_counts

    def calculate_map(self, true_objects, detected_objects):
        # Simplified example of mAP calculation
        if true_objects == 0:
            return 1.0 if detected_objects == 0 else 0.0
        return min(true_objects, detected_objects) / true_objects
