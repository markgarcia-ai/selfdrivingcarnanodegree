import ssl
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import imageio
import time
from natsort import natsorted  # Import natsorted

class ObjectDetector:
    def __init__(self, model_name='fasterrcnn_resnet50_fpn', threshold=0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.model = self.load_model()
        self.transform = self.get_transform()
        self.coco_labels = self.get_coco_labels()
        self.object_counts = {label: 0 for label in self.coco_labels}

    def load_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.ToTensor()
        ])

    def get_coco_labels(self):
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect_objects(self, images_folder):
        image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

        # Use natsorted to sort images naturally
        image_files = natsorted(image_files)

        images = []
        for file_name in image_files:
            img = Image.open(file_name)
            img_tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(img_tensor)

            object_count = 0
            for i in range(len(prediction[0]['labels'])):
                label = prediction[0]['labels'][i].item()
                score = prediction[0]['scores'][i].item()
                if score > self.threshold:
                    box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
                    img = self.draw_rectangle(img, box, label, score)
                    object_count += 1
                    if label < len(self.coco_labels):
                        label_name = self.coco_labels[label]
                        self.object_counts[label_name] += 1
                    else:
                        print(f"Warning: Label index {label} is out of range for COCO labels")

            img = self.draw_counter(img, object_count)
            images.append(img)

        return images

    def draw_rectangle(self, img, box, label, score):
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

        try:
            label_name = self.coco_labels[label] if label < len(self.coco_labels) else 'Unknown'
        except IndexError:
            label_name = 'Unknown'

        label_text = f"{label_name} {score:.2f}"

        font = ImageFont.load_default()
        text_size = draw.textbbox((0, 0), label_text, font=font)[2:]  # Updated text size calculation
        text_location = (box[0], box[1] - text_size[1])
        draw.rectangle([text_location, (text_location[0] + text_size[0], text_location[1] + text_size[1])], fill="red")
        draw.text(text_location, label_text, fill="white", font=font)

        return img_draw

    def draw_counter(self, img, object_count):
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        font = ImageFont.load_default()
        counter_text = f"Objects detected: {object_count}"
        text_size = draw.textbbox((0, 0), counter_text, font=font)[2:]  # Updated text size calculation
        text_location = (10, 10)
        draw.rectangle([text_location, (text_location[0] + text_size[0], text_location[1] + text_size[1])], fill="blue")
        draw.text(text_location, counter_text, fill="white", font=font)
        return img_draw

    def save_counts_to_file(self, output_file):
        with open(output_file, 'w') as file:
            for label, count in self.object_counts.items():
                if count > 0:
                    file.write(f"{label}: {count}\n")

    def create_gif(self, images, output_gif):
        imageio.mimsave(output_gif, [img for img in images], duration=0.5)

if __name__ == "__main__":
    start_time = time.time()

    images_folder_path = 'test_video'  # Replace this with your image folder path
    output_gif_path = 'Fasterrcnn.gif'
    output_txt_path = 'Fasterrcnn_object_counts.txt'

    detector = ObjectDetector()
    detected_images = detector.detect_objects(images_folder_path)
    detector.create_gif(detected_images, output_gif_path)
    detector.save_counts_to_file(output_txt_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    with open(output_txt_path, 'a') as file:
        file.write(f"\nTime taken: {elapsed_time:.2f} seconds")
