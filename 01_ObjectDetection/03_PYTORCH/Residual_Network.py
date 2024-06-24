import ssl
import os
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import imageio
import time
import json
import requests

class ObjectDetector:
    def __init__(self, model_name='vgg16', threshold=0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.model = self.load_model()
        self.transform = self.get_transform()
        self.labels = self.get_imagenet_labels()
        self.object_counts = {label: 0 for label in self.labels}

    def load_model(self):
        model = models.vgg16(pretrained=True)
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_imagenet_labels(self):
        labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(labels_url)
        return json.loads(response.content.decode('utf-8'))

    def detect_objects(self, images_folder):
        image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

        images = []
        for file_name in sorted(image_files):
            img = Image.open(file_name)
            img_tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                output = self.model(img_tensor)
                _, predicted = torch.max(output, 1)

            label = predicted.item()
            label_name = self.labels[label]
            self.object_counts[label_name] += 1

            img = self.draw_label(img, label_name, output[0][label].item())
            images.append(img)

        return images

    def draw_label(self, img, label, score):
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)

        label_text = f"{label} {score:.2f}"
        font = ImageFont.load_default()
        text_size = draw.textsize(label_text, font=font)
        text_location = (10, 10)
        draw.rectangle([text_location, (text_location[0] + text_size[0], text_location[1] + text_size[1])], fill="red")
        draw.text(text_location, label_text, fill="white", font=font)

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
    output_gif_path = 'VGG_output.gif'
    output_txt_path = 'object_counts.txt'

    detector = ObjectDetector()
    detected_images = detector.detect_objects(images_folder_path)
    detector.create_gif(detected_images, output_gif_path)
    detector.save_counts_to_file(output_txt_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    with open(output_txt_path, 'a') as file:
        file.write(f"\nTime taken: {elapsed_time:.2f} seconds")
