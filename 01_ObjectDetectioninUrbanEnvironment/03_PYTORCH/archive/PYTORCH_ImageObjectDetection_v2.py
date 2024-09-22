import ssl
import os
import torch
import torchvision
from torchvision import transforms  # Import transforms module from torchvision
from PIL import Image, ImageDraw, ImageFont
import imageio

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download pre-trained model from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# COCO dataset labels
COCO_INSTANCE_CATEGORY_NAMES = [
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

# Function for object detection
def object_detection(images_folder):
    image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

    images = []
    for file_name in sorted(image_files):
        img = Image.open(file_name)
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            prediction = model(img_tensor)

        # Get bounding boxes, labels, and scores
        for i in range(len(prediction[0]['labels'])):
            label = prediction[0]['labels'][i].item()
            score = prediction[0]['scores'][i].item()
            if score > 0.5:  # You can set a threshold for displaying the labels
                box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
                img = draw_rectangle(img, box, label, score)

        images.append(img)

    return images

# Function to draw rectangles around detected objects and add labels
def draw_rectangle(img, box, label, score):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

    # Get label name and confidence score
    try:
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
    except IndexError:
        label_name = 'Unknown'

    label_text = f"{label_name} {score:.2f}"

    # Load a font
    font = ImageFont.load_default()

    # Draw label text
    text_size = draw.textsize(label_text, font=font)
    text_location = (box[0], box[1] - text_size[1])
    draw.rectangle([text_location, (text_location[0] + text_size[0], text_location[1] + text_size[1])], fill="red")
    draw.text(text_location, label_text, fill="white", font=font)

    return img_draw

# Function to create GIF
def create_gif(images, output_gif):
    imageio.mimsave(output_gif, [img for img in images], duration=0.5)  # Adjust duration between frames as needed

if __name__ == "__main__":
    images_folder_path = 'test_video'  # Replace this with your image folder path
    output_gif_path = 'PYTORCH_output2.gif'

    detected_images = object_detection(images_folder_path)
    create_gif(detected_images, output_gif_path)
