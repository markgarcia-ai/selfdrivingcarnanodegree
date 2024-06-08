import ssl
import os
import torch
import torchvision
from torchvision import transforms  # Import transforms module from torchvision
from PIL import Image, ImageDraw
import imageio
import urllib.request

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download pre-trained model from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Function for object detection
def object_detection(images_folder):
    image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

    images = []
    for file_name in sorted(image_files):
        img = Image.open(file_name)
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            prediction = model(img_tensor)
        
        # Get bounding boxes and draw rectangles around cars
        for element in prediction[0]['labels']:
            if element.item() == 3:  # Assuming car class label is 3
                index = prediction[0]['labels'].tolist().index(element.item())
                box = prediction[0]['boxes'][index].cpu().numpy().astype(int)
                img = draw_rectangle(img, box)

        images.append(img)

    return images

# Function to draw rectangles around detected cars
def draw_rectangle(img, box):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
    return img_draw

# Function to create GIF
def create_gif(images, output_gif):
    imageio.mimsave(output_gif, [img for img in images], duration=0.5)  # Adjust duration between frames as needed

if __name__ == "__main__":
    images_folder_path = 'test_video'  # Replace this with your image folder path
    output_gif_path = 'PYTORCH_output2.gif'

    detected_images = object_detection(images_folder_path)
    create_gif(detected_images, output_gif_path)
