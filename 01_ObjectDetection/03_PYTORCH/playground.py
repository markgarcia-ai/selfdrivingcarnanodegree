import os
from PIL import Image
import imageio
from natsort import natsorted

def create_gif_from_images(image_folder, output_gif, duration=0.5):
    # Get all the PNG files from the folder and sort them naturally
    images = natsorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    
    # Read images and store them in a list
    frames = []
    for image in images:
        frame = Image.open(os.path.join(image_folder, image))
        frames.append(frame)

    # Convert frames to a format suitable for imageio
    frames = [frame.convert('RGB') for frame in frames]

    # Save the frames as a GIF
    imageio.mimsave(output_gif, frames, format='GIF', duration=duration)

if __name__ == "__main__":
    image_folder = 'test_video'  # Replace with your image folder path
    output_gif = 'video.gif'  # The output gif file name
    create_gif_from_images(image_folder, output_gif)
    print(f"GIF saved as {output_gif}")
