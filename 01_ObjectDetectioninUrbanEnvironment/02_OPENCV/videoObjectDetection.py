import cv2
import os

# Load the pre-trained classifiers for car and person detection
car_classifier = cv2.CascadeClassifier('models/haarcascade_car.xml')
person_classifier = cv2.CascadeClassifier('models/haarcascade_fullbody.xml')

# Create a list of image files in a directory
image_folder = 'test_video'
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

# Create VideoWrtier object to save the detections as a video
output_video = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_out = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 480))  # Adjust resolution as needed


# Process each image in the folder
for image_file in image_files:
    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        continue
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect cars in the image
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Detect persons in the image
    persons = person_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw rectangles around the detected persons
    for (x, y, w, h) in persons:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Write the frame to the output video
    video_out.write(cv2.resize(image, (640, 480)))

# Release the video writer
video_out.release()
