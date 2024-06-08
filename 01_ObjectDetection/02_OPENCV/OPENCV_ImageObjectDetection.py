import cv2
import os
import imageio

# Load the pre-trained classifiers for car and person detection
car_classifier = cv2.CascadeClassifier('models/haarcascade_car.xml')
person_classifier = cv2.CascadeClassifier('models/haarcascade_fullbody.xml')

# Add images in array
images_folder = 'test_video'
image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

images = []
for image in sorted(image_files):
    # Convert the image to grayscale
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the image
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Detect persons in the image
    persons = person_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw rectangles around the detected persons
    for (x, y, w, h) in persons:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Resize the image (to keep the same aspect ratio, use interpolation=cv2.INTER_AREA)
    img_resized = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    images.append(img_resized)

imageio.mimsave('OPENCV_output_gif2.gif', images, duration=0.25)  # Adjust duration between frames as needed
