import cv2

# Read the input image
input_image_path = 'marc.jpg'  # Replace with your image path
output_image_path = 'output_drawing.jpg'

# Read the input image using OpenCV
input_image = cv2.imread(input_image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)  # Adjust thresholds as needed

# Invert the edges to create a sketch-like effect
inverted_edges = 255 - edges

# Save the output image
cv2.imwrite(output_image_path, inverted_edges)
