from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model.object_detection import ObjectDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Instantiate the object detector
detector = ObjectDetector()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Detect objects in the image
            result_img, num_objects, object_counts = detector.detect_objects(filepath)
            result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            result_img.save(result_img_path)
            
            return render_template('index.html', uploaded_img_url=filepath, result_img_url=result_img_path, num_objects=num_objects, object_counts=object_counts)

    return render_template('index.html')

@app.route('/calculate_map', methods=['POST'])
def calculate_map():
    if request.method == 'POST':
        true_objects = int(request.form['true_objects'])
        detected_objects = int(request.form['detected_objects'])
        
        # Calculate mAP (for simplicity, assume true_objects == detected_objects as an example)
        map_score = detector.calculate_map(true_objects, detected_objects)
        
        return f"mAP Score: {map_score:.2f}"

if __name__ == "__main__":
    app.run(debug=True, port=8000)
