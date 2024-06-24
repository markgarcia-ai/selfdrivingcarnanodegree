from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from detector import ObjectDetector
import os
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

detector = ObjectDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        annotated_image = detector.detect_objects(file_path)
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
        annotated_image.save(annotated_image_path)

        return redirect(url_for('show_image', filename='annotated_' + filename))

@app.route('/show/<filename>')
def show_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)
