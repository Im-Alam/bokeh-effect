from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

def apply_bokeh_effect(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)
    result = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(output_image_path, result)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No file part"

    input_image = request.files['image']

    if input_image.filename == '':
        return "No selected file"

    if input_image:
        filename = secure_filename(input_image.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        input_image.save(input_path)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

        apply_bokeh_effect(input_path, output_path)

        return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run()
