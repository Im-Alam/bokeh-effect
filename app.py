from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

def segment_image(original_image, num_segments=2, threshold=240):
    """Segments an image using K-means clustering and returns the segmented image and blurred background.

    Args:
        original_image: The input image as a NumPy array.
        num_segments: The number of segments to create (default: 2).
        threshold: Threshold value for creating the binary mask (default: 240).
        blur_kernel: Kernel size for Gaussian blurring the background (default: (15, 15)).

    Returns:
        segmented_image: The segmented image as a NumPy array.
        blurred_background: The blurred background image as a NumPy array.
    """

    # Validate input image (example)
    if original_image is None:
        raise ValueError("Invalid image input")

    # Convert to float32 for K-means (might be unnecessary if BGR works)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    image = original_image.copy().astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10

    # Run K-means clustering
    ret, label, center = cv2.kmeans(image.reshape((-1, 3)), num_segments, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Reconstruct image from cluster centers
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((original_image.shape))

    # Create binary mask
    _, binary_image = cv2.threshold(segmented_image, threshold, 255, cv2.THRESH_BINARY)

    # Get grayscale for further processing
    segmented_gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    # Threshold and optionally invert mask
    _, mask = cv2.threshold(segmented_gray, 1, 255, cv2.THRESH_BINARY_INV)

    # Apply mask and create blurred background
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    return masked_image


def apply_bokeh_effect(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)
    obj = segment_image(img)
    bg = cv2.GaussianBlur(img, (15, 15), 0)
    result = cv2.add(obj, bg)
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
