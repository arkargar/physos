import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from skimage import morphology
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

print("Loading AI model...")
model = tf.saved_model.load('models/unet')
infer = model.signatures['serving_default']
print("Model loaded!")

def calculate_bend_angle(mask):
    pixels = np.sum(mask)
    if pixels < 500:
        return 0, pixels
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, pixels
    
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(cnt)
    
    if hull_area == 0:
        return 0, pixels
    
    defect_ratio = cnt_area / hull_area
    
    if defect_ratio > 0.9:
        angle = 0
    elif defect_ratio > 0.8:
        angle = 15
    elif defect_ratio > 0.7:
        angle = 30
    elif defect_ratio > 0.6:
        angle = 45
    elif defect_ratio > 0.5:
        angle = 60
    elif defect_ratio > 0.4:
        angle = 70
    else:
        angle = 80
    
    return angle, pixels

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize and process
        small = cv2.resize(image, (512, 512))
        input_tensor = tf.convert_to_tensor(small / 255.0, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        output = infer(input_2=input_tensor)
        pred = output['conv2d_transpose_4'].numpy()
        mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        angle, pixels = calculate_bend_angle(mask)
        
        return jsonify({
            'success': True,
            'angle': float(angle),
            'pixels': int(pixels)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)