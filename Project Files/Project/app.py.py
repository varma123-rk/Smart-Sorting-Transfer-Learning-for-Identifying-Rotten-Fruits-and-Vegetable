import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = 'smart_sorting_secret_key_2025'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
try:
    model = load_model('healthy_vs_rotten.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names for 28 classes of fruits and vegetables
CLASS_NAMES = [
    'Apple_healthy', 'Apple_rotten', 'Banana_healthy', 'Banana_rotten',
    'Bell_pepper_healthy', 'Bell_pepper_rotten', 'Carrot_healthy', 'Carrot_rotten',
    'Cucumber_healthy', 'Cucumber_rotten', 'Grapes_healthy', 'Grapes_rotten',
    'Lemon_healthy', 'Lemon_rotten', 'Mango_healthy', 'Mango_rotten',
    'Orange_healthy', 'Orange_rotten', 'Potato_healthy', 'Potato_rotten',
    'Strawberry_healthy', 'Strawberry_rotten', 'Tomato_healthy', 'Tomato_rotten',
    'Watermelon_healthy', 'Watermelon_rotten', 'Onion_healthy', 'Onion_rotten'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for VGG16 model prediction"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(img_path):
    """Make prediction on uploaded image"""
    if model is None:
        return None, 0, "Model not loaded"
    
    try:
        processed_img = preprocess_image(img_path)
        if processed_img is None:
            return None, 0, "Error processing image"
        
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Extract produce type and condition
        parts = predicted_class.split('_')
        produce_type = parts[0]
        condition = parts[1] if len(parts) > 1 else 'unknown'
        
        return {
            'class': predicted_class,
            'produce_type': produce_type,
            'condition': condition,
            'confidence': round(confidence, 2),
            'is_healthy': condition.lower() == 'healthy'
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/blog-single')
def blog_single():
    return render_template('blog-single.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        if result:
            return render_template('index.html', 
                                 prediction=result, 
                                 image_path=f'uploads/{filename}')
        else:
            flash('Error making prediction')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        
        if result:
            return jsonify({
                'success': True,
                'prediction': result,
                'image_url': f'/static/uploads/{filename}'
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'supported_classes': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
