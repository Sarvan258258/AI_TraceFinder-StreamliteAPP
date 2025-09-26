import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import from app.py
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables before importing TensorFlow
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pickle
import base64
from PIL import Image
from io import BytesIO
import json

app = Flask(__name__)

# Import your AI model functions
try:
    # Import core functions from your main app
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Model paths
ART_DIR = "../models"
SCANNER_MODEL_PATH = os.path.join(ART_DIR, "scanner_hybrid.keras")
SCANNER_LABEL_ENCODER_PATH = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
SCANNER_SCALER_PATH = os.path.join(ART_DIR, "hybrid_feat_scaler.pkl")
FP_KEYS_PATH = os.path.join(ART_DIR, "fp_keys.npy")
FP_PATH = os.path.join(ART_DIR, "scanner_fingerprints.pkl")

# Tamper detection models
TAMPER_MODEL_PATH = os.path.join(ART_DIR, "tamper_svm_model.pkl")
TAMPER_SCALER_PATH = os.path.join(ART_DIR, "tamper_svm_scaler.pkl")
TAMPER_THRESHOLD_PATH = os.path.join(ART_DIR, "tamper_svm_threshold.pkl")

# Global model variables
scanner_model = None
scanner_le = None
scanner_scaler = None
scanner_fps = None
fp_keys = None
tamper_model = None
tamper_scaler = None
tamper_threshold = None

def load_models():
    """Load AI models on first request"""
    global scanner_model, scanner_le, scanner_scaler, scanner_fps, fp_keys
    global tamper_model, tamper_scaler, tamper_threshold
    
    if scanner_model is None and TF_AVAILABLE:
        try:
            scanner_model = tf.keras.models.load_model(SCANNER_MODEL_PATH, compile=False)
            with open(SCANNER_LABEL_ENCODER_PATH, "rb") as f:
                scanner_le = pickle.load(f)
            with open(SCANNER_SCALER_PATH, "rb") as f:
                scanner_scaler = pickle.load(f)
            with open(FP_PATH, "rb") as f:
                scanner_fps = pickle.load(f)
            fp_keys = np.load(FP_KEYS_PATH, allow_pickle=True).tolist()
        except Exception as e:
            print(f"Scanner model loading failed: {e}")
    
    if tamper_model is None and JOBLIB_AVAILABLE:
        try:
            tamper_model = joblib.load(TAMPER_MODEL_PATH)
            tamper_scaler = joblib.load(TAMPER_SCALER_PATH)
            tamper_threshold = joblib.load(TAMPER_THRESHOLD_PATH)
        except Exception as e:
            print(f"Tamper model loading failed: {e}")

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI TraceFinder - Vercel</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #4a5568;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #cbd5e0;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #667eea;
        }
        input[type="file"] {
            margin: 20px 0;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 10px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        .error {
            color: #e53e3e;
            background: #fed7d7;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .success {
            color: #38a169;
            background: #c6f6d5;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🕵️ AI TraceFinder</h1>
        <p style="text-align: center; color: #718096;">Scanner Detection & Tamper Analysis</p>
        
        <div class="upload-area">
            <h3>📸 Upload Image for Analysis</h3>
            <input type="file" id="imageFile" accept="image/*" />
            <br>
            <button class="btn" onclick="analyzeImage()">🔍 Analyze Image</button>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        async function analyzeImage() {
            const fileInput = document.getElementById('imageFile');
            const resultsDiv = document.getElementById('results');
            
            if (!fileInput.files[0]) {
                resultsDiv.innerHTML = '<div class="error">Please select an image file first.</div>';
                return;
            }
            
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('image', file);
            
            resultsDiv.innerHTML = '<div class="loading">🔄 Analyzing image with AI models...</div>';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    let html = '<div class="results"><h3>📊 Analysis Results</h3>';
                    
                    if (result.scanner) {
                        html += `<div class="success">
                            🖨️ <strong>Scanner Source:</strong> ${result.scanner.label}<br>
                            📊 <strong>Confidence:</strong> ${result.scanner.confidence.toFixed(1)}%<br>
                            ⏱️ <strong>Processing Time:</strong> ${result.scanner.time.toFixed(2)}s
                        </div>`;
                    }
                    
                    if (result.tamper) {
                        html += `<div class="success">
                            🛡️ <strong>Tamper Status:</strong> ${result.tamper.label}<br>
                            📊 <strong>Confidence:</strong> ${result.tamper.confidence.toFixed(1)}%<br>
                            ⏱️ <strong>Processing Time:</strong> ${result.tamper.time.toFixed(2)}s
                        </div>`;
                    }
                    
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = `<div class="error">❌ Analysis failed: ${result.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">❌ Network error: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint for image analysis"""
    try:
        # Load models on first request
        load_models()
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        img_array = np.array(image)
        
        results = {}
        
        # Scanner detection (simplified version)
        if scanner_model is not None:
            try:
                # Simplified prediction - you'll need to implement your full feature extraction
                resized = cv2.resize(img_array, (256, 256)) if CV2_AVAILABLE else np.array(image.resize((256, 256)))
                processed = resized.astype(np.float32) / 255.0
                processed = np.expand_dims(processed, axis=0)
                
                prediction = scanner_model.predict(processed, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]) * 100)
                
                if scanner_le:
                    label = scanner_le.inverse_transform([predicted_class])[0]
                else:
                    label = f"Class_{predicted_class}"
                
                results['scanner'] = {
                    'label': label,
                    'confidence': confidence,
                    'time': 0.5  # Placeholder
                }
            except Exception as e:
                results['scanner'] = {
                    'label': 'Prediction Failed',
                    'confidence': 0.0,
                    'time': 0.0
                }
        
        # Tamper detection (simplified version)
        if tamper_model is not None:
            try:
                # Simplified tamper detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(img_array, axis=2)
                features = gray.flatten()[:1000]  # Simplified features
                features = features.reshape(1, -1)
                
                if tamper_scaler:
                    features = tamper_scaler.transform(features)
                
                prediction = tamper_model.predict(features)[0]
                confidence = float(tamper_model.predict_proba(features)[0].max() * 100)
                
                results['tamper'] = {
                    'label': 'Clean' if prediction == 0 else 'Tampered',
                    'confidence': confidence,
                    'time': 0.3  # Placeholder
                }
            except Exception as e:
                results['tamper'] = {
                    'label': 'Prediction Failed',
                    'confidence': 0.0,
                    'time': 0.0
                }
        
        return jsonify({'success': True, **results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tensorflow': TF_AVAILABLE,
        'opencv': CV2_AVAILABLE,
        'joblib': JOBLIB_AVAILABLE
    })

# Vercel entry point
def handler(request, context):
    return app(request, context)

if __name__ == '__main__':
    app.run(debug=True)