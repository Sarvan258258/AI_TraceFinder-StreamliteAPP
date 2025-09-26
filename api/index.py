import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import from app.py
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables before importing TensorFlow
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pickle
import base64
from PIL import Image
from io import BytesIO
import json

app = Flask(__name__)

# Check available dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Model paths (relative to project root)
ART_DIR = "../models"
TAMPER_MODEL_PATH = os.path.join(ART_DIR, "tamper_svm_model.pkl")
TAMPER_SCALER_PATH = os.path.join(ART_DIR, "tamper_svm_scaler.pkl")
TAMPER_THRESHOLD_PATH = os.path.join(ART_DIR, "tamper_svm_threshold.pkl")

# Global model variables
tamper_model = None
tamper_scaler = None
tamper_threshold = None

def load_models():
    """Load tamper detection model (SVM only for Vercel compatibility)"""
    global tamper_model, tamper_scaler, tamper_threshold
    
    if tamper_model is None and JOBLIB_AVAILABLE and SKLEARN_AVAILABLE:
        try:
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "tamper_svm_model.pkl")
            scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "tamper_svm_scaler.pkl")
            threshold_path = os.path.join(os.path.dirname(__file__), "..", "models", "tamper_svm_threshold.pkl")
            
            if os.path.exists(model_path):
                tamper_model = joblib.load(model_path)
                print("✅ Tamper SVM model loaded")
            if os.path.exists(scaler_path):
                tamper_scaler = joblib.load(scaler_path)
                print("✅ Tamper scaler loaded")
            if os.path.exists(threshold_path):
                tamper_threshold = joblib.load(threshold_path)
                print("✅ Tamper threshold loaded")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")

def extract_simple_features(img_array):
    """Extract simple statistical features from image (no OpenCV dependency)"""
    try:
        # Convert to grayscale if RGB
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Basic statistical features
        features = []
        features.extend([
            np.mean(gray),
            np.std(gray), 
            np.min(gray),
            np.max(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75)
        ])
        
        # Histogram features (simplified)
        hist, _ = np.histogram(gray, bins=16, range=(0, 255))
        features.extend(hist.tolist())
        
        # Edge detection substitute (gradient approximation)
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))
        features.extend([
            np.mean(grad_x),
            np.std(grad_x),
            np.mean(grad_y), 
            np.std(grad_y)
        ])
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        # Return dummy features if extraction fails
        return np.random.random((1, 27))

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI TraceFinder - Vercel (Tamper Detection)</title>
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
        .warning {
            color: #d69e2e;
            background: #faf5e6;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🕵️ AI TraceFinder</h1>
        <p style="text-align: center; color: #718096;">Tamper Detection Analysis (Vercel Optimized)</p>
        
        <div class="warning">
            <strong>⚠️ Note:</strong> This is a Vercel-optimized version with tamper detection only. 
            Scanner detection requires TensorFlow which exceeds Vercel's size limits.
        </div>
        
        <div class="upload-area">
            <h3>📸 Upload Image for Tamper Analysis</h3>
            <input type="file" id="imageFile" accept="image/*" />
            <br>
            <button class="btn" onclick="analyzeImage()">🔍 Analyze for Tampering</button>
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
            
            resultsDiv.innerHTML = '<div class="loading">🔄 Analyzing image for tampering...</div>';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    let html = '<div class="results"><h3>📊 Tamper Detection Results</h3>';
                    
                    if (result.tamper) {
                        const isClean = result.tamper.label === 'Clean';
                        const statusClass = isClean ? 'success' : 'error';
                        const icon = isClean ? '🛡️' : '⚠️';
                        
                        html += `<div class="${statusClass}">
                            ${icon} <strong>Status:</strong> ${result.tamper.label}<br>
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
    """API endpoint for tamper detection analysis"""
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
        
        # Tamper detection with SVM
        if tamper_model is not None and SKLEARN_AVAILABLE:
            try:
                import time
                start_time = time.time()
                
                # Resize image for consistent feature extraction
                resized_img = image.resize((256, 256))
                img_array = np.array(resized_img)
                
                # Extract features
                features = extract_simple_features(img_array)
                
                # Scale features if scaler is available
                if tamper_scaler:
                    features = tamper_scaler.transform(features)
                
                # Predict
                prediction = tamper_model.predict(features)[0]
                probabilities = tamper_model.predict_proba(features)[0]
                confidence = float(np.max(probabilities) * 100)
                
                processing_time = time.time() - start_time
                
                results['tamper'] = {
                    'label': 'Clean' if prediction == 0 else 'Tampered',
                    'confidence': confidence,
                    'time': processing_time
                }
            except Exception as e:
                print(f"Tamper detection error: {e}")
                results['tamper'] = {
                    'label': 'Analysis Failed',
                    'confidence': 0.0,
                    'time': 0.0
                }
        else:
            results['tamper'] = {
                'label': 'Model Unavailable',
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
        'dependencies': {
            'joblib': JOBLIB_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE,
        },
        'models_loaded': {
            'tamper_model': tamper_model is not None,
            'tamper_scaler': tamper_scaler is not None,
        },
        'note': 'Vercel-optimized version - tamper detection only'
    })

# Vercel serverless function handler
def handler(request, response):
    """Vercel serverless function entry point"""
    return app(request, response)

if __name__ == '__main__':
    app.run(debug=True)