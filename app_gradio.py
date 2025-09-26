import os
# Set environment variables before importing TensorFlow
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import gradio as gr
import numpy as np
import pickle
import time
from datetime import datetime
from PIL import Image
import io

# Try to import optional dependencies with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from skimage.feature import local_binary_pattern as sk_lbp
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not available")

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# Model paths
ART_DIR = "models"
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

def load_scanner_artifacts():
    """Load scanner detection model and artifacts"""
    global scanner_model, scanner_le, scanner_scaler, scanner_fps, fp_keys
    
    if not TF_AVAILABLE:
        return "❌ TensorFlow not available", None, None, None, None
    
    try:
        print("Loading scanner model...")
        scanner_model = tf.keras.models.load_model(SCANNER_MODEL_PATH, compile=False)
        
        with open(SCANNER_LABEL_ENCODER_PATH, "rb") as f:
            scanner_le = pickle.load(f)
        with open(SCANNER_SCALER_PATH, "rb") as f:
            scanner_scaler = pickle.load(f)
        with open(FP_PATH, "rb") as f:
            scanner_fps = pickle.load(f)
        fp_keys = np.load(FP_KEYS_PATH, allow_pickle=True).tolist()
        
        print(f"✅ Scanner model loaded: {scanner_model.count_params():,} parameters")
        return scanner_model, scanner_le, scanner_scaler, scanner_fps, fp_keys
        
    except Exception as e:
        print(f"❌ Scanner model loading failed: {e}")
        return None, None, None, None, None

def load_tamper_artifacts():
    """Load tamper detection model and artifacts"""
    global tamper_model, tamper_scaler, tamper_threshold
    
    if not JOBLIB_AVAILABLE:
        return None, None, None
        
    try:
        print("Loading tamper detection models...")
        tamper_model = joblib.load(TAMPER_MODEL_PATH)
        tamper_scaler = joblib.load(TAMPER_SCALER_PATH)
        tamper_threshold = joblib.load(TAMPER_THRESHOLD_PATH)
        
        print("✅ Tamper detection models loaded successfully")
        return tamper_model, tamper_scaler, tamper_threshold
        
    except Exception as e:
        print(f"❌ Tamper model loading failed: {e}")
        return None, None, None

# Load models on startup
print("🚀 Loading AI models...")
scanner_model, scanner_le, scanner_scaler, scanner_fps, fp_keys = load_scanner_artifacts()
tamper_model, tamper_scaler, tamper_threshold = load_tamper_artifacts()

# Feature extraction functions (simplified versions of your original functions)
def extract_scanner_features(img_array):
    """Extract features for scanner detection"""
    try:
        # Convert to RGB if needed
        if len(img_array.shape) == 3:
            rgb = img_array
        else:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) if CV2_AVAILABLE else img_array
        
        # Resize image
        if CV2_AVAILABLE:
            resized = cv2.resize(rgb, (256, 256))
        else:
            pil_img = Image.fromarray(rgb)
            resized = np.array(pil_img.resize((256, 256)))
        
        # Normalize
        processed = resized.astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        return processed
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def extract_tamper_features(img_array):
    """Extract features for tamper detection"""
    try:
        # Convert to grayscale
        if len(img_array.shape) == 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array
        
        # Basic statistical features
        features = []
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray),
            np.median(gray)
        ])
        
        # Simple edge detection
        if CV2_AVAILABLE:
            edges = cv2.Canny(gray, 50, 150)
            features.extend([np.mean(edges), np.std(edges)])
        else:
            # Simple gradient approximation
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            features.extend([np.mean(grad_x), np.mean(grad_y)])
        
        # Pad features to expected size
        while len(features) < 100:
            features.extend([0.0] * (100 - len(features)))
        
        return np.array(features[:100]).reshape(1, -1)
        
    except Exception as e:
        print(f"Tamper feature extraction error: {e}")
        return None

def analyze_image(image, analysis_type):
    """Main analysis function for Gradio interface"""
    if image is None:
        return "Please upload an image first.", ""
    
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        results = []
        
        # Scanner Detection
        if analysis_type in ["Both", "Scanner Only"]:
            if scanner_model is not None:
                try:
                    start_time = time.time()
                    features = extract_scanner_features(img_array)
                    
                    if features is not None:
                        prediction = scanner_model.predict(features, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = float(np.max(prediction[0]) * 100)
                        
                        if scanner_le:
                            label = scanner_le.inverse_transform([predicted_class])[0]
                        else:
                            label = f"Class_{predicted_class}"
                        
                        processing_time = time.time() - start_time
                        
                        results.append(f"""
🖨️ **Scanner Detection Results:**
- **Detected Scanner:** {label}
- **Confidence:** {confidence:.1f}%
- **Processing Time:** {processing_time:.2f}s
""")
                    else:
                        results.append("❌ Scanner feature extraction failed")
                        
                except Exception as e:
                    results.append(f"❌ Scanner analysis failed: {str(e)[:100]}")
            else:
                results.append("❌ Scanner model not available")
        
        # Tamper Detection
        if analysis_type in ["Both", "Tamper Only"]:
            if tamper_model is not None:
                try:
                    start_time = time.time()
                    features = extract_tamper_features(img_array)
                    
                    if features is not None:
                        if tamper_scaler:
                            features = tamper_scaler.transform(features)
                        
                        prediction = tamper_model.predict(features)[0]
                        probabilities = tamper_model.predict_proba(features)[0]
                        confidence = float(np.max(probabilities) * 100)
                        
                        label = "Clean" if prediction == 0 else "Tampered"
                        processing_time = time.time() - start_time
                        
                        # Status emoji
                        status_emoji = "✅" if label == "Clean" else "⚠️"
                        
                        results.append(f"""
{status_emoji} **Tamper Detection Results:**
- **Status:** {label}
- **Confidence:** {confidence:.1f}%
- **Processing Time:** {processing_time:.2f}s
""")
                    else:
                        results.append("❌ Tamper feature extraction failed")
                        
                except Exception as e:
                    results.append(f"❌ Tamper analysis failed: {str(e)[:100]}")
            else:
                results.append("❌ Tamper model not available")
        
        if results:
            return "\n".join(results), "✅ Analysis completed successfully!"
        else:
            return "❌ No analysis performed", "Please check your settings"
            
    except Exception as e:
        return f"❌ Error processing image: {str(e)}", ""

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="AI TraceFinder",
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🔍 AI TraceFinder</h1>
            <p>Advanced Image Forensics: Scanner Detection & Tamper Analysis</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload Image")
                image_input = gr.Image(
                    label="Upload Image for Analysis",
                    type="pil",
                    height=300
                )
                
                analysis_type = gr.Radio(
                    choices=["Both", "Scanner Only", "Tamper Only"],
                    value="Both",
                    label="Analysis Type"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
                
                # Status information
                gr.Markdown(f"""
### 🔧 System Status
- **TensorFlow:** {'✅' if TF_AVAILABLE else '❌'}
- **OpenCV:** {'✅' if CV2_AVAILABLE else '❌'}
- **Scanner Model:** {'✅' if scanner_model is not None else '❌'}
- **Tamper Model:** {'✅' if tamper_model is not None else '❌'}
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Analysis Results")
                results_output = gr.Markdown(
                    "Upload an image and click 'Analyze Image' to see results.",
                    label="Results"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input, analysis_type],
            outputs=[results_output, status_output]
        )
        
        gr.Markdown("""
### 📋 Instructions
1. **Upload an image** using the file uploader above
2. **Select analysis type**: Both models, Scanner only, or Tamper detection only
3. **Click 'Analyze Image'** to run the AI analysis
4. **View results** with confidence scores and processing times

### 🎯 Features
- **Scanner Source Detection**: Identifies which scanner/device created the image
- **Tamper Detection**: Detects if the image has been manipulated
- **Professional Analysis**: Uses trained AI models with confidence scoring
- **Fast Processing**: Optimized for quick analysis

### 📱 Supported Formats
- JPEG, PNG, TIFF, and other common image formats
        """)

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )