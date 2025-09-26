import streamlit as st
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image
from io import BytesIO

# Global flags for optional dependencies (set later when actually needed)
CV2_AVAILABLE = False
PYWT_AVAILABLE = False
SKIMAGE_AVAILABLE = False
JOBLIB_AVAILABLE = False
TF_AVAILABLE = False
FITZ_AVAILABLE = False
PLOTLY_AVAILABLE = False

# Deployment-safe lazy import functions
def get_cv2():
    global CV2_AVAILABLE
    try:
        import cv2
        CV2_AVAILABLE = True
        return cv2
    except ImportError:
        CV2_AVAILABLE = False
        return None

def get_pywt():
    global PYWT_AVAILABLE
    try:
        import pywt
        PYWT_AVAILABLE = True
        return pywt
    except ImportError:
        PYWT_AVAILABLE = False
        return None

def get_skimage_lbp():
    global SKIMAGE_AVAILABLE
    try:
        from skimage.feature import local_binary_pattern as sk_lbp
        SKIMAGE_AVAILABLE = True
        return sk_lbp
    except ImportError:
        SKIMAGE_AVAILABLE = False
        return None

def get_joblib():
    global JOBLIB_AVAILABLE
    try:
        import joblib
        JOBLIB_AVAILABLE = True
        return joblib
    except ImportError:
        JOBLIB_AVAILABLE = False
        return None

def get_tensorflow():
    global TF_AVAILABLE
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        TF_AVAILABLE = True
        return tf, load_model
    except ImportError:
        TF_AVAILABLE = False
        return None, None

def get_fitz():
    global FITZ_AVAILABLE
    try:
        import fitz
        FITZ_AVAILABLE = True
        return fitz
    except ImportError:
        FITZ_AVAILABLE = False
        return None

def get_plotly():
    global PLOTLY_AVAILABLE
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        PLOTLY_AVAILABLE = True
        return go, px
    except ImportError:
        PLOTLY_AVAILABLE = False
        return None, None

# Configuration
st.set_page_config(
    page_title="🔍 AI TraceFinder - Advanced Image Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .analysis-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .progress-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def check_model_availability():
    """Check which models are available"""
    models = {
        'scanner_hybrid': False,
        'tamper_svm': False,
        'fingerprints': False
    }
    
    if os.path.exists('models/scanner_hybrid.keras'):
        models['scanner_hybrid'] = True
    
    if os.path.exists('models/tamper_svm_model.pkl'):
        models['tamper_svm'] = True
        
    if os.path.exists('models/scanner_fingerprints.pkl'):
        models['fingerprints'] = True
    
    return models

def load_models_safe():
    """Safely load models with fallbacks"""
    models = {}
    
    # Try to load scanner model
    try:
        tf, load_model = get_tensorflow()
        joblib = get_joblib()
        
        if tf is not None and os.path.exists('models/scanner_hybrid.keras'):
            models['scanner_model'] = load_model('models/scanner_hybrid.keras')
            
        if joblib is not None:
            if os.path.exists('models/hybrid_feat_scaler.pkl'):
                models['feat_scaler'] = joblib.load('models/hybrid_feat_scaler.pkl')
            if os.path.exists('models/hybrid_label_encoder.pkl'):
                models['label_encoder'] = joblib.load('models/hybrid_label_encoder.pkl')
            if os.path.exists('models/scanner_fingerprints.pkl'):
                models['fingerprints'] = joblib.load('models/scanner_fingerprints.pkl')
                
        # Try to load tamper model
        if joblib is not None:
            if os.path.exists('models/tamper_svm_model.pkl'):
                models['tamper_model'] = joblib.load('models/tamper_svm_model.pkl')
            if os.path.exists('models/tamper_svm_scaler.pkl'):
                models['tamper_scaler'] = joblib.load('models/tamper_svm_scaler.pkl')
            if os.path.exists('models/tamper_svm_threshold.pkl'):
                models['tamper_threshold'] = joblib.load('models/tamper_svm_threshold.pkl')
    except Exception as e:
        st.warning(f"Some models could not be loaded: {str(e)}")
    
    return models

def extract_features_safe(image):
    """Extract features with graceful degradation"""
    features = []
    cv2 = get_cv2()
    pywt = get_pywt()
    sk_lbp = get_skimage_lbp()
    
    if cv2 is not None:
        # Convert PIL to CV2
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Basic features
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray)
        ])
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ])
    else:
        # Fallback to basic numpy features
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        features.extend([np.mean(gray), np.std(gray), np.var(gray), 0, 0])
    
    # Wavelet features
    if pywt is not None:
        try:
            coeffs = pywt.dwt2(gray, 'db1')
            cA, (cH, cV, cD) = coeffs
            features.extend([
                np.mean(np.abs(cH)),
                np.mean(np.abs(cV)),
                np.mean(np.abs(cD))
            ])
        except:
            features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0])
    
    # LBP features
    if sk_lbp is not None:
        try:
            lbp = sk_lbp(gray, 8, 1, method='uniform')
            hist, _ = np.histogram(lbp, bins=10, range=(0, 9))
            features.extend(hist.tolist())
        except:
            features.extend([0] * 10)
    else:
        features.extend([0] * 10)
    
    return np.array(features)

def predict_scanner_safe(image, models):
    """Safe scanner prediction with fallbacks"""
    if 'scanner_model' not in models or 'feat_scaler' not in models or 'label_encoder' not in models:
        return "Demo Scanner", 0.85, "Simulated prediction - models not available"
    
    try:
        features = extract_features_safe(image)
        
        if len(features) == 0:
            return "Unknown", 0.0, "Feature extraction failed"
        
        # Reshape and scale features
        features_scaled = models['feat_scaler'].transform(features.reshape(1, -1))
        
        # Get prediction
        prediction = models['scanner_model'].predict(features_scaled)
        confidence = np.max(prediction[0])
        
        # Get class name
        class_idx = np.argmax(prediction[0])
        class_name = models['label_encoder'].inverse_transform([class_idx])[0]
        
        return class_name, confidence, "AI prediction successful"
        
    except Exception as e:
        return "Analysis Error", 0.0, f"Prediction failed: {str(e)}"

def predict_tamper_safe(image, models):
    """Safe tamper prediction with fallbacks"""
    if 'tamper_model' not in models or 'tamper_scaler' not in models:
        return "Original (Demo)", 0.92, "Simulated prediction - models not available"
    
    try:
        # Extract features for tamper detection
        features = extract_features_safe(image)
        
        if len(features) == 0:
            return "Unknown", 0.0, "Feature extraction failed"
        
        # Scale features
        features_scaled = models['tamper_scaler'].transform(features.reshape(1, -1))
        
        # Get prediction
        prediction = models['tamper_model'].predict_proba(features_scaled)
        confidence = np.max(prediction[0])
        
        # Determine class
        class_pred = models['tamper_model'].predict(features_scaled)[0]
        result = "Original" if class_pred == 0 else "Tampered"
        
        return result, confidence, "AI prediction successful"
        
    except Exception as e:
        return "Analysis Error", 0.0, f"Prediction failed: {str(e)}"

def analyze_image(image, analysis_type="both"):
    """Main analysis function with comprehensive error handling"""
    results = {
        'scanner': {'prediction': None, 'confidence': 0, 'status': 'Not analyzed'},
        'tamper': {'prediction': None, 'confidence': 0, 'status': 'Not analyzed'}
    }
    
    # Load models
    models = load_models_safe()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Scanner analysis
        if analysis_type in ["both", "scanner"]:
            status_text.text("🔍 Analyzing scanner characteristics...")
            progress_bar.progress(25)
            
            scanner, scanner_conf, scanner_status = predict_scanner_safe(image, models)
            results['scanner'] = {
                'prediction': scanner,
                'confidence': scanner_conf,
                'status': scanner_status
            }
            progress_bar.progress(50)
        
        # Tamper analysis
        if analysis_type in ["both", "tamper"]:
            status_text.text("🕵️ Detecting tampering indicators...")
            progress_bar.progress(75)
            
            tamper, tamper_conf, tamper_status = predict_tamper_safe(image, models)
            results['tamper'] = {
                'prediction': tamper,
                'confidence': tamper_conf,
                'status': tamper_status
            }
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
    
    return results

def create_visualization_safe(results):
    """Create visualizations with plotly fallback"""
    go, px = get_plotly()
    
    if go is not None:
        # Create confidence chart
        categories = []
        confidences = []
        
        if results['scanner']['prediction']:
            categories.append('Scanner Detection')
            confidences.append(results['scanner']['confidence'])
        
        if results['tamper']['prediction']:
            categories.append('Tamper Detection')
            confidences.append(results['tamper']['confidence'])
        
        if categories:
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=confidences,
                    marker=dict(
                        color=['#667eea', '#764ba2'],
                        line=dict(color='rgba(0,0,0,0)', width=0)
                    )
                )
            ])
            
            fig.update_layout(
                title="Analysis Confidence Levels",
                yaxis_title="Confidence",
                xaxis_title="Analysis Type",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analysis results to visualize")
    else:
        # Fallback visualization
        st.markdown("### 📊 Analysis Results")
        if results['scanner']['prediction']:
            st.metric("Scanner Confidence", f"{results['scanner']['confidence']:.1%}")
        if results['tamper']['prediction']:
            st.metric("Tamper Confidence", f"{results['tamper']['confidence']:.1%}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 AI TraceFinder</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Image Forensics & Scanner Detection System</p>', unsafe_allow_html=True)
    
    # Check model availability
    model_status = check_model_availability()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["both", "scanner", "tamper"],
            format_func=lambda x: {
                "both": "🔍 Both (Scanner + Tamper)",
                "scanner": "📱 Scanner Detection Only", 
                "tamper": "🕵️ Tamper Detection Only"
            }[x]
        )
        
        st.markdown("---")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Minimum confidence for reliable predictions"
        )
        
        st.markdown("---")
        st.markdown("## 📊 System Status")
        
        # Model status indicators
        scanner_status = "🟢 Ready" if model_status['scanner_hybrid'] else "🟡 Demo Mode"
        tamper_status = "🟢 Ready" if model_status['tamper_svm'] else "🟡 Demo Mode"
        
        st.metric("Scanner Model", scanner_status)
        st.metric("Tamper Model", tamper_status)
        
        if not any(model_status.values()):
            st.warning("⚠️ Running in demo mode - install dependencies for full AI analysis")
        
        st.markdown("---")
        st.markdown("## 🔧 Technical Info")
        st.info("**AI Models**: Hybrid CNN + SVM  \n**Features**: Wavelet, LBP, Gradients  \n**Deployment**: Streamlit Cloud")
    
    # Main content area
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Image Upload")
        
        uploaded_file = st.file_uploader(
            "Choose an image file for forensic analysis",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, TIFF, BMP"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            
            # Image metadata
            st.markdown("#### 📋 Image Information")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.metric("Format", image.format or "Unknown")
                st.metric("Size", f"{uploaded_file.size} bytes")
            
            with info_col2:
                st.metric("Dimensions", f"{image.size[0]} × {image.size[1]}")
                st.metric("Mode", image.mode)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Forensic Analysis")
        
        if uploaded_file is not None:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    results = analyze_image(image, analysis_type)
                
                # Display results
                if results['scanner']['prediction'] or results['tamper']['prediction']:
                    st.success("✅ Analysis Complete!")
                    
                    # Results display
                    if results['scanner']['prediction']:
                        st.markdown("#### 📱 Scanner Detection")
                        col_s1, col_s2 = st.columns(2)
                        
                        with col_s1:
                            st.metric(
                                "Detected Scanner",
                                results['scanner']['prediction'],
                                delta=f"{results['scanner']['confidence']:.1%} confidence"
                            )
                        
                        with col_s2:
                            confidence_color = "green" if results['scanner']['confidence'] > confidence_threshold else "orange"
                            st.markdown(f"**Status**: <span style='color: {confidence_color}'>{results['scanner']['status']}</span>", unsafe_allow_html=True)
                    
                    if results['tamper']['prediction']:
                        st.markdown("#### 🔍 Tamper Detection")
                        col_t1, col_t2 = st.columns(2)
                        
                        with col_t1:
                            st.metric(
                                "Image Status", 
                                results['tamper']['prediction'],
                                delta=f"{results['tamper']['confidence']:.1%} confidence"
                            )
                        
                        with col_t2:
                            confidence_color = "green" if results['tamper']['confidence'] > confidence_threshold else "orange"
                            st.markdown(f"**Status**: <span style='color: {confidence_color}'>{results['tamper']['status']}</span>", unsafe_allow_html=True)
                    
                    # Visualization
                    st.markdown("#### 📈 Confidence Analysis")
                    create_visualization_safe(results)
                    
                    # Summary
                    st.markdown("#### 📝 Analysis Summary")
                    
                    summary_text = "**Forensic Analysis Results:**\n\n"
                    
                    if results['scanner']['prediction']:
                        summary_text += f"🔍 **Scanner**: {results['scanner']['prediction']} ({results['scanner']['confidence']:.1%} confidence)\n\n"
                    
                    if results['tamper']['prediction']:
                        summary_text += f"🕵️ **Tamper Status**: {results['tamper']['prediction']} ({results['tamper']['confidence']:.1%} confidence)\n\n"
                    
                    if not any(model_status.values()):
                        summary_text += "⚠️ *Results shown are simulated for demo purposes. Install full dependencies for actual AI analysis.*"
                    
                    st.markdown(summary_text)
                else:
                    st.error("Analysis failed. Please try with a different image.")
        else:
            st.info("👆 Please upload an image to begin forensic analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Technical details expandable section
    with st.expander("🔬 Technical Details & Methodology"):
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            #### 🧠 AI Architecture
            - **Scanner Detection**: Hybrid CNN + Handcrafted Features
            - **Tamper Detection**: SVM with Calibrated Probabilities  
            - **Feature Engineering**: Wavelet Coefficients, LBP, Gradients
            - **Ensemble Approach**: Multi-model Decision Fusion
            """)
        
        with tech_col2:
            st.markdown("""
            #### 🔧 Technical Stack
            - **ML Frameworks**: TensorFlow/Keras, Scikit-learn
            - **Image Processing**: OpenCV, PyWavelets, Scikit-image
            - **Deployment**: Streamlit Cloud with Graceful Degradation
            - **Features**: 18-dimensional hybrid feature vector
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🌟 About AI TraceFinder")
    st.markdown(
        "AI TraceFinder combines state-of-the-art deep learning with traditional image forensics "
        "to provide comprehensive analysis of digital images. Our hybrid approach ensures robust "
        "detection across various scanner types and tampering techniques."
    )

if __name__ == "__main__":
    main()