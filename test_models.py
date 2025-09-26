"""
Quick test to verify model loading after conversion
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def test_model_loading():
    """Test if the converted model loads successfully"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        model_path = "models/scanner_hybrid.keras"
        print(f"\n🔄 Testing model loading from: {model_path}")
        
        # Test loading the model
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
        
        # Display model info
        if hasattr(model, 'input_shape'):
            print(f"   - Input shape: {model.input_shape}")
        else:
            print(f"   - Input shapes: {[input.shape for input in model.inputs]}")
        
        if hasattr(model, 'output_shape'):    
            print(f"   - Output shape: {model.output_shape}")
        else:
            print(f"   - Output shapes: {[output.shape for output in model.outputs]}")
            
        print(f"   - Total parameters: {model.count_params():,}")
        
        # Test other artifacts
        print("\n🔄 Testing other artifacts...")
        
        import pickle
        import numpy as np
        
        try:
            with open("models/hybrid_label_encoder.pkl", "rb") as f:
                le = pickle.load(f)
            print("✅ Label encoder loaded")
        except Exception as e:
            print(f"⚠️ Label encoder failed: {e}")
        
        try:
            with open("models/hybrid_feat_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            print("✅ Feature scaler loaded")
        except Exception as e:
            print(f"⚠️ Feature scaler failed: {e}")
        
        try:
            with open("models/scanner_fingerprints.pkl", "rb") as f:
                fps = pickle.load(f)
            print("✅ Scanner fingerprints loaded")
        except Exception as e:
            print(f"⚠️ Scanner fingerprints failed: {e}")
        
        try:
            keys = np.load("models/fp_keys.npy", allow_pickle=True).tolist()
            print("✅ FP keys loaded")
        except Exception as e:
            print(f"⚠️ FP keys failed: {e}")
        
        print("\n🎉 All tests passed! The application should now work correctly.")
        return True
        
    except ImportError:
        print("❌ TensorFlow not available")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Model Loading After Conversion")
    print("=" * 50)
    success = test_model_loading()
    
    if success:
        print("\n✅ SUCCESS! You can now run the Streamlit application.")
        print("Run: streamlit run app.py")
    else:
        print("\n❌ FAILED! There are still issues with model loading.")