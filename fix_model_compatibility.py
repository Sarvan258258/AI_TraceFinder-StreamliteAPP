"""
Model Compatibility Fixer for TensorFlow/Keras Version Issues
This script helps convert models between different TensorFlow/Keras versions.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def check_tensorflow():
    """Check TensorFlow installation and version"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ Keras version: {tf.keras.__version__}")
        return tf
    except ImportError:
        print("❌ TensorFlow not installed!")
        print("Run: pip install tensorflow")
        return None

def convert_model(model_path, output_path=None):
    """Convert model to current TensorFlow version"""
    tf = check_tensorflow()
    if not tf:
        return False
    
    if output_path is None:
        output_path = model_path.replace('.keras', '_converted.keras')
    
    try:
        print(f"\n🔄 Loading model from: {model_path}")
        
        # Try different loading approaches
        model = None
        
        # Method 1: Standard loading
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully with standard method")
        except Exception as e1:
            print(f"⚠️ Standard loading failed: {str(e1)[:100]}...")
            
            # Method 2: Load with custom objects disabled
            try:
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False, 
                    custom_objects={}
                )
                print("✅ Model loaded with custom_objects disabled")
            except Exception as e2:
                print(f"❌ All loading methods failed!")
                print(f"Error details: {str(e2)}")
                return False
        
        if model:
            # Recompile the model with current TensorFlow version
            print("\n🔄 Recompiling model...")
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save in current format
            print(f"\n💾 Saving converted model to: {output_path}")
            model.save(output_path, save_format='keras')
            print("✅ Model conversion completed successfully!")
            
            # Verify the converted model
            print("\n🔍 Verifying converted model...")
            test_model = tf.keras.models.load_model(output_path, compile=False)
            print(f"✅ Model verification successful!")
            print(f"   - Input shape: {test_model.input_shape if hasattr(test_model, 'input_shape') else 'Multiple inputs'}")
            print(f"   - Output shape: {test_model.output_shape if hasattr(test_model, 'output_shape') else 'Multiple outputs'}")
            
            return True
    
    except Exception as e:
        print(f"❌ Model conversion failed: {e}")
        return False

def main():
    """Main function"""
    print("🔧 TensorFlow/Keras Model Compatibility Fixer")
    print("=" * 50)
    
    # Check if model file exists
    model_path = "models/scanner_hybrid.keras"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model file exists in the models/ directory")
        return
    
    print(f"📁 Found model file: {model_path}")
    
    # Convert model
    success = convert_model(model_path)
    
    if success:
        print("\n🎉 SUCCESS! Model has been converted successfully!")
        print("\n📋 Next steps:")
        print("1. Replace the original model file with the converted one")
        print("2. Restart your Streamlit application")
        print("3. The model should now load without compatibility issues")
    else:
        print("\n❌ FAILED! Model conversion was not successful.")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check TensorFlow version compatibility")
        print("2. Try reinstalling TensorFlow: pip install --upgrade tensorflow")
        print("3. Consider retraining the model with current TensorFlow version")

if __name__ == "__main__":
    main()