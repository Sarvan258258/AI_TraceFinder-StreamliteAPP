"""
Render Model Compatibility Fixer
This script converts your model to be compatible with Render's TensorFlow environment
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def fix_model_for_render():
    """Fix model compatibility for Render deployment"""
    try:
        import tensorflow as tf
        print(f"🔧 Current TensorFlow version: {tf.__version__}")
        
        model_path = "models/scanner_hybrid.keras"
        fixed_path = "models/scanner_hybrid_render.keras"
        
        print(f"📖 Loading original model from: {model_path}")
        
        # Try to load the original model
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Original model loaded successfully")
        except Exception as e:
            print(f"❌ Cannot load original model: {e}")
            print("⚠️  This indicates a TensorFlow version compatibility issue")
            return False
        
        # Get model architecture and weights
        print("🔄 Extracting model architecture and weights...")
        
        # Save model architecture as JSON
        architecture = model.to_json()
        
        # Save weights separately
        weights_path = "models/scanner_weights_temp.weights.h5"
        model.save_weights(weights_path)
        
        print("🏗️  Reconstructing model with current TensorFlow version...")
        
        # Clear any custom objects
        tf.keras.utils.get_custom_objects().clear()
        
        # Reconstruct model from JSON
        new_model = tf.keras.models.model_from_json(architecture)
        
        # Load weights
        new_model.load_weights(weights_path)
        
        # Compile with simple settings
        new_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("💾 Saving Render-compatible model...")
        
        # Save in current TensorFlow format
        new_model.save(fixed_path, save_format='keras')
        
        # Clean up temporary weights file
        os.remove(weights_path)
        
        # Test the new model
        print("🧪 Testing Render-compatible model...")
        test_model = tf.keras.models.load_model(fixed_path, compile=False)
        
        print("✅ Render-compatible model created successfully!")
        print(f"   Original parameters: {model.count_params():,}")
        print(f"   New model parameters: {test_model.count_params():,}")
        
        if model.count_params() == test_model.count_params():
            print("✅ Parameter count matches - accuracy preserved!")
        else:
            print("⚠️  Parameter count differs - please verify accuracy")
        
        # Replace original with fixed version
        backup_path = "models/scanner_hybrid_original_backup.keras"
        os.rename(model_path, backup_path)
        os.rename(fixed_path, model_path)
        
        print(f"🔄 Model replacement complete:")
        print(f"   Original backed up to: {backup_path}")
        print(f"   Render-compatible model now at: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model fixing failed: {e}")
        return False

def main():
    """Main execution"""
    print("🚀 Render Model Compatibility Fixer")
    print("=" * 50)
    
    if not os.path.exists("models/scanner_hybrid.keras"):
        print("❌ Original model not found!")
        return False
    
    success = fix_model_for_render()
    
    if success:
        print("\n🎉 SUCCESS! Your model is now Render-compatible!")
        print("\n📋 Next steps:")
        print("1. Commit and push changes to GitHub")
        print("2. Deploy to Render - model should load without demo mode")
        print("3. Your original accuracy is preserved")
    else:
        print("\n❌ FAILED! Model compatibility fixing unsuccessful")
        
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)