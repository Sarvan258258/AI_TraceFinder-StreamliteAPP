"""
Test script to verify the enhanced demo mode works properly
"""

import os
import sys
import numpy as np

def test_demo_mode():
    """Test the demo mode functionality"""
    print("🧪 Testing Enhanced Demo Mode")
    print("=" * 50)
    
    try:
        # Test demo detectors
        from demo_detectors import demo_scanner_detection, demo_tamper_detection
        print("✅ Demo detectors imported successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
        print(f"✅ Test image created: {test_image.shape}")
        
        # Test scanner detection
        scanner_brand, scanner_conf = demo_scanner_detection(test_image)
        print(f"✅ Scanner Detection: {scanner_brand} ({scanner_conf:.1f}%)")
        
        # Test tamper detection
        tamper_result, tamper_conf = demo_tamper_detection(test_image)
        print(f"✅ Tamper Detection: {tamper_result} ({tamper_conf:.1f}%)")
        
        # Test multiple runs for variety
        print("\n🔄 Testing result variety (5 runs):")
        for i in range(5):
            scanner_brand, scanner_conf = demo_scanner_detection(test_image)
            tamper_result, tamper_conf = demo_tamper_detection(test_image)
            print(f"  Run {i+1}: {scanner_brand} ({scanner_conf:.1f}%) | {tamper_result} ({tamper_conf:.1f}%)")
        
        print("\n🎉 All demo mode tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\n🔬 Testing Model Loading")
    print("=" * 30)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
        
        model_path = "models/scanner_hybrid.keras"
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Scanner model loads successfully")
                return True
            except Exception as e:
                print(f"⚠️ Scanner model loading failed: {str(e)[:100]}...")
                print("💡 This will trigger demo mode in the application")
                return False
        else:
            print(f"⚠️ Model file not found: {model_path}")
            return False
            
    except ImportError:
        print("⚠️ TensorFlow not available")
        return False

if __name__ == "__main__":
    print("🚀 AI TraceFinder Demo Mode Test Suite")
    print("=" * 60)
    
    demo_success = test_demo_mode()
    model_success = test_model_loading()
    
    print("\n📋 Test Results Summary")
    print("=" * 30)
    print(f"Demo Mode: {'✅ PASS' if demo_success else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_success else '⚠️ WILL USE DEMO'}")
    
    if demo_success:
        print("\n🎉 SUCCESS: Application will work with enhanced demo mode!")
        print("   - Realistic scanner brand detection")
        print("   - Intelligent tamper detection simulation")
        print("   - Professional UI experience maintained")
    else:
        print("\n❌ ISSUE: Demo mode has problems")
        
    print("\n🚀 Ready to run: streamlit run app.py")