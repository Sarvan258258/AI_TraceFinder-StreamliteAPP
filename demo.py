"""
Demo script for AI TraceFinder
This script helps test the application with sample data
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_demo_images():
    """Create sample demo images for testing"""
    demo_dir = "demo_images"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    # Create sample images
    for i in range(3):
        # Create a sample image
        img = Image.new('RGB', (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add some patterns to make it interesting
        for _ in range(50):
            x1, y1 = random.randint(0, 512), random.randint(0, 512)
            x2, y2 = random.randint(0, 512), random.randint(0, 512)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
        
        # Add text
        try:
            draw.text((50, 50), f"Demo Image {i+1}", fill=(0, 0, 0))
        except:
            # If font loading fails, just continue
            pass
        
        # Save image
        img.save(os.path.join(demo_dir, f"demo_image_{i+1}.jpg"))
    
    print(f"✅ Created {3} demo images in {demo_dir}/")

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        "models/scanner_hybrid.keras",
        "models/hybrid_label_encoder.pkl", 
        "models/hybrid_feat_scaler.pkl",
        "models/scanner_fingerprints.pkl",
        "models/fp_keys.npy",
        "models/tamper_svm_model.pkl",
        "models/tamper_svm_scaler.pkl",
        "models/tamper_svm_threshold.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Please ensure all model files are in the 'models/' directory")
        return False
    else:
        print("✅ All required model files found!")
        return True

def run_demo():
    """Run demo setup"""
    print("🚀 AI TraceFinder Demo Setup")
    print("=" * 40)
    
    # Check model files
    model_status = check_model_files()
    print()
    
    # Create demo images
    create_demo_images()
    print()
    
    if model_status:
        print("🎉 Demo setup complete!")
        print("You can now run: streamlit run app.py")
    else:
        print("⚠️  Demo setup incomplete - missing model files")
        print("The application may not work properly without all model files")

if __name__ == "__main__":
    run_demo()