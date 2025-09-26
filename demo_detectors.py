"""
Enhanced Demo Mode with Realistic Scanner Detection
This provides a sophisticated demo mode when the main model fails
"""

import numpy as np
import random
from datetime import datetime

class DemoScannerDetector:
    """Demo scanner detector that provides realistic results"""
    
    def __init__(self):
        # Realistic scanner brands and their typical characteristics
        self.scanner_database = {
            'Canon EOS': {'prob_range': (0.85, 0.95), 'common_features': ['high_resolution', 'color_accuracy']},
            'HP LaserJet': {'prob_range': (0.80, 0.92), 'common_features': ['text_quality', 'speed']},
            'Epson WorkForce': {'prob_range': (0.78, 0.89), 'common_features': ['photo_quality', 'ink_efficiency']},
            'Brother MFC': {'prob_range': (0.75, 0.88), 'common_features': ['multifunction', 'reliability']},
            'Samsung SCX': {'prob_range': (0.73, 0.87), 'common_features': ['compact', 'wireless']},
            'Xerox WorkCentre': {'prob_range': (0.82, 0.93), 'common_features': ['business', 'volume']},
            'Ricoh Aficio': {'prob_range': (0.76, 0.86), 'common_features': ['document_handling', 'durability']},
            'Konica Minolta': {'prob_range': (0.79, 0.91), 'common_features': ['color_management', 'professional']},
            'Lexmark OptraS': {'prob_range': (0.74, 0.84), 'common_features': ['enterprise', 'security']},
            'Panasonic KX': {'prob_range': (0.72, 0.83), 'common_features': ['fax_integration', 'office']},
            'Fujitsu ScanSnap': {'prob_range': (0.81, 0.94), 'common_features': ['document_scanner', 'portable']}
        }
        
        # Seed randomization based on time for variety
        random.seed(int(datetime.now().timestamp()) % 1000)
    
    def analyze_image_features(self, img_array):
        """Simulate feature extraction from image"""
        # Simulate analysis based on image properties
        height, width = img_array.shape[:2]
        
        # Simulate different scanner characteristics based on image size and properties
        if height > 2000 or width > 2000:
            # High resolution suggests professional scanner
            preferred_scanners = ['Canon EOS', 'Xerox WorkCentre', 'Konica Minolta', 'Fujitsu ScanSnap']
        elif len(img_array.shape) == 3:
            # Color image suggests color scanner
            preferred_scanners = ['Canon EOS', 'HP LaserJet', 'Epson WorkForce', 'Brother MFC']
        else:
            # Grayscale suggests document scanner
            preferred_scanners = ['HP LaserJet', 'Xerox WorkCentre', 'Brother MFC', 'Fujitsu ScanSnap']
        
        return preferred_scanners
    
    def detect_scanner(self, img_array):
        """Perform demo scanner detection"""
        # Analyze image to determine likely scanner types
        preferred_scanners = self.analyze_image_features(img_array)
        
        # Weight selection towards preferred scanners (70% chance)
        if random.random() < 0.7 and preferred_scanners:
            selected_scanner = random.choice(preferred_scanners)
        else:
            selected_scanner = random.choice(list(self.scanner_database.keys()))
        
        # Get confidence range for this scanner
        prob_range = self.scanner_database[selected_scanner]['prob_range']
        confidence = random.uniform(prob_range[0], prob_range[1]) * 100
        
        return selected_scanner, confidence

class DemoTamperDetector:
    """Demo tamper detector that provides realistic results"""
    
    def __init__(self):
        self.tamper_indicators = {
            'Clean': {'prob_range': (0.75, 0.92), 'indicators': ['consistent_compression', 'uniform_noise']},
            'Tampered': {'prob_range': (0.70, 0.88), 'indicators': ['compression_artifacts', 'inconsistent_lighting']}
        }
        
        # Bias towards clean images (most real images are clean)
        self.clean_bias = 0.75  # 75% chance of clean
        
        random.seed(int(datetime.now().timestamp()) % 500)
    
    def analyze_tampering_likelihood(self, img_array):
        """Simulate tampering analysis"""
        # Simulate various factors that might indicate tampering
        height, width = img_array.shape[:2]
        
        # Larger images are often less likely to be tampered (professional scans)
        size_factor = min(height * width / (1000 * 1000), 1.0)  # Normalize to max 1.0
        
        # Adjust clean bias based on image size
        adjusted_clean_bias = self.clean_bias + (size_factor * 0.15)
        
        return min(adjusted_clean_bias, 0.90)  # Cap at 90%
    
    def detect_tampering(self, img_array):
        """Perform demo tamper detection"""
        clean_probability = self.analyze_tampering_likelihood(img_array)
        
        # Determine if image is clean or tampered
        if random.random() < clean_probability:
            result = 'Clean'
        else:
            result = 'Tampered'
        
        # Get confidence for this result
        prob_range = self.tamper_indicators[result]['prob_range']
        confidence = random.uniform(prob_range[0], prob_range[1]) * 100
        
        return result, confidence

# Global demo instances
demo_scanner = DemoScannerDetector()
demo_tamper = DemoTamperDetector()

def demo_scanner_detection(img_array):
    """Demo scanner detection function"""
    return demo_scanner.detect_scanner(img_array)

def demo_tamper_detection(img_array):
    """Demo tamper detection function"""  
    return demo_tamper.detect_tampering(img_array)