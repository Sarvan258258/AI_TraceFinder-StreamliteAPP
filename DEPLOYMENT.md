# Demo Mode Configuration for Streamlit Cloud Deployment

This file helps the application run in demo mode when model files are not available.

## For Streamlit Cloud Deployment

When deploying to Streamlit Cloud, if you don't have the actual trained model files, the application will run in demo mode with:

1. **Mock predictions** - Simulated results for demonstration
2. **Reduced functionality** - Only basic UI features
3. **Educational content** - Information about the AI models

## Model Files (Not included in repository for size reasons)

To run with full functionality, add these files to the `models/` directory:

### Scanner Detection Models
- `scanner_hybrid.keras` (TensorFlow/Keras CNN model)
- `hybrid_label_encoder.pkl` (Label encoder)
- `hybrid_feat_scaler.pkl` (Feature scaler)
- `scanner_fingerprints.pkl` (Scanner fingerprint database)
- `fp_keys.npy` (Fingerprint keys array)

### Tamper Detection Models
- `tamper_svm_model.pkl` (SVM model)
- `tamper_svm_scaler.pkl` (Feature scaler)
- `tamper_svm_threshold.pkl` (Classification threshold)

## Deployment Notes

- Model files should be uploaded separately to your cloud deployment
- Consider using cloud storage services for large model files
- Use environment variables for model paths in production