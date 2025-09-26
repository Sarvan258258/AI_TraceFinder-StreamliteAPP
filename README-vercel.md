# AI TraceFinder - Vercel Deployment

This is the Vercel-optimized version of AI TraceFinder with Flask backend.

## 🚀 Deploy to Vercel

### Method 1: One-Click Deploy
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Sarvan258258/AI_TraceFinder-StreamliteAPP)

### Method 2: Manual Deployment

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel --prod
   ```

3. **Or connect your GitHub repo to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Connect your GitHub account
   - Select this repository
   - Deploy automatically

## 📁 Project Structure

```
├── api/
│   └── index.py          # Flask API endpoint
├── models/               # AI model files
├── vercel.json          # Vercel configuration
├── requirements.txt     # Python dependencies
└── runtime.txt         # Python version
```

## 🔧 Configuration

- **Memory**: 1024MB (configurable in vercel.json)
- **Timeout**: 30 seconds
- **Runtime**: Python 3.11
- **Framework**: Flask (serverless functions)

## 🧠 AI Models

Make sure your model files are in the `models/` directory:
- `scanner_hybrid.keras`
- `hybrid_label_encoder.pkl`
- `hybrid_feat_scaler.pkl` 
- `scanner_fingerprints.pkl`
- `fp_keys.npy`
- `tamper_svm_model.pkl`
- `tamper_svm_scaler.pkl`
- `tamper_svm_threshold.pkl`

## 💡 Usage

Once deployed, visit your Vercel URL to access the web interface. Upload an image to get:
- 🖨️ Scanner source detection
- 🛡️ Tamper detection analysis

## 🔍 API Endpoints

- `GET /` - Web interface
- `POST /api/analyze` - Image analysis endpoint
- `GET /api/health` - Health check

## ⚠️ Limitations

- Vercel has a 50MB deployment size limit
- 30-second function timeout
- Cold start latency for first requests
- Simplified feature extraction (due to size constraints)

## 🔄 Migration from Streamlit

This version uses Flask instead of Streamlit for better Vercel compatibility. The core AI functionality remains the same.