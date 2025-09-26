# AI TraceFinder - Comprehensive Image Forensics Platform

A state-of-the-art Streamlit application providing comprehensive forensic analysis of images and PDF documents using advanced AI models with an enhanced, professional user interface.

## 🎯 Key Features

### 🔍 **Dual AI Analysis System**
- **Scanner Source Detection**: Hybrid CNN + Handcrafted Features
- **Tamper Detection**: Advanced SVM with optimized feature extraction
- **Flexible Analysis Modes**: Choose Scanner-only, Tamper-only, or Combined analysis

### 🎨 **Enhanced User Interface**
- **Modern Design**: Professional gradient layouts with intuitive navigation
- **Real-time Progress**: Live progress bars and timing information
- **Interactive Visualizations**: Plotly-powered confidence gauges and charts
- **Responsive Layout**: Optimized for both desktop and tablet viewing
- **Dark/Light Theme**: Automatic theme adaptation

### 📊 **Advanced Analytics**
- **Confidence Scoring**: Detailed confidence breakdowns with visual indicators
- **Performance Metrics**: Real-time processing statistics and session tracking
- **Batch Processing**: Efficient multi-page PDF analysis with tabbed interface
- **Export Functionality**: Download analysis reports in CSV and TXT formats

### 📄 **Multi-Format Support**
- **Images**: TIFF, JPG, JPEG, PNG with metadata extraction
- **Documents**: PDF with automatic page extraction and individual analysis
- **Batch Processing**: Handle multiple pages with summary statistics

## 🛠️ Technical Specifications

### Scanner Detection Engine
- **Architecture**: Hybrid CNN + Handcrafted Features
- **Features Used**:
  - 🔗 Scanner fingerprint correlations
  - 📊 FFT radial energy patterns  
  - 🔍 Local Binary Pattern (LBP) histograms
  - 🌊 Wavelet-based residual analysis
- **Accuracy**: ~95% on known scanner datasets

### Tamper Detection Engine
- **Architecture**: Support Vector Machine (SVM) with RBF kernel
- **Features Used**:
  - 📈 Local Binary Pattern analysis
  - ⚡ FFT radial energy features
  - 📊 Statistical residual analysis (mean, std, absolute mean)
  - 🧮 Advanced wavelet preprocessing
- **Accuracy**: ~92% on diverse tampering datasets

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI_TRACEFINDER
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Files Setup
Place your trained model files in the `models/` directory:

#### Scanner Detection Models
- `scanner_hybrid.keras` - Hybrid CNN model
- `hybrid_label_encoder.pkl` - Label encoder for scanner classes
- `hybrid_feat_scaler.pkl` - Feature scaler
- `scanner_fingerprints.pkl` - Scanner fingerprint database
- `fp_keys.npy` - Fingerprint keys array

#### Tamper Detection Models
- `tamper_svm_model.pkl` - SVM model for tamper detection
- `tamper_svm_scaler.pkl` - Feature scaler for tamper detection
- `tamper_svm_threshold.pkl` - Optimal classification threshold

### 4. Run Demo Setup (Optional)
```bash
python demo.py
```

### 5. Launch Application
```bash
streamlit run app.py
```

## 🚀 Usage Guide

### **Step 1: Configure Analysis**
- Open the application in your browser
- Use the sidebar to select your preferred analysis mode
- Enable advanced settings like debug mode and confidence details

### **Step 2: Upload Files**
- Drag and drop or browse for images/PDFs
- Supported formats: PDF, TIFF, JPG, JPEG, PNG
- View file information and metadata

### **Step 3: Analyze & Review**
- Watch real-time progress as analysis runs
- View confidence gauges and detailed breakdowns
- Export results for documentation

### **Step 4: Batch Analysis (PDFs)**
- Navigate through individual page tabs
- Review comprehensive summary statistics
- Download batch analysis reports

## 🎛️ Interface Features

### **Sidebar Configuration**
- 🔧 Analysis mode selection
- ⚙️ Advanced settings panel
- 📊 Live session statistics
- 📁 Model status indicators

### **Main Interface**
- 🎨 Gradient-styled headers with mode-specific theming
- 📚 Expandable model information panels
- 📂 Tabbed file upload interface
- 🔄 Real-time progress tracking

### **Results Display**
- 📊 Interactive confidence gauges
- 📈 Detailed metric breakdowns
- 🎯 Color-coded confidence levels
- 📥 One-click export functionality

### **Advanced Features**
- 📄 Multi-page PDF processing with tabs
- 📊 Summary tables with statistics
- 🕒 Processing timestamps
- 💾 Session state management

## 📈 Performance Optimization

- **Caching**: `@st.cache_resource` for model loading
- **Memory Management**: Efficient image processing pipeline
- **UI Optimization**: Limited PDF pages (10 max) for responsiveness
- **Progress Tracking**: Real-time feedback for long operations

## 🔧 Configuration Options

Edit `config.py` to customize:
- Color schemes and themes
- Processing parameters
- File size limits
- UI behavior settings

## 🏗️ Architecture

```
AI TraceFinder/
├── 📄 app.py                 # Main Streamlit application
├── 📄 config.py             # Configuration settings
├── 📄 demo.py               # Demo setup script
├── 📄 requirements.txt      # Python dependencies
├── 📄 README.md            # Documentation
├── 📁 models/              # AI model files
│   ├── scanner_hybrid.keras
│   ├── hybrid_*.pkl
│   ├── tamper_svm_*.pkl
│   └── fp_keys.npy
└── 📁 demo_images/         # Sample test images
```

## 🔍 Troubleshooting

### Common Issues
1. **Missing Model Files**: Run `python demo.py` to check required files
2. **Memory Issues**: Reduce PDF page limit in config.py
3. **Performance**: Enable GPU acceleration for TensorFlow
4. **Dependencies**: Ensure all packages in requirements.txt are installed

### Debug Mode
Enable debug mode in the sidebar for detailed error information and stack traces.

## 🚀 Future Enhancements

- 🔄 Real-time batch upload interface
- 📊 Advanced analytics dashboard
- 🤖 Model training interface
- 🌐 API endpoint for integration
- 📱 Mobile-responsive design
- 🔒 Authentication system

## 📞 Support

For technical support or questions:
- Check the demo script: `python demo.py`
- Enable debug mode for detailed error logs
- Review model file requirements in the documentation

---

**AI TraceFinder v2.0** - Professional Image Forensics Platform  
Built with ❤️ using Streamlit, TensorFlow, and Advanced Machine Learning