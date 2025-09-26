# Streamlit Cloud Emergency Deployment Guide

## If requirements.txt keeps failing:

### Step 1: Use Minimal Requirements
Replace the contents of `requirements.txt` with just:
```
streamlit
numpy
pillow
```

### Step 2: The app will automatically:
- ✅ Detect missing dependencies  
- ✅ Run in full demo mode
- ✅ Show realistic simulated results
- ✅ Display all UI features

### Step 3: Gradually add dependencies
Once the basic app works, you can add dependencies one by one:

**First add:**
```
streamlit
numpy
pillow
pandas
```

**Then add:**
```
streamlit
numpy  
pillow
pandas
plotly
```

**Finally add (if needed):**
```
streamlit
numpy
pillow
pandas
plotly
opencv-python-headless
scikit-learn
```

### Common Issues:
1. **TensorFlow**: Use `tensorflow-cpu` not `tensorflow`
2. **OpenCV**: Use `opencv-python-headless` not `opencv-python`  
3. **Versions**: Avoid pinning versions, let Streamlit Cloud choose
4. **PyMuPDF**: Can cause conflicts, add last or skip for now

### Emergency Deploy:
If all else fails, just use:
```
streamlit
numpy
pillow
```

The app is designed to work with just these 3 packages in demo mode!