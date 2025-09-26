@echo off
echo 🚀 Starting AI TraceFinder...
echo.
echo Checking dependencies...
python -c "import streamlit; print('✅ Streamlit OK')" 2>nul || (
    echo ❌ Streamlit not found. Installing...
    pip install streamlit
)

echo.
echo Starting application...
echo 🌐 Your browser should open automatically
echo 📍 If not, visit: http://localhost:8501
echo.
streamlit run app.py

pause