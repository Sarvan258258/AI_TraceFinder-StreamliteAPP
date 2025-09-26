# AI TraceFinder Configuration

# This file contains configuration settings for the AI TraceFinder application

## Application Settings
DEBUG_MODE = False
SHOW_CONFIDENCE_DETAILS = True
BATCH_PROCESSING = True

## Model Paths
MODELS_DIR = "models"
MAX_PDF_PAGES = 10  # Limit PDF pages for UI performance
DEFAULT_DPI = 300  # DPI for PDF conversion
IMAGE_SIZE = (256, 256)  # Standard image size for processing

## UI Settings
PAGE_TITLE = "AI TraceFinder"
PAGE_ICON = "🕵️"
LAYOUT = "wide"

## Analysis Types
ANALYSIS_MODES = [
    "Both (Scanner + Tamper Detection)",
    "Scanner Source Only", 
    "Tamper Detection Only"
]

## Supported File Types
SUPPORTED_FORMATS = ["pdf", "tif", "tiff", "jpg", "jpeg", "png"]

## Color Scheme
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72", 
    "accent": "#F18F01",
    "success": "#11998e",
    "warning": "#fc4a1a",
    "info": "#667eea"
}