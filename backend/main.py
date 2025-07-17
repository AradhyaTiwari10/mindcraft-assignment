from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
import io
from typing import Optional
import re

from ocr import OCRProcessor
from pii_detector import PIIDetector
from masker import ImageMasker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PII Masking System",
    description="A system to detect and mask Personally Identifiable Information from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    ocr_processor = OCRProcessor()
    pii_detector = PIIDetector()
    image_masker = ImageMasker()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "PII Masking System API",
        "version": "1.0.0",
        "endpoints": {
            "mask": "/mask - Upload image and get masked version",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "ocr": "initialized",
            "pii_detector": "initialized",
            "image_masker": "initialized"
        }
    }

@app.post("/mask")
async def mask_pii(
    file: UploadFile = File(...),
    mask_method: str = Form("black_box"),
    show_indicators: bool = Form(False),
    padding: int = Form(5)
):
    """
    Upload an image and get back a masked version with PII redacted.
    
    Args:
        file: Image file (JPG, PNG)
        mask_method: Masking method ('black_box', 'blur', 'pixelate', 'solid_color')
        show_indicators: Whether to show redaction indicators
        padding: Extra padding around masked regions
        
    Returns:
        JSON response with masked image and statistics
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Processing image: {file.filename}, size: {image.shape}")
        
        # Step 1: Extract text using OCR
        logger.info("Extracting text using OCR...")
        extracted_text = ocr_processor.extract_text(image)
        
        if not extracted_text:
            logger.warning("No text extracted from image")
            return JSONResponse({
                "message": "No text found in image",
                "masked_image": image_masker.image_to_base64(image),
                "statistics": {
                    "total_regions": 0,
                    "by_type": {},
                    "total_characters": 0
                }
            })
        
        # Debug: Log extracted text regions
        logger.info(f"Extracted {len(extracted_text)} text regions")
        for i, region in enumerate(extracted_text[:10]):  # Log first 10 regions
            logger.info(f"Region {i}: '{region['text']}' at {region['bbox']}")
        
        # Step 2: Detect PII directly in OCR regions (NEW APPROACH)
        logger.info("Detecting PII in OCR regions...")
        detected_pii = pii_detector.detect_pii_in_ocr_regions(extracted_text)
        
        logger.info(f"Found {len(detected_pii)} PII regions to mask")
        pii_info = [f"{r['type']}: '{r['text']}'" for r in detected_pii]
        logger.info(f"Detected PII: {pii_info}")
        
        # Enhanced debugging: Log all text regions for analysis
        logger.info("=== ALL EXTRACTED TEXT REGIONS ===")
        for i, region in enumerate(extracted_text):
            logger.info(f"Region {i+1}: '{region['text']}' | Position: {region['bbox']} | Confidence: {region['confidence']:.2f}")
        
        # Log specific name-related regions
        logger.info("=== NAME-RELATED REGIONS ===")
        for i, region in enumerate(extracted_text):
            text_lower = region['text'].lower()
            if any(keyword in text_lower for keyword in ['name', 'son', 'daughter', 'wife', 's/o', 'd/o', 'w/o']):
                logger.info(f"Potential name region {i+1}: '{region['text']}' | Position: {region['bbox']}")
        
        # Log DOB regions for positional analysis
        logger.info("=== DOB REGIONS ===")
        for i, region in enumerate(extracted_text):
            if re.search(r'\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b', region['text']):
                logger.info(f"DOB region {i+1}: '{region['text']}' | Position: {region['bbox']}")
        
        # Log colon-containing regions for form field detection
        logger.info("=== COLON-CONTAINING REGIONS ===")
        for i, region in enumerate(extracted_text):
            if ':' in region['text']:
                logger.info(f"Colon region {i+1}: '{region['text']}' | Position: {region['bbox']}")
        
        # Log regions that might be split names
        logger.info("=== POTENTIAL SPLIT NAME REGIONS ===")
        for i, region in enumerate(extracted_text):
            text = region['text'].strip()
            if (len(text) > 2 and len(text) < 20 and 
                text.replace(' ', '').isalpha() and 
                not any(skip in text.lower() for skip in ['date', 'birth', 'address', 'phone', 'email', 'government', 'india'])):
                logger.info(f"Potential name part {i+1}: '{region['text']}' | Position: {region['bbox']}")
        
        logger.info("=== END DEBUG INFO ===")
        
        # Step 3: Mask the image
        logger.info(f"Masking image using method: {mask_method}")
        masked_image = image_masker.mask_pii_regions(
            image, detected_pii, method=mask_method, padding=padding
        )
        
        # Step 4: Add indicators if requested
        if show_indicators:
            masked_image = image_masker.add_redaction_indicators(masked_image, detected_pii)
        
        # Step 5: Convert to base64
        masked_image_b64 = image_masker.image_to_base64(masked_image)
        
        # Step 6: Get statistics
        stats = image_masker.get_masking_statistics(detected_pii)
        
        # Step 7: Create comparison image
        comparison_image = image_masker.create_comparison_image(image, masked_image)
        comparison_b64 = image_masker.image_to_base64(comparison_image)
        
        return JSONResponse({
            "message": "PII masking completed successfully",
            "masked_image": masked_image_b64,
            "comparison_image": comparison_b64,
            "statistics": stats,
            "detected_pii": detected_pii
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/test")
async def test_endpoint():
    """Test endpoint for development."""
    return {
        "message": "Test endpoint working",
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 