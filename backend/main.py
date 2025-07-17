from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
import io
from typing import Optional

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
        
        # Step 2: Get full text for PII detection
        full_text = ocr_processor.get_full_text(extracted_text)
        logger.info(f"Extracted text: {full_text[:100]}...")
        
        # Step 3: Detect PII in the text
        logger.info("Detecting PII...")
        detected_pii = pii_detector.get_all_pii(full_text)
        
        # Step 4: Map PII detection to image regions
        pii_regions = []
        for pii_item in detected_pii:
            # Find corresponding text regions in the image
            for text_item in extracted_text:
                if pii_item['text'].lower() in text_item['text'].lower():
                    pii_regions.append({
                        'type': pii_item['type'],
                        'text': pii_item['text'],
                        'bbox': text_item['bbox'],
                        'confidence': pii_item['confidence']
                    })
                    break
        
        logger.info(f"Found {len(pii_regions)} PII regions to mask")
        
        # Step 5: Mask the image
        logger.info(f"Masking image using method: {mask_method}")
        masked_image = image_masker.mask_pii_regions(
            image, pii_regions, method=mask_method, padding=padding
        )
        
        # Step 6: Add indicators if requested
        if show_indicators:
            masked_image = image_masker.add_redaction_indicators(masked_image, pii_regions)
        
        # Step 7: Convert to base64
        masked_image_b64 = image_masker.image_to_base64(masked_image)
        
        # Step 8: Get statistics
        stats = image_masker.get_masking_statistics(pii_regions)
        
        # Step 9: Create comparison image
        comparison_image = image_masker.create_comparison_image(image, masked_image)
        comparison_b64 = image_masker.image_to_base64(comparison_image)
        
        logger.info(f"Processing completed. Masked {stats['total_regions']} regions")
        
        return JSONResponse({
            "message": "Image processed successfully",
            "masked_image": masked_image_b64,
            "comparison_image": comparison_b64,
            "statistics": stats,
            "detected_pii": pii_regions,
            "extracted_text": full_text
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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