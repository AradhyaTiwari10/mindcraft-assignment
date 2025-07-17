from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import logging
import io
import json
import time
from typing import Optional, List
from datetime import datetime
import asyncio
from collections import defaultdict

from ocr import OCRProcessor
from advanced_pii_detector import AdvancedPIIDetector, PIIEntity
from advanced_masker import AdvancedImageMasker, MaskingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced PII Masking System",
    description="Advanced system to detect and mask Personally Identifiable Information from images with ML capabilities",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analytics
analytics = {
    'total_requests': 0,
    'successful_masks': 0,
    'failed_masks': 0,
    'average_processing_time': 0,
    'document_types': defaultdict(int),
    'pii_types_detected': defaultdict(int),
    'masking_methods_used': defaultdict(int)
}

# Initialize components
try:
    ocr_processor = OCRProcessor()
    advanced_pii_detector = AdvancedPIIDetector()
    advanced_masker = AdvancedImageMasker()
    logger.info("All advanced components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

@app.get("/")
async def root():
    """Root endpoint with advanced system information."""
    return {
        "message": "Advanced PII Masking System API",
        "version": "2.0.0",
        "features": [
            "Advanced PII detection with ML models",
            "Face detection and masking",
            "QR code detection and masking",
            "Multiple masking techniques",
            "Batch processing",
            "Real-time analytics",
            "Document type detection"
        ],
        "endpoints": {
            "mask": "/mask - Upload image and get masked version",
            "batch_mask": "/batch-mask - Process multiple images",
            "analytics": "/analytics - Get system analytics",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Advanced health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ocr": "initialized",
            "advanced_pii_detector": "initialized",
            "advanced_masker": "initialized",
            "face_detection": advanced_masker.use_face_detection,
            "spacy_ner": advanced_pii_detector.use_spacy
        },
        "analytics": {
            "total_requests": analytics['total_requests'],
            "success_rate": f"{(analytics['successful_masks'] / max(analytics['total_requests'], 1)) * 100:.1f}%"
        }
    }

@app.get("/analytics")
async def get_analytics():
    """Get system analytics and performance metrics."""
    return {
        "overview": {
            "total_requests": analytics['total_requests'],
            "successful_masks": analytics['successful_masks'],
            "failed_masks": analytics['failed_masks'],
            "success_rate": f"{(analytics['successful_masks'] / max(analytics['total_requests'], 1)) * 100:.1f}%",
            "average_processing_time": f"{analytics['average_processing_time']:.2f}s"
        },
        "document_types": dict(analytics['document_types']),
        "pii_types_detected": dict(analytics['pii_types_detected']),
        "masking_methods_used": dict(analytics['masking_methods_used']),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/mask")
async def advanced_mask_pii(
    file: UploadFile = File(...),
    mask_method: str = Form("black_box"),
    show_indicators: bool = Form(True),
    padding: int = Form(5),
    mask_faces: bool = Form(True),
    mask_qr_codes: bool = Form(True),
    confidence_threshold: float = Form(0.6)
):
    """
    Advanced PII masking with ML capabilities.
    
    Args:
        file: Image file (JPG, PNG)
        mask_method: Masking method ('black_box', 'blur', 'pixelate', 'solid_color', 'gaussian_noise', 'mosaic', 'inpaint')
        show_indicators: Whether to show redaction indicators
        padding: Extra padding around masked regions
        mask_faces: Whether to mask detected faces
        mask_qr_codes: Whether to mask detected QR codes
        confidence_threshold: Minimum confidence for PII detection
        
    Returns:
        JSON response with masked image and advanced statistics
    """
    start_time = time.time()
    analytics['total_requests'] += 1
    
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
            analytics['failed_masks'] += 1
            return JSONResponse({
                "message": "No text found in image",
                "masked_image": advanced_masker.image_to_base64(image),
                "statistics": {
                    "total_regions": 0,
                    "by_type": {},
                    "faces_detected": 0,
                    "qr_codes_detected": 0,
                    "processing_time": 0
                }
            })
        
        # Step 2: Get full text for PII detection
        full_text = ocr_processor.get_full_text(extracted_text)
        logger.info(f"Extracted text: {full_text[:100]}...")
        
        # Step 3: Advanced PII detection
        logger.info("Detecting PII with advanced methods...")
        detected_pii_entities = advanced_pii_detector.get_all_pii(full_text)
        
        # Filter by confidence threshold
        filtered_entities = [entity for entity in detected_pii_entities if entity.confidence >= confidence_threshold]
        
        # Step 4: Map PII detection to image regions
        pii_regions = []
        for entity in filtered_entities:
            # Find corresponding text regions in the image
            for text_item in extracted_text:
                if entity.text.lower() in text_item['text'].lower():
                    pii_regions.append({
                        'type': entity.type,
                        'text': entity.text,
                        'bbox': text_item['bbox'],
                        'confidence': entity.confidence,
                        'context': entity.context
                    })
                    break
        
        logger.info(f"Found {len(pii_regions)} PII regions to mask")
        
        # Step 5: Advanced masking
        logger.info(f"Masking image using method: {mask_method}")
        masking_result = advanced_masker.mask_pii_regions(
            image, pii_regions, method=mask_method, padding=padding,
            mask_faces=mask_faces, mask_qr_codes=mask_qr_codes
        )
        
        # Step 6: Add indicators if requested
        if show_indicators:
            masked_image = advanced_masker.add_advanced_indicators(
                masking_result.masked_image, pii_regions,
                masking_result.detected_faces, masking_result.detected_qr_codes
            )
        else:
            masked_image = masking_result.masked_image
        
        # Step 7: Convert to base64
        masked_image_b64 = advanced_masker.image_to_base64(masked_image)
        
        # Step 8: Create advanced comparison
        comparison_image = advanced_masker.create_advanced_comparison(
            image, masked_image, masking_result.statistics
        )
        comparison_b64 = advanced_masker.image_to_base64(comparison_image)
        
        # Step 9: Update analytics
        processing_time = time.time() - start_time
        analytics['successful_masks'] += 1
        analytics['masking_methods_used'][mask_method] += 1
        
        # Update average processing time
        total_time = analytics['average_processing_time'] * (analytics['successful_masks'] - 1) + processing_time
        analytics['average_processing_time'] = total_time / analytics['successful_masks']
        
        # Update PII type statistics
        for region in pii_regions:
            analytics['pii_types_detected'][region['type']] += 1
        
        logger.info(f"Advanced processing completed. Masked {masking_result.statistics['total_regions']} regions in {processing_time:.2f}s")
        
        return JSONResponse({
            "message": "Advanced image processing completed successfully",
            "masked_image": masked_image_b64,
            "comparison_image": comparison_b64,
            "statistics": masking_result.statistics,
            "detected_pii": pii_regions,
            "extracted_text": full_text,
            "processing_time": processing_time,
            "document_type": advanced_pii_detector.detect_document_type(full_text),
            "advanced_features": {
                "face_detection": len(masking_result.detected_faces),
                "qr_code_detection": len(masking_result.detected_qr_codes),
                "ml_enhanced_detection": advanced_pii_detector.use_spacy,
                "confidence_scoring": True
            }
        })
        
    except HTTPException:
        analytics['failed_masks'] += 1
        raise
    except Exception as e:
        analytics['failed_masks'] += 1
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch-mask")
async def batch_mask_pii(
    files: List[UploadFile] = File(...),
    mask_method: str = Form("black_box"),
    show_indicators: bool = Form(True),
    padding: int = Form(5),
    mask_faces: bool = Form(True),
    mask_qr_codes: bool = Form(True),
    confidence_threshold: float = Form(0.6)
):
    """
    Batch process multiple images for PII masking.
    
    Args:
        files: List of image files
        mask_method: Masking method
        show_indicators: Whether to show redaction indicators
        padding: Extra padding around masked regions
        mask_faces: Whether to mask detected faces
        mask_qr_codes: Whether to mask detected QR codes
        confidence_threshold: Minimum confidence for PII detection
        
    Returns:
        JSON response with batch processing results
    """
    batch_results = []
    total_start_time = time.time()
    
    for i, file in enumerate(files):
        try:
            # Process each file individually
            result = await advanced_mask_pii(
                file=file,
                mask_method=mask_method,
                show_indicators=show_indicators,
                padding=padding,
                mask_faces=mask_faces,
                mask_qr_codes=mask_qr_codes,
                confidence_threshold=confidence_threshold
            )
            
            batch_results.append({
                "filename": file.filename,
                "status": "success",
                "result": result.body.decode() if hasattr(result, 'body') else result
            })
            
        except Exception as e:
            batch_results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    total_time = time.time() - total_start_time
    
    return JSONResponse({
        "message": f"Batch processing completed for {len(files)} files",
        "total_processing_time": total_time,
        "successful": len([r for r in batch_results if r['status'] == 'success']),
        "failed": len([r for r in batch_results if r['status'] == 'failed']),
        "results": batch_results
    })

@app.post("/test")
async def test_endpoint():
    """Test endpoint for development."""
    return {
        "message": "Advanced PII Masking System is working",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 