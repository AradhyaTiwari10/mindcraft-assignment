#!/usr/bin/env python3
"""
Test script for PII detection system.
This script helps debug and verify the PII detection is working correctly.
"""

import cv2
import numpy as np
import logging
from ocr import OCRProcessor
from pii_detector import PIIDetector
from masker import ImageMasker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pii_detection(image_path: str):
    """
    Test PII detection on a sample image.
    
    Args:
        image_path: Path to the test image
    """
    try:
        # Load image
        logger.info(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        logger.info(f"Image loaded successfully. Shape: {image.shape}")
        
        # Initialize components
        logger.info("Initializing components...")
        ocr_processor = OCRProcessor()
        pii_detector = PIIDetector()
        image_masker = ImageMasker()
        
        # Step 1: Extract text using OCR
        logger.info("Extracting text using OCR...")
        extracted_text = ocr_processor.extract_text(image)
        
        if not extracted_text:
            logger.warning("No text extracted from image")
            return
        
        logger.info(f"Extracted {len(extracted_text)} text regions")
        
        # Display extracted text regions
        logger.info("Extracted text regions:")
        for i, region in enumerate(extracted_text):
            logger.info(f"  {i+1}. '{region['text']}' at {region['bbox']} (confidence: {region['confidence']:.2f})")
        
        # Step 2: Detect PII
        logger.info("Detecting PII...")
        detected_pii = pii_detector.detect_pii_in_ocr_regions(extracted_text)
        
        logger.info(f"Found {len(detected_pii)} PII regions")
        
        # Display detected PII
        if detected_pii:
            logger.info("Detected PII:")
            for i, pii in enumerate(detected_pii):
                logger.info(f"  {i+1}. {pii['type']}: '{pii['text']}' (confidence: {pii['confidence']:.2f})")
        else:
            logger.warning("No PII detected!")
        
        # Step 3: Test masking
        logger.info("Testing masking...")
        masked_image = image_masker.mask_pii_regions(image, detected_pii, method='black_box', padding=5)
        
        # Save results
        output_path = image_path.replace('.', '_masked.')
        cv2.imwrite(output_path, masked_image)
        logger.info(f"Masked image saved to: {output_path}")
        
        # Create comparison image
        comparison = image_masker.create_comparison_image(image, masked_image)
        comparison_path = image_path.replace('.', '_comparison.')
        cv2.imwrite(comparison_path, comparison)
        logger.info(f"Comparison image saved to: {comparison_path}")
        
        # Get statistics
        stats = image_masker.get_masking_statistics(detected_pii)
        logger.info(f"Masking statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_with_sample_aadhaar():
    """Test with the sample Aadhaar image."""
    test_pii_detection("sample_images/sample_aadhaar.jpg")

def test_with_sample_driving_license():
    """Test with the sample driving license image."""
    test_pii_detection("sample_images/sample_driving_license.jpg")

if __name__ == "__main__":
    logger.info("Starting PII detection test...")
    
    # Test with sample images
    try:
        test_with_sample_aadhaar()
        print("\n" + "="*50 + "\n")
        test_with_sample_driving_license()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 