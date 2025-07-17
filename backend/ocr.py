import easyocr
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

class OCRProcessor:
    """Handles OCR processing using EasyOCR for text extraction."""
    
    def __init__(self):
        """Initialize EasyOCR reader with English language."""
        try:
            # Initialize EasyOCR with English language
            self.reader = easyocr.Reader(['en'], gpu=False)
            logging.info("EasyOCR initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing text and bounding box information
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Perform OCR
            results = self.reader.readtext(processed_image)
            
            # Format results
            extracted_text = []
            for (bbox, text, confidence) in results:
                # Convert bbox to format: [x1, y1, x2, y2]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                bbox_formatted = [
                    int(min(x_coords)),  # x1
                    int(min(y_coords)),  # y1
                    int(max(x_coords)),  # x2
                    int(max(y_coords))   # y2
                ]
                
                extracted_text.append({
                    'text': text.strip(),
                    'bbox': bbox_formatted,
                    'confidence': float(confidence),
                    'center_x': int(sum(x_coords) / 4),
                    'center_y': int(sum(y_coords) / 4)
                })
            
            logging.info(f"Extracted {len(extracted_text)} text regions")
            return extracted_text
            
        except Exception as e:
            logging.error(f"Error during OCR processing: {e}")
            # Try with original image if preprocessing fails
            try:
                results = self.reader.readtext(image)
                extracted_text = []
                for (bbox, text, confidence) in results:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    bbox_formatted = [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    ]
                    
                    extracted_text.append({
                        'text': text.strip(),
                        'bbox': bbox_formatted,
                        'confidence': float(confidence),
                        'center_x': int(sum(x_coords) / 4),
                        'center_y': int(sum(y_coords) / 4)
                    })
                
                logging.info(f"Extracted {len(extracted_text)} text regions (fallback)")
                return extracted_text
            except Exception as e2:
                logging.error(f"Fallback OCR also failed: {e2}")
                return []
    
    def get_full_text(self, extracted_text: List[Dict]) -> str:
        """
        Combine all extracted text into a single string.
        
        Args:
            extracted_text: List of extracted text dictionaries
            
        Returns:
            Combined text string
        """
        # Sort by y-coordinate (top to bottom) and then by x-coordinate (left to right)
        sorted_text = sorted(extracted_text, key=lambda x: (x['center_y'], x['center_x']))
        
        # Combine text with spaces
        full_text = ' '.join([item['text'] for item in sorted_text])
        
        return full_text
    
    def filter_low_confidence(self, extracted_text: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """
        Filter out text with low confidence scores.
        
        Args:
            extracted_text: List of extracted text dictionaries
            threshold: Minimum confidence threshold
            
        Returns:
            Filtered list of text dictionaries
        """
        return [item for item in extracted_text if item['confidence'] >= threshold]
    
    def get_text_by_type(self, extracted_text: List[Dict], text_type: str) -> List[Dict]:
        """
        Get text items that match a specific type (e.g., numbers, names).
        
        Args:
            extracted_text: List of extracted text dictionaries
            text_type: Type of text to filter for ('number', 'name', etc.)
            
        Returns:
            Filtered list of text dictionaries
        """
        if text_type == 'number':
            # Filter for numeric text
            return [item for item in extracted_text if item['text'].replace(' ', '').isdigit()]
        elif text_type == 'name':
            # Filter for text that looks like names (contains letters and spaces)
            return [item for item in extracted_text if any(c.isalpha() for c in item['text'])]
        else:
            return extracted_text 