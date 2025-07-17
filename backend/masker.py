import cv2
import numpy as np
from typing import List, Dict, Tuple
import base64
from PIL import Image, ImageDraw, ImageFont
import io

class ImageMasker:
    """Handles masking of PII in images using various techniques."""
    
    def __init__(self):
        """Initialize the image masker."""
        self.mask_methods = {
            'black_box': self._black_box_mask,
            'blur': self._blur_mask,
            'pixelate': self._pixelate_mask,
            'solid_color': self._solid_color_mask
        }
    
    def mask_pii_regions(self, image: np.ndarray, pii_regions: List[Dict], 
                        method: str = 'black_box', padding: int = 5) -> np.ndarray:
        """
        Mask PII regions in the image.
        
        Args:
            image: Input image as numpy array
            pii_regions: List of PII regions with bounding boxes
            method: Masking method ('black_box', 'blur', 'pixelate', 'solid_color')
            padding: Extra padding around the bounding box
            
        Returns:
            Masked image as numpy array
        """
        masked_image = image.copy()
        
        if method not in self.mask_methods:
            method = 'black_box'  # Default to black box
        
        for region in pii_regions:
            bbox = region.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Add padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Apply masking
                masked_image = self.mask_methods[method](
                    masked_image, x1, y1, x2, y2
                )
        
        return masked_image
    
    def _black_box_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Apply black box masking."""
        image[y1:y2, x1:x2] = [0, 0, 0]  # Black color
        return image
    
    def _blur_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Apply blur masking."""
        # Extract the region
        region = image[y1:y2, x1:x2]
        
        # Apply heavy blur
        blurred_region = cv2.GaussianBlur(region, (99, 99), 30)
        
        # Replace the region
        image[y1:y2, x1:x2] = blurred_region
        return image
    
    def _pixelate_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Apply pixelation masking."""
        # Extract the region
        region = image[y1:y2, x1:x2]
        
        # Resize to very small size and then back to original size
        small = cv2.resize(region, (8, 8), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        
        # Replace the region
        image[y1:y2, x1:x2] = pixelated
        return image
    
    def _solid_color_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Apply solid color masking (red)."""
        image[y1:y2, x1:x2] = [255, 0, 0]  # Red color
        return image
    
    def add_redaction_indicators(self, image: np.ndarray, pii_regions: List[Dict]) -> np.ndarray:
        """
        Add visual indicators showing what was redacted.
        
        Args:
            image: Input image as numpy array
            pii_regions: List of PII regions
            
        Returns:
            Image with redaction indicators
        """
        # Convert to PIL for easier text drawing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for region in pii_regions:
            bbox = region.get('bbox')
            pii_type = region.get('type', 'PII')
            
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Draw a red border around the redacted area
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                
                # Add label above the redacted area
                label = f"[{pii_type.upper()}]"
                draw.text((x1, max(0, y1-20)), label, fill='red', font=font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def create_comparison_image(self, original: np.ndarray, masked: np.ndarray) -> np.ndarray:
        """
        Create a side-by-side comparison of original and masked images.
        
        Args:
            original: Original image
            masked: Masked image
            
        Returns:
            Side-by-side comparison image
        """
        # Ensure both images have the same dimensions
        h1, w1 = original.shape[:2]
        h2, w2 = masked.shape[:2]
        
        # Use the larger dimensions
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        # Resize images if necessary
        if h1 != max_h or w1 != max_w:
            original = cv2.resize(original, (max_w, max_h))
        if h2 != max_h or w2 != max_w:
            masked = cv2.resize(masked, (max_w, max_h))
        
        # Create side-by-side image
        comparison = np.hstack([original, masked])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Masked', (max_w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return comparison
    
    def image_to_base64(self, image: np.ndarray, format: str = 'JPEG') -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: Image as numpy array
            format: Image format ('JPEG', 'PNG')
            
        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """
        Convert base64 string to image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Image as numpy array
        """
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    
    def get_masking_statistics(self, pii_regions: List[Dict]) -> Dict:
        """
        Get statistics about the masking process.
        
        Args:
            pii_regions: List of PII regions that were masked
            
        Returns:
            Dictionary with masking statistics
        """
        stats = {
            'total_regions': len(pii_regions),
            'by_type': {},
            'total_characters': 0
        }
        
        for region in pii_regions:
            pii_type = region.get('type', 'unknown')
            text = region.get('text', '')
            
            # Count by type
            if pii_type not in stats['by_type']:
                stats['by_type'][pii_type] = 0
            stats['by_type'][pii_type] += 1
            
            # Count characters
            stats['total_characters'] += len(text)
        
        return stats 