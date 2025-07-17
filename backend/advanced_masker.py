import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import base64
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import logging
from dataclasses import dataclass
import qrcode
from pyzbar import pyzbar

@dataclass
class MaskingResult:
    """Result of masking operation with metadata."""
    masked_image: np.ndarray
    statistics: Dict
    detected_faces: List[Tuple[int, int, int, int]]
    detected_qr_codes: List[Tuple[int, int, int, int]]
    processing_time: float

class AdvancedImageMasker:
    """Advanced image masking with face detection, QR code detection, and multiple techniques."""
    
    def __init__(self):
        """Initialize advanced image masker with multiple capabilities."""
        self.mask_methods = {
            'black_box': self._black_box_mask,
            'blur': self._blur_mask,
            'pixelate': self._pixelate_mask,
            'solid_color': self._solid_color_mask,
            'gaussian_noise': self._gaussian_noise_mask,
            'mosaic': self._mosaic_mask,
            'inpaint': self._inpaint_mask
        }
        
        # Initialize face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.use_face_detection = True
            logging.info("Face detection initialized")
        except:
            self.use_face_detection = False
            logging.warning("Face detection not available")
        
        # Color schemes for different PII types
        self.pii_colors = {
            'aadhaar': (0, 0, 255),      # Red
            'pan': (255, 0, 0),          # Blue
            'phone': (0, 255, 0),        # Green
            'email': (255, 255, 0),      # Cyan
            'dob': (255, 0, 255),        # Magenta
            'pincode': (0, 255, 255),    # Yellow
            'name': (128, 0, 128),       # Purple
            'address': (255, 165, 0),    # Orange
            'face': (255, 0, 0),         # Red for faces
            'qr_code': (0, 0, 255)       # Blue for QR codes
        }
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the image."""
        if not self.use_face_detection:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        detected_faces = []
        for (x, y, w, h) in faces:
            detected_faces.append((x, y, x+w, y+h))
        
        logging.info(f"Detected {len(detected_faces)} faces")
        return detected_faces
    
    def detect_qr_codes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect QR codes and barcodes in the image."""
        try:
            # Convert to PIL Image for pyzbar
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            barcodes = pyzbar.decode(pil_image)
            
            qr_regions = []
            for barcode in barcodes:
                # Get bounding box
                x, y, w, h = barcode.rect
                qr_regions.append((x, y, x+w, y+h))
            
            logging.info(f"Detected {len(qr_regions)} QR codes/barcodes")
            return qr_regions
        except Exception as e:
            logging.warning(f"QR code detection failed: {e}")
            return []
    
    def mask_pii_regions(self, image: np.ndarray, pii_regions: List[Dict], 
                        method: str = 'black_box', padding: int = 5,
                        mask_faces: bool = True, mask_qr_codes: bool = True) -> MaskingResult:
        """
        Advanced masking with face and QR code detection.
        
        Args:
            image: Input image
            pii_regions: List of PII regions
            method: Masking method
            padding: Extra padding
            mask_faces: Whether to mask detected faces
            mask_qr_codes: Whether to mask detected QR codes
            
        Returns:
            MaskingResult with masked image and metadata
        """
        import time
        start_time = time.time()
        
        masked_image = image.copy()
        statistics = {
            'total_regions': 0,
            'by_type': {},
            'faces_detected': 0,
            'qr_codes_detected': 0,
            'processing_time': 0
        }
        
        # Detect faces and QR codes
        detected_faces = self.detect_faces(image) if mask_faces else []
        detected_qr_codes = self.detect_qr_codes(image) if mask_qr_codes else []
        
        # Mask PII regions
        for region in pii_regions:
            bbox = region.get('bbox')
            pii_type = region.get('type', 'unknown')
            
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Add padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Apply masking
                masked_image = self.mask_methods[method](
                    masked_image, x1, y1, x2, y2, pii_type
                )
                
                # Update statistics
                statistics['total_regions'] += 1
                if pii_type not in statistics['by_type']:
                    statistics['by_type'][pii_type] = 0
                statistics['by_type'][pii_type] += 1
        
        # Mask faces
        for face_bbox in detected_faces:
            x1, y1, x2, y2 = face_bbox
            masked_image = self._blur_mask(masked_image, x1, y1, x2, y2, 'face')
            statistics['faces_detected'] += 1
            statistics['total_regions'] += 1
        
        # Mask QR codes
        for qr_bbox in detected_qr_codes:
            x1, y1, x2, y2 = qr_bbox
            masked_image = self._black_box_mask(masked_image, x1, y1, x2, y2, 'qr_code')
            statistics['qr_codes_detected'] += 1
            statistics['total_regions'] += 1
        
        processing_time = time.time() - start_time
        statistics['processing_time'] = processing_time
        
        return MaskingResult(
            masked_image=masked_image,
            statistics=statistics,
            detected_faces=detected_faces,
            detected_qr_codes=detected_qr_codes,
            processing_time=processing_time
        )
    
    def _black_box_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pii_type: str = 'unknown') -> np.ndarray:
        """Apply black box masking with color coding."""
        color = self.pii_colors.get(pii_type, (0, 0, 0))
        image[y1:y2, x1:x2] = color
        return image
    
    def _blur_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pii_type: str = 'unknown') -> np.ndarray:
        """Apply blur masking."""
        region = image[y1:y2, x1:x2]
        blurred_region = cv2.GaussianBlur(region, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_region
        return image
    
    def _pixelate_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pii_type: str = 'unknown') -> np.ndarray:
        """Apply pixelation masking."""
        region = image[y1:y2, x1:x2]
        small = cv2.resize(region, (8, 8), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = pixelated
        return image
    
    def _solid_color_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pii_type: str = 'unknown') -> np.ndarray:
        """Apply solid color masking."""
        color = self.pii_colors.get(pii_type, (255, 0, 0))
        image[y1:y2, x1:x2] = color
        return image
    
    def _gaussian_noise_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pii_type: str = 'unknown') -> np.ndarray:
        """Apply Gaussian noise masking."""
        region = image[y1:y2, x1:x2]
        noise = np.random.normal(0, 50, region.shape).astype(np.uint8)
        noisy_region = cv2.add(region, noise)
        image[y1:y2, x1:x2] = noisy_region
        return image
    
    def _mosaic_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pii_type: str = 'unknown') -> np.ndarray:
        """Apply mosaic masking."""
        region = image[y1:y2, x1:x2]
        h, w = region.shape[:2]
        block_size = max(1, min(h, w) // 10)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = region[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    mean_color = np.mean(block, axis=(0, 1))
                    region[i:i+block_size, j:j+block_size] = mean_color
        
        image[y1:y2, x1:x2] = region
        return image
    
    def _inpaint_mask(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pii_type: str = 'unknown') -> np.ndarray:
        """Apply inpainting masking."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return inpainted
    
    def add_advanced_indicators(self, image: np.ndarray, pii_regions: List[Dict], 
                              detected_faces: List[Tuple], detected_qr_codes: List[Tuple]) -> np.ndarray:
        """Add advanced visual indicators with confidence scores and metadata."""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw PII region indicators
        for region in pii_regions:
            bbox = region.get('bbox')
            pii_type = region.get('type', 'PII')
            confidence = region.get('confidence', 0.0)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Draw colored border
                color = self.pii_colors.get(pii_type, (255, 0, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Add label with confidence
                label = f"[{pii_type.upper()}] {confidence:.2f}"
                draw.text((x1, max(0, y1-25)), label, fill=color, font=font)
        
        # Draw face indicators
        for face_bbox in detected_faces:
            x1, y1, x2, y2 = face_bbox
            color = self.pii_colors['face']
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1-25)), "[FACE]", fill=color, font=font)
        
        # Draw QR code indicators
        for qr_bbox in detected_qr_codes:
            x1, y1, x2, y2 = qr_bbox
            color = self.pii_colors['qr_code']
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1-25)), "[QR CODE]", fill=color, font=font)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def create_advanced_comparison(self, original: np.ndarray, masked: np.ndarray, 
                                 statistics: Dict) -> np.ndarray:
        """Create an advanced comparison with statistics overlay."""
        # Ensure same dimensions
        h1, w1 = original.shape[:2]
        h2, w2 = masked.shape[:2]
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        if h1 != max_h or w1 != max_w:
            original = cv2.resize(original, (max_w, max_h))
        if h2 != max_h or w2 != max_w:
            masked = cv2.resize(masked, (max_w, max_h))
        
        # Create side-by-side image
        comparison = np.hstack([original, masked])
        
        # Add labels and statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Masked', (max_w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Add statistics overlay
        y_offset = 70
        cv2.putText(comparison, f"Total Regions: {statistics['total_regions']}", 
                   (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(comparison, f"Faces: {statistics['faces_detected']}", 
                   (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(comparison, f"QR Codes: {statistics['qr_codes_detected']}", 
                   (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(comparison, f"Time: {statistics['processing_time']:.2f}s", 
                   (10, y_offset), font, 0.7, (255, 255, 255), 2)
        
        return comparison
    
    def image_to_base64(self, image: np.ndarray, format: str = 'JPEG') -> str:
        """Convert image to base64 string."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str 