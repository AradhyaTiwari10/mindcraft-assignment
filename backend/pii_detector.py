import re
from typing import List, Dict, Tuple

class PIIDetector:
    """Detects Personally Identifiable Information in text using regex patterns."""
    
    def __init__(self):
        # Initialize regex patterns for different PII types
        self.patterns = {
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',  # Aadhaar number format
            'phone': r'\b(?:\+91\s?)?[6-9]\d{9}\b',  # Indian phone numbers
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'dob': r'\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b',  # DD/MM/YYYY
            'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',  # PAN card format
            'pincode': r'\b[1-9][0-9]{5}\b',  # Indian pincode
            'driving_license': r'\b[A-Z]{2}\d{2}[A-Z]\d{4}\d{7}\b',  # Driving license format
            'issue_date': r'\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b',  # Issue dates
        }
        
        # Common Indian names (expanded list for better detection)
        self.indian_names = [
            'raj', 'kumar', 'singh', 'patel', 'sharma', 'verma', 'gupta', 'malhotra',
            'kapoor', 'reddy', 'khan', 'ahmed', 'ali', 'shah', 'mehta', 'jain',
            'agarwal', 'bhatt', 'chopra', 'dubey', 'goswami', 'iyer', 'joshi',
            'kulkarni', 'mahajan', 'nair', 'oberoi', 'pandey', 'rao', 'saxena',
            'tripathi', 'uday', 'varma', 'wadhwa', 'yadav', 'zaveri', 'tiwari',
            'shukla', 'mishra', 'pandit', 'choudhary', 'thakur', 'yadav', 'aradhya',
            'priya', 'neha', 'kavya', 'aditi', 'ananya', 'isha', 'riya', 'zara',
            'aarav', 'advait', 'dhruv', 'kabir', 'rishabh', 'vivaan', 'ayaan',
            'arjun', 'dev', 'kartik', 'om', 'rudra', 'shiv', 'ved', 'yash', 'Aradhya'
        ]
    
    def detect_pii_in_ocr_regions(self, ocr_regions: List[Dict]) -> List[Dict]:
        """
        Detect PII directly in OCR regions with bounding boxes.
        
        Args:
            ocr_regions: List of OCR regions with text and bounding boxes
            
        Returns:
            List of PII regions with bounding boxes
        """
        detected_pii = []
        
        # Sort OCR regions by y-coordinate (top to bottom)
        sorted_regions = sorted(ocr_regions, key=lambda x: x['center_y'])
        
        # Step 1: Detect structured PII (numbers, dates, etc.)
        for region in sorted_regions:
            text = region['text'].strip()
            if not text:
                continue
                
            # Detect Aadhaar numbers
            if re.search(self.patterns['aadhaar'], text):
                detected_pii.append({
                    'type': 'aadhaar',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.95
                })
            
            # Detect phone numbers
            if re.search(self.patterns['phone'], text):
                detected_pii.append({
                    'type': 'phone',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.90
                })
            
            # Detect email addresses
            if re.search(self.patterns['email'], text):
                detected_pii.append({
                    'type': 'email',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.95
                })
            
            # Detect dates of birth
            if re.search(self.patterns['dob'], text):
                detected_pii.append({
                    'type': 'dob',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.85
                })
            
            # Detect PAN numbers
            if re.search(self.patterns['pan'], text):
                detected_pii.append({
                    'type': 'pan',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.95
                })
            
            # Detect pincodes
            if re.search(self.patterns['pincode'], text):
                detected_pii.append({
                    'type': 'pincode',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.80
                })
            
            # Detect driving license numbers
            if re.search(self.patterns['driving_license'], text):
                detected_pii.append({
                    'type': 'driving_license',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.95
                })
        
        # Step 2: Detect names using positional logic
        name_regions = self._detect_names_positional(sorted_regions)
        detected_pii.extend(name_regions)
        
        # Step 3: Detect addresses
        address_regions = self._detect_addresses(sorted_regions)
        detected_pii.extend(address_regions)
        
        return detected_pii
    
    def _detect_names_positional(self, sorted_regions: List[Dict]) -> List[Dict]:
        """
        Detect names using positional logic (most reliable for Indian documents).
        
        Args:
            sorted_regions: OCR regions sorted by y-coordinate
            
        Returns:
            List of detected name regions
        """
        name_regions = []
        
        # Find DOB regions first
        dob_regions = []
        for i, region in enumerate(sorted_regions):
            text = region['text'].strip()
            if re.search(self.patterns['dob'], text):
                dob_regions.append((i, region))
        
        # For each DOB region, look for name above it
        for dob_idx, dob_region in dob_regions:
            # Look at regions above the DOB (within reasonable distance)
            dob_y = dob_region['center_y']
            
            # Find the closest region above DOB that looks like a name
            best_name_region = None
            best_distance = float('inf')
            
            for i in range(dob_idx - 1, max(0, dob_idx - 5), -1):  # Check up to 5 regions above
                region = sorted_regions[i]
                text = region['text'].strip()
                
                # Skip if too far above DOB
                if dob_y - region['center_y'] > 100:  # More than 100 pixels above
                    continue
                
                # Check if this region looks like a name
                if self._is_likely_name(text):
                    distance = dob_y - region['center_y']
                    if distance < best_distance:
                        best_distance = distance
                        best_name_region = region
            
            if best_name_region:
                name_regions.append({
                    'type': 'name',
                    'text': best_name_region['text'],
                    'bbox': best_name_region['bbox'],
                    'confidence': 0.95
                })
        
        # Enhanced name detection for driving licenses and other documents
        # Look for "Name:" labels and mask the following text
        for i, region in enumerate(sorted_regions):
            text = region['text'].strip().lower()
            
            # Check for name labels
            if any(label in text for label in ['name:', 'name', 'नाम:', 'नाम']):
                # Look for the name in the next region or same region after colon
                if ':' in region['text']:
                    # Name might be in the same region after colon
                    parts = region['text'].split(':', 1)
                    if len(parts) > 1:
                        name_part = parts[1].strip()
                        if self._is_likely_name(name_part):
                            name_regions.append({
                                'type': 'name',
                                'text': name_part,
                                'bbox': region['bbox'],
                                'confidence': 0.90
                            })
                else:
                    # Name might be in the next region
                    if i + 1 < len(sorted_regions):
                        next_region = sorted_regions[i + 1]
                        next_text = next_region['text'].strip()
                        if self._is_likely_name(next_text):
                            name_regions.append({
                                'type': 'name',
                                'text': next_text,
                                'bbox': next_region['bbox'],
                                'confidence': 0.90
                            })
        
        # Detect parent names (Son/Daughter/Wife of)
        for i, region in enumerate(sorted_regions):
            text = region['text'].strip().lower()
            
            # Check for parent name patterns
            parent_patterns = [
                'son of:', 'daughter of:', 'wife of:', 's/o:', 'd/o:', 'w/o:',
                'son of', 'daughter of', 'wife of', 's/o', 'd/o', 'w/o'
            ]
            
            if any(pattern in text for pattern in parent_patterns):
                # Look for the parent name in the next region or same region after colon
                if ':' in region['text']:
                    # Parent name might be in the same region after colon
                    parts = region['text'].split(':', 1)
                    if len(parts) > 1:
                        parent_name = parts[1].strip()
                        if self._is_likely_name(parent_name):
                            name_regions.append({
                                'type': 'parent_name',
                                'text': parent_name,
                                'bbox': region['bbox'],
                                'confidence': 0.90
                            })
                else:
                    # Parent name might be in the next region
                    if i + 1 < len(sorted_regions):
                        next_region = sorted_regions[i + 1]
                        next_text = next_region['text'].strip()
                        if self._is_likely_name(next_text):
                            name_regions.append({
                                'type': 'parent_name',
                                'text': next_text,
                                'bbox': next_region['bbox'],
                                'confidence': 0.90
                            })
        
        # NEW: Detect text after colons (:) - this is a very practical approach
        colon_regions = self._detect_text_after_colons(sorted_regions)
        name_regions.extend(colon_regions)
        
        # NEW: Detect split name regions (names that are split across multiple OCR regions)
        split_name_regions = self._detect_split_names(sorted_regions)
        name_regions.extend(split_name_regions)
        

        
        # If no names found via DOB, try alternative methods
        if not name_regions:
            # Look for regions that contain common name patterns
            for region in sorted_regions:
                text = region['text'].strip()
                if self._is_likely_name(text):
                    # Check if it's not already detected as another PII type
                    is_other_pii = any(
                        re.search(pattern, text) for pattern in self.patterns.values()
                    )
                    
                    if not is_other_pii:
                        name_regions.append({
                            'type': 'name',
                            'text': text,
                            'bbox': region['bbox'],
                            'confidence': 0.80
                        })
        
        return name_regions
    
    def _detect_text_after_colons(self, sorted_regions: List[Dict]) -> List[Dict]:
        """
        Detect and mask text that appears after colons (:) in forms and documents.
        This is a very practical approach for masking form field values.
        IMPORTANT: Only masks the text AFTER the colon, preserving the label.
        
        Args:
            sorted_regions: OCR regions sorted by y-coordinate
            
        Returns:
            List of detected regions to mask
        """
        colon_regions = []
        
        # Common form field labels that should have their values masked
        field_labels = [
            'name', 'नाम', 'full name', 'first name', 'last name',
            'father', 'mother', 'parent', 'guardian',
            'son of', 'daughter of', 'wife of', 's/o', 'd/o', 'w/o',
            'address', 'पता', 'permanent address', 'current address',
            'phone', 'mobile', 'contact', 'फोन',
            'email', 'e-mail', 'ईमेल',
            'dob', 'date of birth', 'birth date', 'जन्म तिथि',
            'aadhaar', 'आधार', 'aadhaar number', 'आधार संख्या',
            'pan', 'पैन', 'pan number',
            'license', 'licence', 'driving license', 'driving licence',
            'passport', 'passport number',
            'blood group', 'blood type',
            'signature', 'holder signature'
        ]
        
        for i, region in enumerate(sorted_regions):
            text = region['text'].strip()
            text_lower = text.lower()
            
            # Check if this region contains a colon and a field label
            if ':' in text:
                # Split by colon
                parts = text.split(':', 1)
                if len(parts) > 1:
                    label_part = parts[0].strip().lower()
                    value_part = parts[1].strip()
                    
                    # Check if the label part contains any of our field labels
                    if any(label in label_part for label in field_labels):
                        # This is a form field with a value after colon
                        if value_part and len(value_part) > 1:  # Make sure there's actually a value
                            # Calculate the bounding box for just the value part (after colon)
                            # This is approximate - we'll mask from the colon position to the end
                            bbox = region['bbox']
                            colon_position = text.find(':')
                            
                            # Estimate the x-coordinate where the colon appears
                            # This is a rough calculation based on text length
                            total_width = bbox[2] - bbox[0]
                            colon_ratio = colon_position / len(text)
                            colon_x = bbox[0] + int(total_width * colon_ratio)
                            
                            # Create a new bbox for just the value part (after colon)
                            value_bbox = [colon_x + 5, bbox[1], bbox[2], bbox[3]]  # +5 for some spacing
                            
                            colon_regions.append({
                                'type': 'form_field',
                                'text': value_part,
                                'bbox': value_bbox,
                                'confidence': 0.95,
                                'field_label': label_part
                            })
                            continue
            
            # Also check for cases where the label and value are in separate regions
            # Look for regions that contain field labels
            if any(label in text_lower for label in field_labels):
                # Check if the next region contains a value (no colon in current region)
                if i + 1 < len(sorted_regions) and ':' not in text:
                    next_region = sorted_regions[i + 1]
                    next_text = next_region['text'].strip()
                    
                    # If next region looks like a value (not a label)
                    if (next_text and len(next_text) > 1 and 
                        not any(label in next_text.lower() for label in field_labels) and
                        not re.search(r'^\d+$', next_text)):  # Not just a number
                        
                        # Check if they're on roughly the same line (within 20 pixels)
                        if abs(region['center_y'] - next_region['center_y']) < 20:
                            colon_regions.append({
                                'type': 'form_field',
                                'text': next_text,
                                'bbox': next_region['bbox'],
                                'confidence': 0.90,
                                'field_label': text_lower
                            })
        
        return colon_regions
    
    def _detect_split_names(self, sorted_regions: List[Dict]) -> List[Dict]:
        """
        Detect names that are split across multiple OCR regions on the same line.
        This handles cases like "ARADHYA" and "TIWARI" being in separate regions.
        
        Args:
            sorted_regions: OCR regions sorted by y-coordinate
            
        Returns:
            List of detected split name regions
        """
        split_name_regions = []
        
        # Group regions by y-coordinate (same line)
        line_groups = {}
        for region in sorted_regions:
            y_coord = region['center_y']
            # Group regions within 10 pixels of each other (same line)
            line_key = round(y_coord / 10) * 10
            if line_key not in line_groups:
                line_groups[line_key] = []
            line_groups[line_key].append(region)
        
        # Look for lines that contain "Name" label
        for line_key, regions in line_groups.items():
            # Sort regions in this line by x-coordinate (left to right)
            line_regions = sorted(regions, key=lambda x: x['center_x'])
            
            # Check if this line contains a "Name" label
            has_name_label = False
            name_label_idx = -1
            
            for i, region in enumerate(line_regions):
                text_lower = region['text'].strip().lower()
                if 'name' in text_lower:
                    has_name_label = True
                    name_label_idx = i
                    break
            
            if has_name_label and name_label_idx + 1 < len(line_regions):
                # Found a "Name" label, now look for name parts after it
                name_parts = []
                combined_bbox = None
                
                for i in range(name_label_idx + 1, len(line_regions)):
                    region = line_regions[i]
                    text = region['text'].strip()
                    
                    # Skip if it's clearly not a name part
                    if (len(text) < 2 or 
                        any(char.isdigit() for char in text) or
                        any(skip in text.lower() for skip in ['date', 'birth', 'address', 'phone', 'email'])):
                        continue
                    
                    # Check if it looks like a name part
                    if self._is_likely_name_part(text):
                        name_parts.append(text)
                        
                        # Update combined bounding box (only for name parts, not the label)
                        if combined_bbox is None:
                            combined_bbox = list(region['bbox'])
                        else:
                            # Expand bbox to include this region
                            combined_bbox[0] = min(combined_bbox[0], region['bbox'][0])
                            combined_bbox[1] = min(combined_bbox[1], region['bbox'][1])
                            combined_bbox[2] = max(combined_bbox[2], region['bbox'][2])
                            combined_bbox[3] = max(combined_bbox[3], region['bbox'][3])
                
                # If we found name parts, add them as a combined region
                if name_parts and combined_bbox:
                    full_name = ' '.join(name_parts)
                    split_name_regions.append({
                        'type': 'name',
                        'text': full_name,
                        'bbox': combined_bbox,
                        'confidence': 0.95,
                        'parts': name_parts
                    })
        
        # Also look for parent name patterns (Son/Daughter/Wife of)
        for line_key, regions in line_groups.items():
            line_regions = sorted(regions, key=lambda x: x['center_x'])
            
            # Check if this line contains parent name patterns
            parent_patterns = ['son of', 'daughter of', 'wife of', 's/o', 'd/o', 'w/o']
            has_parent_label = False
            parent_label_idx = -1
            
            for i, region in enumerate(line_regions):
                text_lower = region['text'].strip().lower()
                if any(pattern in text_lower for pattern in parent_patterns):
                    has_parent_label = True
                    parent_label_idx = i
                    break
            
            if has_parent_label and parent_label_idx + 1 < len(line_regions):
                # Found a parent label, now look for parent name parts after it
                parent_name_parts = []
                combined_bbox = None
                
                for i in range(parent_label_idx + 1, len(line_regions)):
                    region = line_regions[i]
                    text = region['text'].strip()
                    
                    # Skip if it's clearly not a name part
                    if (len(text) < 2 or 
                        any(char.isdigit() for char in text) or
                        any(skip in text.lower() for skip in ['date', 'birth', 'address', 'phone', 'email'])):
                        continue
                    
                    # Check if it looks like a name part
                    if self._is_likely_name_part(text):
                        parent_name_parts.append(text)
                        
                        # Update combined bounding box
                        if combined_bbox is None:
                            combined_bbox = list(region['bbox'])
                        else:
                            # Expand bbox to include this region
                            combined_bbox[0] = min(combined_bbox[0], region['bbox'][0])
                            combined_bbox[1] = min(combined_bbox[1], region['bbox'][1])
                            combined_bbox[2] = max(combined_bbox[2], region['bbox'][2])
                            combined_bbox[3] = max(combined_bbox[3], region['bbox'][3])
                
                # If we found parent name parts, add them as a combined region
                if parent_name_parts and combined_bbox:
                    full_parent_name = ' '.join(parent_name_parts)
                    split_name_regions.append({
                        'type': 'parent_name',
                        'text': full_parent_name,
                        'bbox': combined_bbox,
                        'confidence': 0.95,
                        'parts': parent_name_parts
                    })
        
        return split_name_regions
    

    
    def _is_likely_name(self, text: str) -> bool:
        """
        Check if text looks like a name.
        
        Args:
            text: Text to check
            
        Returns:
            True if text looks like a name
        """
        # Skip if too short or too long
        if len(text) < 3 or len(text) > 50:
            return False
        
        # Skip if contains numbers or special characters (except spaces and dots)
        if re.search(r'[0-9@#$%^&*()_+\-=\[\]{}|\\:";\'<>?,./]', text):
            return False
        
        # Skip if it's clearly not a name
        skip_keywords = [
            'government', 'india', 'aadhaar', 'number', 'dob', 'male', 'female',
            'address', 'phone', 'email', 'issue', 'validity', 'driving', 'licence',
            'card', 'date', 'year', 'month', 'day', 'issued', 'valid', 'until'
        ]
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in skip_keywords):
            return False
        
        # Check if it contains at least 2 words (first and last name)
        words = text.split()
        if len(words) < 2:
            return False
        
        # Check if words contain common name patterns
        name_score = 0
        for word in words:
            word_clean = word.lower().strip('.,!?')
            if word_clean in self.indian_names or len(word) > 2:
                name_score += 1
        
        # At least 2 words should look like names
        return name_score >= 2
    
    def _is_likely_name_part(self, text: str) -> bool:
        """
        Check if text looks like a name part (for split name detection).
        
        Args:
            text: Text to check
            
        Returns:
            True if text looks like a name part
        """
        # Skip if too short or too long
        if len(text) < 2 or len(text) > 20:
            return False
        
        # Skip if contains numbers or special characters (except spaces and dots)
        if re.search(r'[0-9@#$%^&*()_+\-=\[\]{}|\\:";\'<>?,./]', text):
            return False
        
        # Skip if it's clearly not a name part
        skip_keywords = [
            'government', 'india', 'aadhaar', 'number', 'dob', 'male', 'female',
            'address', 'phone', 'email', 'issue', 'validity', 'driving', 'licence',
            'card', 'date', 'year', 'month', 'day', 'issued', 'valid', 'until',
            'blood', 'group', 'signature', 'holder'
        ]
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in skip_keywords):
            return False
        
        # Check if it contains letters (at least 2)
        if sum(1 for c in text if c.isalpha()) < 2:
            return False
        
        # Check if words contain common name patterns
        words = text.split()
        name_score = 0
        for word in words:
            word_clean = word.lower().strip('.,!?')
            if word_clean in self.indian_names or len(word) > 2:
                name_score += 1
        
        # At least one word should look like a name
        return name_score >= 1
    
    def _detect_addresses(self, sorted_regions: List[Dict]) -> List[Dict]:
        """
        Detect address patterns in OCR regions.
        
        Args:
            sorted_regions: OCR regions sorted by y-coordinate
            
        Returns:
            List of detected address regions
        """
        address_regions = []
        
        # Common address keywords
        address_keywords = [
            'street', 'road', 'lane', 'avenue', 'colony', 'sector', 'block',
            'flat', 'apartment', 'house', 'building', 'floor', 'room',
            'village', 'town', 'city', 'district', 'state', 'country',
            'post', 'office', 'pincode', 'pin', 'code', 'address', 'पता',
            'nagar', 'mohalla', 'basti', 'chowk', 'market', 'area', 'zone',
            's/o', 'c/o', 'near', 'behind', 'opposite', 'next to'
        ]
        
        for region in sorted_regions:
            text = region['text'].strip()
            if len(text) < 10:  # Addresses are usually long
                continue
            
            text_lower = text.lower()
            
            # Check if it contains address keywords
            keyword_count = sum(1 for keyword in address_keywords if keyword in text_lower)
            
            # Check if it contains a pincode
            has_pincode = bool(re.search(self.patterns['pincode'], text))
            
            # If it has multiple address keywords or a pincode, it's likely an address
            if keyword_count >= 2 or has_pincode:
                address_regions.append({
                    'type': 'address',
                    'text': text,
                    'bbox': region['bbox'],
                    'confidence': 0.75 if keyword_count >= 2 else 0.80
                })
        
        return address_regions
    
    def detect_pii(self, text: str) -> List[Dict]:
        """
        Legacy method - kept for backward compatibility.
        Use detect_pii_in_ocr_regions instead for better results.
        """
        # This method is kept for compatibility but should not be used
        # as it doesn't provide bounding box information
        detected_pii = []
        
        # Basic PII detection without bounding boxes
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detected_pii.append({
                    'type': pii_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })
        
        return detected_pii
    
    def get_all_pii(self, text: str) -> List[Dict]:
        """
        Legacy method - kept for backward compatibility.
        Use detect_pii_in_ocr_regions instead for better results.
        """
        return self.detect_pii(text) 