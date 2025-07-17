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
        }
        
        # Common Indian names (partial list for demonstration)
        self.indian_names = [
            'raj', 'kumar', 'singh', 'patel', 'sharma', 'verma', 'gupta', 'malhotra',
            'kapoor', 'reddy', 'khan', 'ahmed', 'ali', 'kumar', 'shah', 'mehta',
            'jain', 'agarwal', 'bhatt', 'chopra', 'dubey', 'goswami', 'iyer',
            'joshi', 'kulkarni', 'mahajan', 'nair', 'oberoi', 'pandey', 'rao',
            'saxena', 'tripathi', 'uday', 'varma', 'wadhwa', 'yadav', 'zaveri'
        ]
    
    def detect_pii(self, text: str) -> List[Dict]:
        """
        Detect PII in the given text and return bounding box information.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of dictionaries containing PII information
        """
        detected_pii = []
        
        # Convert text to lowercase for name detection
        text_lower = text.lower()
        
        # Detect Aadhaar numbers
        aadhaar_matches = re.finditer(self.patterns['aadhaar'], text)
        for match in aadhaar_matches:
            detected_pii.append({
                'type': 'aadhaar',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # Detect phone numbers
        phone_matches = re.finditer(self.patterns['phone'], text)
        for match in phone_matches:
            detected_pii.append({
                'type': 'phone',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.90
            })
        
        # Detect email addresses
        email_matches = re.finditer(self.patterns['email'], text)
        for match in email_matches:
            detected_pii.append({
                'type': 'email',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # Detect dates of birth
        dob_matches = re.finditer(self.patterns['dob'], text)
        for match in dob_matches:
            detected_pii.append({
                'type': 'dob',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.85
            })
        
        # Detect PAN numbers
        pan_matches = re.finditer(self.patterns['pan'], text)
        for match in pan_matches:
            detected_pii.append({
                'type': 'pan',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # Detect pincodes
        pincode_matches = re.finditer(self.patterns['pincode'], text)
        for match in pincode_matches:
            detected_pii.append({
                'type': 'pincode',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.80
            })
        
        # Detect names (simple heuristic based on common Indian names)
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.indian_names and len(word) > 2:
                # Find the position in original text
                start_pos = text.find(word)
                if start_pos != -1:
                    detected_pii.append({
                        'type': 'name',
                        'text': word,
                        'start': start_pos,
                        'end': start_pos + len(word),
                        'confidence': 0.70
                    })
        
        return detected_pii
    
    def detect_address_patterns(self, text: str) -> List[Dict]:
        """
        Detect address patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of address patterns found
        """
        address_patterns = []
        
        # Common address keywords
        address_keywords = [
            'street', 'road', 'lane', 'avenue', 'colony', 'sector', 'block',
            'flat', 'apartment', 'house', 'building', 'floor', 'room',
            'village', 'town', 'city', 'district', 'state', 'country',
            'post', 'office', 'pincode', 'pin', 'code'
        ]
        
        # Look for address patterns
        lines = text.split('\n')
        for line_num, line in enumerate(lines):
            line_lower = line.lower()
            keyword_count = sum(1 for keyword in address_keywords if keyword in line_lower)
            
            # If line contains multiple address keywords, it's likely an address
            if keyword_count >= 2 and len(line.strip()) > 10:
                start_pos = text.find(line)
                if start_pos != -1:
                    address_patterns.append({
                        'type': 'address',
                        'text': line.strip(),
                        'start': start_pos,
                        'end': start_pos + len(line),
                        'confidence': 0.75
                    })
        
        return address_patterns
    
    def get_all_pii(self, text: str) -> List[Dict]:
        """
        Get all detected PII including addresses.
        
        Args:
            text: Text to analyze
            
        Returns:
            Combined list of all PII detected
        """
        pii_list = self.detect_pii(text)
        address_list = self.detect_address_patterns(text)
        
        # Combine and sort by position
        all_pii = pii_list + address_list
        all_pii.sort(key=lambda x: x['start'])
        
        return all_pii 