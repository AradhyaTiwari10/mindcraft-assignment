import re
import spacy
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import json

@dataclass
class PIIEntity:
    """Advanced PII entity with confidence scoring and context."""
    text: str
    type: str
    start: int
    end: int
    confidence: float
    context: str
    bbox: Optional[List[int]] = None
    metadata: Optional[Dict] = None

class AdvancedPIIDetector:
    """Advanced PII detection with ML models and sophisticated pattern matching."""
    
    def __init__(self):
        """Initialize advanced PII detector with multiple detection methods."""
        self.patterns = {
            # Indian Documents
            'aadhaar': {
                'pattern': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
                'confidence': 0.95,
                'context_keywords': ['aadhaar', 'uid', 'आधार', 'संख्या']
            },
            'pan': {
                'pattern': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
                'confidence': 0.95,
                'context_keywords': ['pan', 'permanent', 'account', 'number']
            },
            'phone': {
                'pattern': r'\b(?:\+91\s?)?[6-9]\d{9}\b',
                'confidence': 0.90,
                'context_keywords': ['phone', 'mobile', 'contact', 'फोन']
            },
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'confidence': 0.95,
                'context_keywords': ['email', 'e-mail', 'ईमेल']
            },
            'dob': {
                'pattern': r'\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b',
                'confidence': 0.85,
                'context_keywords': ['dob', 'birth', 'date', 'जन्म', 'तिथि']
            },
            'pincode': {
                'pattern': r'\b[1-9][0-9]{5}\b',
                'confidence': 0.80,
                'context_keywords': ['pincode', 'pin', 'postal', 'code']
            },
            'passport': {
                'pattern': r'\b[A-Z]{1}[0-9]{7}\b',
                'confidence': 0.90,
                'context_keywords': ['passport', 'number', 'पासपोर्ट']
            },
            'driving_license': {
                'pattern': r'\b[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}\b',
                'confidence': 0.85,
                'context_keywords': ['driving', 'license', 'dl', 'ड्राइविंग']
            }
        }
        
        # Advanced name patterns for Indian names
        self.name_patterns = {
            'indian_names': [
                'raj', 'kumar', 'singh', 'patel', 'sharma', 'verma', 'gupta', 'malhotra',
                'kapoor', 'reddy', 'khan', 'ahmed', 'ali', 'shah', 'mehta', 'jain',
                'agarwal', 'bhatt', 'chopra', 'dubey', 'goswami', 'iyer', 'joshi',
                'kulkarni', 'mahajan', 'nair', 'oberoi', 'pandey', 'rao', 'saxena',
                'tripathi', 'uday', 'varma', 'wadhwa', 'yadav', 'zaveri', 'tiwari',
                'shukla', 'mishra', 'pandit', 'choudhary', 'thakur', 'yadav'
            ],
            'titles': ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'shri', 'smt']
        }
        
        # Document type detection
        self.document_keywords = {
            'aadhaar': ['भारत सरकार', 'government of india', 'आधार', 'aadhaar', 'uidai'],
            'pan': ['permanent account number', 'pan', 'income tax'],
            'passport': ['passport', 'पासपोर्ट', 'republic of india'],
            'driving_license': ['driving license', 'transport', 'rto'],
            'bank_statement': ['bank', 'account', 'balance', 'transaction'],
            'credit_card': ['credit card', 'card number', 'expiry', 'cvv']
        }
        
        # Try to load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
            logging.info("spaCy model loaded successfully")
        except:
            self.use_spacy = False
            logging.warning("spaCy model not available, using fallback NER")
    
    def detect_document_type(self, text: str) -> str:
        """Detect the type of document based on keywords."""
        text_lower = text.lower()
        scores = {}
        
        for doc_type, keywords in self.document_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'unknown'
    
    def extract_names_with_context(self, text: str) -> List[PIIEntity]:
        """Extract names using context and patterns."""
        entities = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for name patterns with context
            for title in self.name_patterns['titles']:
                if title in line_lower:
                    # Extract name after title
                    title_pos = line_lower.find(title)
                    name_start = title_pos + len(title)
                    name_part = line[name_start:].strip()
                    
                    if name_part and len(name_part) > 2:
                        # Check if it contains Indian name patterns
                        words = name_part.split()
                        for word in words:
                            if word.lower() in self.name_patterns['indian_names']:
                                entities.append(PIIEntity(
                                    text=name_part,
                                    type='name',
                                    start=line.find(name_part),
                                    end=line.find(name_part) + len(name_part),
                                    confidence=0.75,
                                    context=f"Found after title '{title}'"
                                ))
                                break
        
        return entities
    
    def detect_with_context(self, text: str) -> List[PIIEntity]:
        """Detect PII with context awareness."""
        entities = []
        text_lower = text.lower()
        
        for pii_type, config in self.patterns.items():
            pattern = config['pattern']
            base_confidence = config['confidence']
            context_keywords = config['context_keywords']
            
            matches = re.finditer(pattern, text)
            for match in matches:
                # Calculate context score
                context_score = 0
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos].lower()
                
                for keyword in context_keywords:
                    if keyword in context:
                        context_score += 0.1
                
                # Adjust confidence based on context
                adjusted_confidence = min(0.99, base_confidence + context_score)
                
                entities.append(PIIEntity(
                    text=match.group(),
                    type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=adjusted_confidence,
                    context=context
                ))
        
        return entities
    
    def extract_addresses(self, text: str) -> List[PIIEntity]:
        """Extract addresses using pattern matching and context."""
        entities = []
        lines = text.split('\n')
        
        address_keywords = [
            'address', 'पता', 'street', 'road', 'lane', 'avenue', 'colony',
            'sector', 'block', 'flat', 'apartment', 'house', 'building',
            'village', 'town', 'city', 'district', 'state', 'country',
            'post', 'office', 'pincode', 'pin', 'code', 's/o', 'c/o'
        ]
        
        for line_num, line in enumerate(lines):
            line_lower = line.lower()
            keyword_count = sum(1 for keyword in address_keywords if keyword in line_lower)
            
            # If line contains multiple address keywords and is long enough
            if keyword_count >= 2 and len(line.strip()) > 15:
                # Check if it contains pincode (strong indicator of address)
                pincode_match = re.search(r'\b[1-9][0-9]{5}\b', line)
                if pincode_match:
                    entities.append(PIIEntity(
                        text=line.strip(),
                        type='address',
                        start=text.find(line),
                        end=text.find(line) + len(line),
                        confidence=0.85,
                        context=f"Contains {keyword_count} address keywords and pincode"
                    ))
        
        return entities
    
    def detect_with_spacy(self, text: str) -> List[PIIEntity]:
        """Use spaCy for Named Entity Recognition."""
        if not self.use_spacy:
            return []
        
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                entities.append(PIIEntity(
                    text=ent.text,
                    type=ent.label_.lower(),
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.70,
                    context=f"Detected by spaCy as {ent.label_}"
                ))
        
        return entities
    
    def get_all_pii(self, text: str) -> List[PIIEntity]:
        """Get all detected PII using multiple methods."""
        all_entities = []
        
        # Detect document type
        doc_type = self.detect_document_type(text)
        logging.info(f"Detected document type: {doc_type}")
        
        # Pattern-based detection
        pattern_entities = self.detect_with_context(text)
        all_entities.extend(pattern_entities)
        
        # Name detection with context
        name_entities = self.extract_names_with_context(text)
        all_entities.extend(name_entities)
        
        # Address detection
        address_entities = self.extract_addresses(text)
        all_entities.extend(address_entities)
        
        # spaCy NER (if available)
        spacy_entities = self.detect_with_spacy(text)
        all_entities.extend(spacy_entities)
        
        # Remove duplicates and sort by confidence
        unique_entities = self._remove_duplicates(all_entities)
        unique_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        logging.info(f"Detected {len(unique_entities)} PII entities")
        return unique_entities
    
    def _remove_duplicates(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove duplicate entities based on overlap."""
        if not entities:
            return []
        
        # Sort by confidence (highest first)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        unique_entities = []
        for entity in entities:
            is_duplicate = False
            for existing in unique_entities:
                # Check for overlap
                if (entity.start < existing.end and entity.end > existing.start and
                    entity.type == existing.type):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def get_statistics(self, entities: List[PIIEntity]) -> Dict:
        """Get detailed statistics about detected PII."""
        stats = {
            'total_entities': len(entities),
            'by_type': {},
            'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
            'total_characters': 0,
            'document_type': 'unknown'
        }
        
        for entity in entities:
            # Count by type
            if entity.type not in stats['by_type']:
                stats['by_type'][entity.type] = 0
            stats['by_type'][entity.type] += 1
            
            # Count by confidence
            if entity.confidence >= 0.8:
                stats['by_confidence']['high'] += 1
            elif entity.confidence >= 0.6:
                stats['by_confidence']['medium'] += 1
            else:
                stats['by_confidence']['low'] += 1
            
            # Count characters
            stats['total_characters'] += len(entity.text)
        
        return stats 