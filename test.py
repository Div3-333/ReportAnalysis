import fitz  # PyMuPDF
import re
import spacy
from collections import Counter
from typing import List, Optional, Tuple, Set, Dict, Any
import os
import pickle
import json
from datetime import datetime, date
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process, fuzz
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rejection and threshold definitions
NAME_REJECTION = {
    "year", "special comments", "occupation", "leadership", "mathematics",
    "id", "activities", "behaviour", "report", "teacher", "class", "subject",
    "remarks", "date", "school", "name", "comments", "address", "parent", "surname", 
    "road", "farmer", "street", "strcct", "birth", "mother's", "avenue", 
    "anticipated", "jeweller", "mile", "hawskburn", "manufacturer"
}

# VIC Suburbs (your existing list)
VIC_SUBURBS = {
    'abbotsford', 'airport west', 'albany', 'albanvale', 'alberton', 'albert park',
    'albion', 'altona', 'altona east', 'altona meadows', 'altona north', 'anderson',
    'anglesea', 'armadale', 'ashburton', 'ashwood', 'aspendale', 'aspendale gardens',
    'attwood', 'auburn', 'avondale heights', 'bayswater', 'bayswater north', 'beaconsfield',
    'beaconsfield upper', 'beaumaris', 'bellfield', 'belgrave', 'belgrave heights', 'bentleigh',
    'bentleigh east', 'berwick', 'briar hill', 'brighton', 'brighton east', 'broadmeadows',
    'brookfield', 'brooklyn', 'brunswick', 'brunswick east', 'brunswick west', 'bulleen',
    'bundoora', 'burnley', 'burwood', 'burwood east', 'cairnlea', 'camberwell', 'carnegie',
    'carlton', 'carlton north', 'caroline springs', 'carrum', 'carrum downs', 'caulfield',
    'caulfield east', 'caulfield north', 'caulfield south', 'chelsea', 'chelsea heights',
    'cheltenham', 'chadstone', 'chapel street', 'clifton hill', 'clayton', 'clayton south',
    'clyde', 'clyde north', 'coburg', 'coburg north', 'cockatoo', 'coldstream', 'collingwood',
    'coolaroo', 'cranbourne', 'cranbourne east', 'cranbourne north', 'cranbourne south',
    'cranbourne west', 'dandenong', 'dandenong north', 'dandenong south', 'deer park', 'delahey',
    'diamond creek', 'diggers rest', 'doncaster', 'doncaster east', 'donvale', 'doreen', 'doveton',
    'elsternwick', 'eltham', 'eltham north', 'elwood', 'endeavour hills', 'epping', 'essendon',
    'essendon north', 'essendon west', 'exford', 'ferntree gully', 'fitzroy', 'fitzroy north',
    'flemington', 'footscray', 'gardenvale', 'garfield', 'glen huntly', 'glen iris', 'glenroy', 'glen waverley', 'greensborough', 'greenvale', 'hallam', 'hampton', 'hampton east',
    'hampton park', 'hawthorn', 'hawthorn east', 'heidelberg', 'heidelberg heights', 'heidelberg west',
    'highett', 'hillside', 'hoppers crossing', 'huntingdale', 'ivanhoe', 'ivanhoe east', 'jacana',
    'junction village', 'kalkallo', 'kallista', 'kangaroo ground', 'kealba', 'keilor', 'keilor east',
    'keilor north', 'keilor lodge', 'kew', 'kew east', 'kilsyth', 'kilsyth south', 'kings park',
    'kingsville', 'knoxfield', 'kooyong', 'lalor', 'langwarrin', 'laverton', 'laverton north',
    'lilydale', 'little river', 'lower plenty', 'lysterfield', 'macclesfield', 'macleod', 'maidstone',
    'malvern', 'malvern east', 'mentone', 'mentone east', 'melton', 'melton south', 'melton west',
    'mernda', 'middle park', 'mill park', 'mitcham', 'monbulk', 'mont albert north', 'montmorency',
    'moonee ponds', 'mooroolbark', 'mordialloc', 'mornington', 'mount eliza', 'mount martha',
    'mount waverley', 'mulgrave', 'narre warren', 'narre warren north', 'noble park', 'noble park north',
    'north melbourne', 'northcote', 'north warrandyte', 'notting hill', 'nunawading', 'oak park',
    'oakleigh', 'oakleigh east', 'oakleigh south', 'olinda', 'ormond', 'pakenham', 'park orchards',
    'pascoe vale', 'pascoe vale south', 'point cook', 'port melbourne', 'prahran', 'preston', 'richmond',
    'ringwood', 'ringwood east', 'ringwood north', 'rosanna', 'rowville', 'rye', 'sandringham',
    'seaford', 'seaholme', 'south yarra', 'southbank', 'south melbourne', 'south morang', 'spotswood',
    'springvale', 'st albans', 'st andrews', 'st kilda', 'st kilda east', 'st kilda west', 'sunbury',
    'sunshine', 'sunshine north', 'sunshine west', 'surrey hills', 'sydenham', 'tarneit', 'templeton',
    'thomastown', 'thornbury', 'toorak', 'tooradin', 'truganina', 'tullamarine', 'tyabb', 'vermont',
    'vermont south', 'viewbank', 'wantirna', 'wantirna south', 'warrandyte', 'warrandyte south',
    'warranwood', 'waterways', 'werribee', 'werribee south', 'west melbourne', 'west footscray',
    'wheelers hill', 'whittlesea', 'williamstown', 'williamstown north', 'windsor', 'wollert', 'wonga park',
    'yarra junction', 'yarra glen', 'yarraville'
}


# Comprehensive number pattern - captures all numeric sequences
ALL_NUMBERS_PATTERN = re.compile(r'\d+')

# Date context patterns for better detection
DATE_CONTEXT_PATTERNS = [
    r'(?i)date\s+of\s+birth\s*[:\-]?\s*(\d+[/.\-]\d+[/.\-]?\d*)',
    r'(?i)born\s*[:\-]?\s*(\d+[/.\-]\d+[/.\-]?\d*)',
    r'(?i)birth\s*[:\-]?\s*(\d+[/.\-]\d+[/.\-]?\d*)',
    r'(?i)d\.?o\.?b\.?\s*[:\-]?\s*(\d+[/.\-]\d+[/.\-]?\d*)',
    r'(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})',
    r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})',
]

# Bounding boxes
NAME_BBOXES = {
    0: [(290, 510, 716, 510), (290, 530, 716, 530)],
    1: [(15, 500, 170, 500), (15, 525, 170, 525)],
}
SUBURB_BBOXES = {
    0: [(115, 485, 275, 485), (115, 515, 275, 515)],
}
YEAR_BBOXES = {
    0: [(100, 172, 160, 172), (100, 381, 160, 381)],
}

class YearClassifier:
    """ML-based year classifier that learns to identify years from context"""
    
    def __init__(self, model_path: str = "year_classifier.pkl"):
        self.model_path = model_path
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.is_trained = False
        self.training_data = []
        self.confidence_threshold = 0.6
        
        # Load existing model if available
        self.load_model()
        
    def extract_features(self, number: str, context: str) -> Dict[str, Any]:
        """Extract features for year classification"""
        features = {}
        
        # Number-based features
        try:
            num_val = int(number)
            features['number_value'] = num_val
            features['number_length'] = len(number)
            features['is_two_digit'] = len(number) == 2
            features['is_four_digit'] = len(number) == 4
            features['in_birth_range'] = 1900 <= num_val <= 2000  # valid birth year range
            features['in_school_range'] = 1900 <= num_val <= 2000  # school document range
            
            # Convert 2-digit years to 4-digit (heuristic) - all map to 1900-1999
            if len(number) == 2:
                # All 2-digit years map to 19XX (1900-1999)
                full_year = 1900 + num_val
                features['converted_year'] = full_year
                features['converted_in_range'] = 1900 <= full_year <= 2000
            else:
                features['converted_year'] = num_val
                features['converted_in_range'] = features['in_birth_range']
                
        except ValueError:
            features.update({
                'number_value': 0, 'number_length': 0, 'is_two_digit': 0,
                'is_four_digit': 0, 'in_birth_range': 0, 'in_school_range': 0,
                'converted_year': 0, 'converted_in_range': 0
            })
        
        # Context-based features
        context_lower = context.lower()
        features['has_birth_context'] = int(any(word in context_lower for word in 
                                               ['birth', 'born', 'dob', 'date of birth']))
        features['has_date_context'] = int(any(word in context_lower for word in 
                                             ['date', 'year', 'age']))
        features['has_slash'] = int('/' in context)
        features['has_dot'] = int('.' in context)
        features['has_dash'] = int('-' in context)
        features['context_length'] = len(context)
        
        # Pattern matching features
        features['in_date_pattern'] = 0
        for pattern in DATE_CONTEXT_PATTERNS:
            if re.search(pattern, context):
                features['in_date_pattern'] = 1
                break
                
        return features
    
    def create_training_data(self):
        """Create initial training data with known patterns"""
        training_examples = [
            # Positive examples (years) - all within 1900-2000 range
            ("36", "DATE OF BIRTH 8/11/36", 1),   # -> 1936
            ("1936", "DATE OF BIRTH 8/11/1936", 1),
            ("31", "DATE OF BIRTH 6.12.31", 1),   # -> 1931
            ("1931", "DATE OF BIRTH 6.12.1931", 1),
            ("45", "Born 15/3/45", 1),            # -> 1945
            ("1945", "Born 15/3/1945", 1),
            ("23", "YEAR: 23", 1),                # -> 1923
            ("1923", "YEAR: 1923", 1),
            ("42", "DOB: 12-8-42", 1),            # -> 1942
            ("1942", "DOB: 12-8-1942", 1),
            ("55", "Birth date: 22.5.55", 1),     # -> 1955
            ("1955", "Birth date: 22.5.1955", 1),
            ("68", "Born: 1/1/68", 1),            # -> 1968
            ("1968", "Born: 1/1/1968", 1),
            
            # Negative examples (not years)
            ("12", "12 Smith Street", 0),
            ("8", "Grade 8", 0),
            ("100", "Score: 100", 0),
            ("25", "Age: 25 years old", 0),
            ("3", "Room 3", 0),
            ("15", "Page 15", 0),
            ("7", "Section 7", 0),
            ("123", "Student ID: 123", 0),
            ("456", "Phone: 456-789", 0),
            ("2", "Class 2A", 0),
        ]
        
        self.training_data.extend(training_examples)
    
    def train_initial_model(self):
        """Train the initial model with bootstrap data"""
        if not self.training_data:
            self.create_training_data()
            
        if len(self.training_data) < 5:
            logger.warning("Not enough training data for initial model")
            return
            
        X_features = []
        contexts = []
        y = []
        
        for number, context, label in self.training_data:
            features = self.extract_features(number, context)
            X_features.append(list(features.values()))
            contexts.append(context)
            y.append(label)
        
        # Fit vectorizer on contexts
        X_text = self.vectorizer.fit_transform(contexts)
        X_numeric = np.array(X_features)
        
        # Combine features
        X_combined = np.hstack([X_numeric, X_text.toarray()])
        
        # Train classifier
        self.classifier.fit(X_combined, y)
        self.is_trained = True
        self.save_model()
        
        logger.info(f"Initial model trained with {len(self.training_data)} examples")
    
    def predict_year_probability(self, number: str, context: str) -> float:
        """Predict probability that a number is a year"""
        if not self.is_trained:
            self.train_initial_model()
            
        if not self.is_trained:
            # Fallback heuristic if model training fails
            return self._heuristic_year_probability(number, context)
            
        try:
            features = self.extract_features(number, context)
            X_numeric = np.array([list(features.values())])
            X_text = self.vectorizer.transform([context])
            X_combined = np.hstack([X_numeric, X_text.toarray()])
            
            probabilities = self.classifier.predict_proba(X_combined)[0]
            return probabilities[1] if len(probabilities) > 1 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in prediction: {e}")
            return self._heuristic_year_probability(number, context)
    
    def _heuristic_year_probability(self, number: str, context: str) -> float:
        """Fallback heuristic scoring"""
        score = 0.0
        
        try:
            num_val = int(number)
            context_lower = context.lower()
            
            # Length-based scoring
            if len(number) == 2:
                score += 0.3
            elif len(number) == 4:
                score += 0.4
            else:
                return 0.1  # Very unlikely
                
            # Range-based scoring
            if len(number) == 2:
                # All 2-digit years become 19XX
                converted_year = 1900 + num_val
                if 1900 <= converted_year <= 2000:
                    score += 0.4
            elif 1900 <= num_val <= 2000:
                score += 0.4
                
            # Context-based scoring
            if any(word in context_lower for word in ['birth', 'born', 'dob']):
                score += 0.4
            if any(word in context_lower for word in ['date', 'year']):
                score += 0.2
            if any(char in context for char in ['/', '.', '-']):
                score += 0.2
                
            return min(score, 1.0)
            
        except ValueError:
            return 0.0
    
    def add_training_example(self, number: str, context: str, is_year: bool):
        """Add a new training example and retrain if needed"""
        self.training_data.append((number, context, int(is_year)))
        
        # Retrain periodically
        if len(self.training_data) % 10 == 0:
            self.train_initial_model()
    
    def save_model(self):
        """Save the trained model and data"""
        try:
            model_data = {
                'classifier': self.classifier,
                'vectorizer': self.vectorizer,
                'training_data': self.training_data,
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def load_model(self):
        """Load existing model and data"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.vectorizer = model_data['vectorizer']
                self.training_data = model_data['training_data']
                self.is_trained = model_data['is_trained']
                logger.info(f"Loaded model with {len(self.training_data)} training examples")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")


class StudentInfoExtractor:
    def __init__(self, vic_suburbs: Set[str]):
        self.vic_suburbs = set(s.lower() for s in vic_suburbs)
        self.nlp = spacy.load("en_core_web_sm")
        self.year_classifier = YearClassifier()
        
        # Initialize ML model
        if not self.year_classifier.is_trained:
            self.year_classifier.train_initial_model()

    def extract_text_from_bbox(self, doc: fitz.Document, page_num: int, bbox: Tuple[int, int, int, int]) -> str:
        page = doc[page_num]
        rect = fitz.Rect(bbox)
        words = page.get_text("words")
        selected = [w[4] for w in words if rect.contains(fitz.Point(w[0], w[1]))]
        return " ".join(selected).strip()

    def extract_bbox_candidates(self, doc: fitz.Document, bboxes: dict) -> List[str]:
        c = []
        for page_num, boxes in bboxes.items():
            if page_num < len(doc):
                for bbox in boxes:
                    t = self.extract_text_from_bbox(doc, page_num, bbox)
                    if t:
                        c.append(t)
        return c

    def clean_name_candidate(self, name: str) -> Optional[str]:
        orig = name.strip()
        if not orig:
            return None
        low = orig.lower()
        # Reject if full string is in rejection list
        if low in NAME_REJECTION:
            return None
        # Reject if any word in name is in rejection
        for w in low.split():
            if w in NAME_REJECTION:
                return None
        # Reject if contains suburb name (to avoid mixing suburb with name)
        for sub in self.vic_suburbs:
            if sub in low:
                return None
        # Filter non-letters out, allow apostrophes or hyphens maybe?
        cleaned = re.sub(r"[^A-Za-z\-\s']", " ", orig)
        cleaned_parts = [p.strip() for p in cleaned.split() if p.strip()]
        if not cleaned_parts:
            return None
        # Now parts length rule: allow one part if it's decent length
        if len(cleaned_parts) == 1:
            part = cleaned_parts[0]
            if len(part) < 3:
                return None
            # single word but accept
            return part.capitalize()
        # If multiple parts, ensure none is one-letter
        if any(len(p) <= 1 for p in cleaned_parts):
            return None
        return " ".join(p.capitalize() for p in cleaned_parts)

    def extract_name_from_bbox(self, doc: fitz.Document) -> Optional[str]:
        candidates = self.extract_bbox_candidates(doc, NAME_BBOXES)
        cleaned = []
        for c in candidates:
            # strip label text
            c2 = re.sub(r'(christian name|first name|surname|:)', '', c, flags=re.I).strip()
            cn = self.clean_name_candidate(c2)
            if cn:
                cleaned.append(cn)
        if cleaned:
            # perhaps pick the candidate which has most letters or appears first
            # return the one with max length
            return max(cleaned, key=lambda x: len(x))
        return None

    def extract_names_from_ner(self, text: str) -> List[str]:
        docn = self.nlp(text)
        names = []
        for ent in docn.ents:
            if ent.label_ == "PERSON":
                cn = self.clean_name_candidate(ent.text)
                if cn:
                    names.append(cn)
        return names

    def extract_name_after_label(self, text: str) -> Optional[str]:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            low = line.lower()
            if "christian name" in low or "first name" in low:
                # look after
                if i + 1 < len(lines):
                    candidate = lines[i+1].strip()
                    cn = self.clean_name_candidate(candidate)
                    if cn:
                        return cn
            # also consider if label and name on same line
            m = re.search(r'(?:christian name|first name)\s*[:\-]\s*(.+)$', line, flags=re.I)
            if m:
                cn = self.clean_name_candidate(m.group(1).strip())
                if cn:
                    return cn
        return None

    def get_most_frequent_name(self, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        cnt = Counter(candidates)
        most, freq = cnt.most_common(1)[0]
        # accept even if frequency = 1 now (relaxing)
        return most

    def extract_suburb_from_bbox(self, doc: fitz.Document) -> Optional[str]:
        candidates = self.extract_bbox_candidates(doc, SUBURB_BBOXES)
        return self._suburb_best_match(candidates)

    def extract_suburb_from_text(self, text: str) -> Optional[str]:
        # try to find exact substrings first
        for sub in self.vic_suburbs:
            if sub in text.lower():
                return sub.title()
        # else try fuzzy over multi-word substrings
        # get all sequences of up to 3 words
        words = [w for w in re.findall(r"[A-Za-z']+", text)]
        # build candidate substrings
        substrings = []
        for idx in range(len(words)):
            for length in (2,3):
                if idx + length <= len(words):
                    substr = " ".join(words[idx: idx+length])
                    substrings.append(substr)
        best = None
        best_score = 0.0
        for cand in substrings:
            if len(cand) < 3:
                continue
            match, score, _ = process.extractOne(cand.lower(), self.vic_suburbs, scorer=fuzz.WRatio)
            if score is not None and score > best_score:
                best_score = score
                best = match
        if best and best_score > 80:  # threshold
            return best.title()
        return None

    def _suburb_best_match(self, candidates: List[str]) -> Optional[str]:
        best = None
        best_score = 0.0
        for cand in candidates:
            if not cand:
                continue
            match, score, _ = process.extractOne(cand.lower(), self.vic_suburbs, scorer=fuzz.WRatio)
            if score is not None and score > best_score:
                best_score = score
                best = match
        if best and best_score > 90:
            return best.title()
        return None

    def extract_years_with_ml(self, text: str, bbox_candidates: List[str] = None) -> Set[int]:
        """Extract years using ML classification"""
        all_numbers = []
        contexts = []
        
        # Extract all numbers from full text with context
        lines = text.splitlines()
        for i, line in enumerate(lines):
            numbers = ALL_NUMBERS_PATTERN.findall(line)
            for num in numbers:
                # Create context window (current line + surrounding lines)
                context_lines = []
                for j in range(max(0, i-1), min(len(lines), i+2)):
                    context_lines.append(lines[j])
                context = " ".join(context_lines)
                
                all_numbers.append(num)
                contexts.append(context)
        
        # Also check bbox candidates
        if bbox_candidates:
            for candidate_text in bbox_candidates:
                numbers = ALL_NUMBERS_PATTERN.findall(candidate_text)
                for num in numbers:
                    all_numbers.append(num)
                    contexts.append(candidate_text)
        
        # Use ML to classify each number
        year_candidates = []
        for num, context in zip(all_numbers, contexts):
            probability = self.year_classifier.predict_year_probability(num, context)
            
            if probability > self.year_classifier.confidence_threshold:
                # Convert to 4-digit year
                try:
                    num_val = int(num)
                    if len(num) == 2:
                        # All 2-digit years become 19XX (1900-1999)
                        year = 1900 + num_val
                    else:
                        year = num_val
                    
                    # Strict validation: only 1900-2000 are valid
                    if 1900 <= year <= 2000:
                        year_candidates.append((year, probability, context))
                        
                except ValueError:
                    continue
        
        # Sort by probability and take the most confident predictions
        year_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Extract years, but limit to most confident ones
        final_years = set()
        for year, prob, context in year_candidates[:5]:  # Top 5 most confident
            final_years.add(year)
            
            # Self-improvement: add high-confidence predictions to training
            if prob > 0.8:
                self.year_classifier.add_training_example(str(year), context, True)
        
        return final_years

    def extract_years_from_bbox(self, doc: fitz.Document) -> List[str]:
        """Extract text from year bounding boxes"""
        return self.extract_bbox_candidates(doc, YEAR_BBOXES)

    def extract_all(self, pdf_path: str) -> Tuple[Optional[str], str, Set[int]]:
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)

        # Names (unchanged)
        name_after_label = self.extract_name_after_label(full_text)
        name_bbox = self.extract_name_from_bbox(doc)
        names_ner = self.extract_names_from_ner(full_text)
        name_ner = self.get_most_frequent_name(names_ner)

        final_name = name_after_label or name_bbox or name_ner or "UNKNOWN"

        # Suburbs (unchanged)
        suburb_bbox = self.extract_suburb_from_bbox(doc)
        suburb_text = self.extract_suburb_from_text(full_text)
        final_suburb = suburb_bbox or suburb_text or "Melbourne"

        # Years (new ML-based approach)
        bbox_candidates = self.extract_years_from_bbox(doc)
        final_years = self.extract_years_with_ml(full_text, bbox_candidates)

        doc.close()
        return final_name, final_suburb.title(), final_years


if __name__ == "__main__":
    folder = os.path.abspath(os.path.dirname(__file__))
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]

    extractor = StudentInfoExtractor(VIC_SUBURBS)

    print("Enhanced Student Info Extractor with ML Year Detection")
    print("=" * 60)
    
    for pdf in pdfs:
        path = os.path.join(folder, pdf)
        try:
            name, suburb, years = extractor.extract_all(path)
            print(f"File: {pdf}")
            print(f"  Name: {name}")
            print(f"  Suburb: {suburb}")
            print(f"  Years Detected: {sorted(years) if years else 'None'}")
            
            # Show some debug info about the ML predictions
            if years:
                print(f"  Total Years Found: {len(years)}")
                current_year = 2000  # Use fixed upper bound since we only accept years 1900-2000
                birth_years = [y for y in years if 1900 <= y <= 2000]
                if birth_years:
                    print(f"  Likely Birth Years: {sorted(birth_years)}")
            print()
            
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
            print()
    
    # Show model training statistics
    print("Model Statistics:")
    print(f"  Training Examples: {len(extractor.year_classifier.training_data)}")
    print(f"  Model Trained: {extractor.year_classifier.is_trained}")
    print(f"  Confidence Threshold: {extractor.year_classifier.confidence_threshold}")
    
    # Interactive improvement mode
    print("\n" + "=" * 60)
    print("Interactive Improvement Mode")
    print("Review the results above. You can provide feedback to improve the model.")
    print("Enter 'quit' to exit, or provide feedback in format:")
    print("filename.pdf: year YYYY is/isn't a birth year")
    print("Example: 'document1.pdf: year 1936 is a birth year'")
    print("Example: 'document1.pdf: year 123 is not a birth year'")
    
    while True:
        feedback = input("\nFeedback (or 'quit'): ").strip()
        if feedback.lower() in ['quit', 'exit', '']:
            break
            
        try:
            # Parse feedback
            if ':' not in feedback:
                print("Invalid format. Use: filename.pdf: year YYYY is/isn't a birth year")
                continue
                
            file_part, year_part = feedback.split(':', 1)
            file_part = file_part.strip()
            year_part = year_part.strip().lower()
            
            # Extract year and label
            import re
            year_match = re.search(r'year (\d+)', year_part)
            if not year_match:
                print("Could not find year in feedback. Use format: year YYYY is/isn't a birth year")
                continue
                
            year_str = year_match.group(1)
            is_year = 'is a' in year_part or 'is birth' in year_part
            
            # Find the context from the processed file
            pdf_path = os.path.join(folder, file_part)
            if not os.path.exists(pdf_path):
                print(f"File {file_part} not found")
                continue
                
            # Re-extract text to find context
            doc = fitz.open(pdf_path)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()
            
            # Find context containing this year
            lines = full_text.splitlines()
            context = ""
            for i, line in enumerate(lines):
                if year_str in line:
                    context_lines = []
                    for j in range(max(0, i-1), min(len(lines), i+2)):
                        context_lines.append(lines[j])
                    context = " ".join(context_lines)
                    break
            
            if not context:
                context = f"Manual feedback for year {year_str}"
            
            # Add training example
            extractor.year_classifier.add_training_example(year_str, context, is_year)
            print(f"Added training example: {year_str} -> {'Year' if is_year else 'Not Year'}")
            print("Model will be retrained with new examples automatically.")
            
        except Exception as e:
            print(f"Error processing feedback: {e}")
            print("Please use format: filename.pdf: year YYYY is/isn't a birth year")
    
    print("\nThank you! The model has been improved with your feedback.")
    print("The enhanced model will be saved and used for future extractions.")