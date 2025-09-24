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
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process, fuzz
import logging

# NLTK imports for name bank validation
try:
    import nltk
    from nltk.corpus import names
    # Download required NLTK data if not present
    try:
        nltk.data.find('corpora/names')
    except LookupError:
        print("Downloading NLTK names corpus...")
        nltk.download('names', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initial rejection and threshold definitions
NAME_REJECTION = {
    "year", "special comments", "occupation", "leadership", "mathematics",
    "id", "activities", "behaviour", "report", "teacher", "class", "subject",
    "remarks", "date", "school", "name", "comments", "address", "parent", "surname", 
    "road", "farmer", "street", "strcct", "birth", "mother's", "avenue", 
    "anticipated", "jeweller", "mile", "hawskburn", "manufacturer", "myer",
    "melbourne", "grammar", "barjna", "graxnmar", 'anticipated occupation', 'anticipated occupat'
}

# Common non-name patterns to reject
NON_NAME_PATTERNS = [
    r'.*road.*', r'.*street.*', r'.*avenue.*', r'.*lane.*', r'.*drive.*',
    r'.*school.*', r'.*grammar.*', r'.*college.*', r'.*university.*',
    r'.*myer.*', r'.*store.*', r'.*shop.*', r'.*company.*', r'.*ltd.*',
    r'.*melbourne.*', r'.*sydney.*', r'.*brisbane.*', r'.*perth.*',
    r'.*primary.*', r'.*secondary.*', r'.*high.*', r'.*junior.*', r'.*senior.*',
    r'.*church.*', r'.*hospital.*', r'.*clinic.*', r'.*centre.*', r'.*center.*'
]

# VIC Suburbs (existing list)
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


class NameBankValidator:
    """Name bank validator using NLTK and other name databases"""
    
    def __init__(self):
        self.nltk_first_names = set()
        self.common_surnames = set()
        
        # Load NLTK names if available
        if NLTK_AVAILABLE:
            try:
                male_names = set(name.lower() for name in names.words('male.txt'))
                female_names = set(name.lower() for name in names.words('female.txt'))
                self.nltk_first_names = male_names.union(female_names)
                logger.info(f"Loaded {len(self.nltk_first_names)} names from NLTK corpus")
            except Exception as e:
                logger.warning(f"Could not load NLTK names: {e}")
        
        # Backup name database (common names)
        self.backup_first_names = {
            'james', 'mary', 'john', 'patricia', 'robert', 'jennifer', 'michael', 'linda',
            'william', 'elizabeth', 'david', 'barbara', 'richard', 'susan', 'joseph', 'jessica',
            'thomas', 'sarah', 'christopher', 'karen', 'charles', 'helen', 'daniel', 'nancy',
            'matthew', 'betty', 'anthony', 'dorothy', 'mark', 'lisa', 'donald', 'sandra',
            'steven', 'donna', 'paul', 'carol', 'andrew', 'ruth', 'joshua', 'sharon',
            'kenneth', 'michelle', 'kevin', 'laura', 'brian', 'sarah', 'george', 'kimberly',
            'edward', 'deborah', 'ronald', 'rebecca', 'timothy', 'amy', 'jason', 'angela',
            'jeffrey', 'brenda', 'ryan', 'emma', 'jacob', 'olivia', 'gary', 'cynthia',
            'nicholas', 'marie', 'eric', 'janet', 'jonathan', 'catherine', 'stephen', 'frances',
            'larry', 'christine', 'justin', 'debra', 'scott', 'rachel', 'brandon', 'carolyn',
            'benjamin', 'janet', 'samuel', 'virginia', 'gregory', 'maria', 'alexander', 'heather',
            'patrick', 'diane', 'frank', 'julie', 'raymond', 'joyce', 'jack', 'victoria',
            'dennis', 'kelly', 'jerry', 'christina', 'tyler', 'joan', 'aaron', 'evelyn',
            'jose', 'judith', 'henry', 'megan', 'adam', 'cheryl', 'douglas', 'andrea',
            'nathan', 'hannah', 'peter', 'jacqueline', 'zachary', 'martha', 'kyle', 'gloria', 'azahir'
        }
        
        # Common surnames
        self.common_surnames = {
            'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis',
            'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson',
            'thomas', 'taylor', 'moore', 'jackson', 'martin', 'lee', 'perez', 'thompson',
            'white', 'harris', 'sanchez', 'clark', 'ramirez', 'lewis', 'robinson', 'walker',
            'young', 'allen', 'king', 'wright', 'scott', 'torres', 'nguyen', 'hill',
            'flores', 'green', 'adams', 'nelson', 'baker', 'hall', 'rivera', 'campbell',
            'mitchell', 'carter', 'roberts', 'gomez', 'phillips', 'evans', 'turner', 'diaz',
            'parker', 'cruz', 'edwards', 'collins', 'reyes', 'stewart', 'morris', 'morales',
            'murphy', 'cook', 'rogers', 'gutierrez', 'ortiz', 'morgan', 'cooper', 'peterson',
            'bailey', 'reed', 'kelly', 'howard', 'ramos', 'kim', 'cox', 'ward', 'richardson',
            'watson', 'brooks', 'chavez', 'wood', 'james', 'bennett', 'gray', 'mendoza',
            'ruiz', 'hughes', 'price', 'alvarez', 'castillo', 'sanders', 'patel', 'myers',
            'long', 'ross', 'foster', 'jimenez', 'powell', 'jenkins', 'perry', 'russell',
            'sullivan', 'bell', 'coleman', 'butler', 'henderson', 'barnes', 'gonzales',
            'fisher', 'vasquez', 'simmons', 'romero', 'jordan', 'patterson', 'alexander',
            'hamilton', 'graham', 'reynolds', 'griffin', 'wallace', 'moreno', 'west', 'bustamam'
        }
    
    def is_valid_first_name(self, name: str) -> Tuple[bool, str]:
        """Check if a name is a valid first name"""
        name_lower = name.lower().strip()
        
        # Check NLTK corpus first
        if self.nltk_first_names and name_lower in self.nltk_first_names:
            return True, "NLTK_corpus"
        
        # Check backup first names
        if name_lower in self.backup_first_names:
            return True, "backup_database"
        
        # Check for common name variations
        # Handle names like "Mary-Jane", "De Silva", etc.
        if '-' in name_lower:
            parts = name_lower.split('-')
            if all(part in self.nltk_first_names or part in self.backup_first_names for part in parts):
                return True, "hyphenated_name"
        
        # Handle apostrophes like "O'Connor"
        if "'" in name_lower:
            base_name = name_lower.replace("'", "")
            if base_name in self.nltk_first_names or base_name in self.backup_first_names:
                return True, "apostrophe_name"
        
        return False, "not_found"
    
    def is_valid_surname(self, name: str) -> Tuple[bool, str]:
        """Check if a name is a valid surname"""
        name_lower = name.lower().strip()
        
        if name_lower in self.common_surnames:
            return True, "common_surname"
        
        # Handle compound surnames
        if '-' in name_lower:
            parts = name_lower.split('-')
            if all(part in self.common_surnames for part in parts):
                return True, "hyphenated_surname"
        
        # Handle prefixes like "De", "Van", "Mc", "O'"
        prefixes = ['de', 'van', 'mc', 'o', 'la', 'le', 'del', 'della']
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                remaining = name_lower[len(prefix):].lstrip("'")
                if remaining in self.common_surnames:
                    return True, "prefixed_surname"
        
        return False, "not_found"
    
    def validate_full_name(self, full_name: str) -> Tuple[bool, str, List[str]]:
        """Validate a full name (first + last, or just first)"""
        parts = full_name.strip().split()
        validations = []
        
        if len(parts) == 1:
            # Single name - could be first or last
            is_first, first_reason = self.is_valid_first_name(parts[0])
            is_last, last_reason = self.is_valid_surname(parts[0])
            
            if is_first or is_last:
                reason = f"single_name_valid_{first_reason if is_first else last_reason}"
                validations.append(f"{parts[0]}={reason}")
                return True, reason, validations
        
        elif len(parts) == 2:
            # First + Last name
            first_valid, first_reason = self.is_valid_first_name(parts[0])
            last_valid, last_reason = self.is_valid_surname(parts[1])
            
            validations.append(f"{parts[0]}={first_reason}")
            validations.append(f"{parts[1]}={last_reason}")
            
            # Accept if either part is valid (not both need to be valid)
            if first_valid or last_valid:
                reason = f"partial_match_first={first_valid}_last={last_valid}"
                return True, reason, validations
        
        elif len(parts) >= 3:
            # First + Middle + Last or multiple names
            first_valid, first_reason = self.is_valid_first_name(parts[0])
            last_valid, last_reason = self.is_valid_surname(parts[-1])
            
            validations.append(f"{parts[0]}={first_reason}")
            validations.append(f"{parts[-1]}={last_reason}")
            
            # Check middle names
            middle_valid = False
            for middle in parts[1:-1]:
                mid_first, mid_first_reason = self.is_valid_first_name(middle)
                mid_last, mid_last_reason = self.is_valid_surname(middle)
                validations.append(f"{middle}={mid_first_reason if mid_first else mid_last_reason}")
                if mid_first or mid_last:
                    middle_valid = True
            
            # Accept if first name or last name or any middle name is valid
            if first_valid or last_valid or middle_valid:
                reason = f"multiple_parts_first={first_valid}_last={last_valid}_middle={middle_valid}"
                return True, reason, validations
        
        return False, "no_valid_parts", validations


class NameClassifier:
    """ML-based name classifier to distinguish real names from non-names"""
    
    def __init__(self, model_path: str = "name_classifier.pkl"):
        self.model_path = model_path
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.is_trained = False
        self.training_data = []
        self.confidence_threshold = 0.7
        
        # Initialize name bank validator
        self.name_bank = NameBankValidator()
        
        # Load existing model if available
        self.load_model()
        
    def extract_name_features(self, name: str) -> Dict[str, Any]:
        """Extract features for name classification"""
        features = {}
        name_lower = name.lower()
        
        # Length-based features
        features['length'] = len(name)
        features['word_count'] = len(name.split())
        features['avg_word_length'] = float(np.mean([len(w) for w in name.split()])) if name.split() else 0.0
        
        # Character-based features
        features['has_numbers'] = int(any(c.isdigit() for c in name))
        features['has_special_chars'] = int(any(c in '.,;:!@#$%^&*()' for c in name))
        features['has_apostrophe'] = int("'" in name)
        features['has_hyphen'] = int('-' in name)
        features['all_caps'] = int(name.isupper())
        features['title_case'] = int(name.istitle())
        
        # Pattern-based features
        features['contains_road_street'] = int(any(word in name_lower for word in 
                                                  ['road', 'street', 'avenue', 'lane', 'drive']))
        features['contains_school_words'] = int(any(word in name_lower for word in 
                                                   ['school', 'grammar', 'college', 'university', 'primary', 'secondary']))
        features['contains_business_words'] = int(any(word in name_lower for word in 
                                                     ['myer', 'store', 'shop', 'company', 'ltd', 'inc']))
        features['contains_place_words'] = int(any(word in name_lower for word in 
                                                  ['melbourne', 'sydney', 'brisbane', 'perth', 'adelaide']))
        features['contains_institution'] = int(any(word in name_lower for word in 
                                                  ['church', 'hospital', 'clinic', 'centre', 'center']))
        
        # Check against known non-name patterns
        features['matches_non_name_pattern'] = 0
        for pattern in NON_NAME_PATTERNS:
            if re.match(pattern, name_lower):
                features['matches_non_name_pattern'] = 1
                break
        
        # Check against rejection list
        features['in_rejection_list'] = int(name_lower in NAME_REJECTION)
        
        # Name-like features
        features['starts_with_capital'] = int(name[0].isupper() if name else 0)
        features['vowel_ratio'] = (len([c for c in name_lower if c in 'aeiou']) / len(name)) if name else 0.0
        features['consonant_clusters'] = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', name_lower))
        
        # Name bank features
        is_valid_name, validation_reason, validation_details = self.name_bank.validate_full_name(name)
        features['name_bank_valid'] = int(is_valid_name)
        features['name_bank_confidence'] = 1.0 if is_valid_name else 0.0
        
        return features
    
    def create_initial_training_data(self):
        """Create initial training data with known real names and non-names"""
        training_examples = [
            # Positive examples (real names)
            ("John Smith", 1), ("Mary Johnson", 1), ("David Brown", 1), ("Sarah Wilson", 1),
            ("Michael Davis", 1), ("Emma Thompson", 1), ("James Miller", 1), ("Anna Garcia", 1),
            ("Robert Martinez", 1), ("Lisa Anderson", 1), ("William Taylor", 1), ("Jennifer Moore", 1),
            ("Christopher Lee", 1), ("Michelle White", 1), ("Daniel Harris", 1), ("Jessica Clark", 1),
            ("Matthew Lewis", 1), ("Amanda Walker", 1), ("Anthony Hall", 1), ("Stephanie Allen", 1),
            ("O'Connor", 1), ("Mary-Jane", 1), ("De Silva", 1), ("Van Der Berg", 1),
            # Negative examples (non-names)
            ("Barjna Road", 0), ("Melbourne Grammar", 0), ("Myer", 0), ("Smith Street", 0),
            ("Collins Avenue", 0), ("Primary School", 0), ("Secondary College", 0), ("University", 0),
            ("Grammar School", 0), ("High School", 0), ("St Kilda Road", 0), ("Chapel Street", 0),
            ("Brighton Road", 0), ("Chadstone Shopping Centre", 0), ("Royal Melbourne Hospital", 0),
            ("Melbourne Cricket Ground", 0), ("Flinders Street Station", 0), ("Department Store", 0),
            ("Shopping Mall", 0), ("Business Centre", 0), ("Medical Clinic", 0), ("Dental Surgery", 0),
            ("Law Firm", 0), ("Accounting Office", 0), ("Real Estate", 0), ("Insurance Company", 0),
            ("Bank Branch", 0), ("Post Office", 0), ("Police Station", 0), ("Fire Station", 0),
            ("Train Station", 0), ("Bus Stop", 0), ("Car Park", 0), ("Sports Club", 0), ("Tennis Club", 0),
            ("Golf Course", 0), ("Swimming Pool", 0), ("Gymnasium", 0), ("Library", 0), ("Museum", 0),
            ("Art Gallery", 0), ("Theatre", 0), ("Cinema", 0), ("Restaurant", 0), ("Cafe", 0),
            ("Hotel", 0), ("Motel", 0),
        ]
        self.training_data.extend(training_examples)
    
    def train_initial_model(self):
        """Train the initial model with bootstrap data"""
        if not self.training_data:
            self.create_initial_training_data()
        if len(self.training_data) < 10:
            logger.warning("Not enough training data for initial name classifier model")
            return
            
        X_features = []
        X_text = []
        y = []
        
        for name, label in self.training_data:
            features = self.extract_name_features(name)
            X_features.append(list(features.values()))
            X_text.append(name)
            y.append(label)
        
        # Fit vectorizer on text
        X_text_vec = self.vectorizer.fit_transform(X_text)
        X_numeric = np.array(X_features)
        
        # Combine features
        X_combined = np.hstack([X_numeric, X_text_vec.toarray()])
        
        # Train classifier
        self.classifier.fit(X_combined, y)
        self.is_trained = True
        self.save_model()
        logger.info(f"Name classifier trained with {len(self.training_data)} examples")
    
    def predict_is_real_name(self, name: str) -> Tuple[bool, float]:
        """Predict if a string is a real name with name bank validation first"""
        
        # STEP 1: Name bank validation (highest priority)
        is_valid_name, validation_reason, validation_details = self.name_bank.validate_full_name(name)
        
        if is_valid_name:
            logger.info(f"✅ NAME BANK APPROVED: '{name}' - {validation_reason}")
            logger.info(f"   Validation details: {validation_details}")
            return True, 0.95  # High confidence for name bank matches
        
        # STEP 2: Check obvious rejection cases
        name_lower = name.lower()
        
        # Immediate rejections
        if name_lower in NAME_REJECTION:
            logger.info(f"❌ REJECTION LIST: '{name}' found in rejection list")
            return False, 0.1
        
        for pattern in NON_NAME_PATTERNS:
            if re.match(pattern, name_lower):
                logger.info(f"❌ PATTERN REJECTION: '{name}' matches non-name pattern")
                return False, 0.1
        
        # STEP 3: ML-based validation (if name bank didn't approve)
        if not self.is_trained:
            self.train_initial_model()
        if not self.is_trained:
            return self._heuristic_name_check(name)
        
        try:
            features = self.extract_name_features(name)
            X_numeric = np.array([list(features.values())])
            X_text = self.vectorizer.transform([name])
            X_combined = np.hstack([X_numeric, X_text.toarray()])
            probabilities = self.classifier.predict_proba(X_combined)[0]
            name_probability = probabilities[1] if len(probabilities) > 1 else 0.0
            
            # Adjust confidence based on name bank validation
            # If name bank said no, lower the ML confidence
            adjusted_confidence = name_probability * 0.7  # Reduce confidence when name bank rejects
            
            is_name = adjusted_confidence > self.confidence_threshold
            
            if is_name:
                logger.info(f"✅ ML APPROVED: '{name}' - confidence: {adjusted_confidence:.2f}")
            else:
                logger.info(f"❌ ML REJECTED: '{name}' - confidence: {adjusted_confidence:.2f}")
                logger.info(f"   Name bank validation: {validation_reason}")
            
            return is_name, float(adjusted_confidence)
            
        except Exception as e:
            logger.warning(f"Error in ML name prediction: {e}")
            return self._heuristic_name_check(name)
    
    def _heuristic_name_check(self, name: str) -> Tuple[bool, float]:
        """Fallback heuristic for name validation"""
        name_lower = name.lower()
        score = 0.5  # Start neutral
        
        # Check name bank first in heuristic too
        is_valid_name, validation_reason, _ = self.name_bank.validate_full_name(name)
        if is_valid_name:
            logger.info(f"✅ HEURISTIC + NAME BANK: '{name}' approved by name bank")
            return True, 0.9
        
        # Negative indicators
        if any(word in name_lower for word in ['road', 'street', 'avenue', 'school', 'grammar']):
            score -= 0.4
        if any(re.match(pattern, name_lower) for pattern in NON_NAME_PATTERNS):
            score -= 0.3
        if name_lower in NAME_REJECTION:
            score -= 0.5
        if any(c.isdigit() for c in name):
            score -= 0.2
        if len(name.split()) > 3:
            score -= 0.2
            
        # Positive indicators
        if name.istitle():
            score += 0.2
        if 2 <= len(name.split()) <= 3:
            score += 0.2
        if all(c.isalpha() or c in " '-" for c in name):
            score += 0.2
            
        is_name = score > 0.5
        
        if is_name:
            logger.info(f"✅ HEURISTIC APPROVED: '{name}' - score: {score:.2f}")
        else:
            logger.info(f"❌ HEURISTIC REJECTED: '{name}' - score: {score:.2f}")
        
        return is_name, float(max(0.0, min(1.0, score)))
    
    def add_training_example(self, name: str, is_real_name: bool):
        """Add a new training example and retrain if needed"""
        self.training_data.append((name, int(is_real_name)))
        # Add to rejection list if it's not a real name
        if not is_real_name:
            NAME_REJECTION.add(name.lower())
            logger.info(f"Added '{name}' to NAME_REJECTION list")
        # Retrain periodically
        if len(self.training_data) % 15 == 0:
            self.train_initial_model()
    
    def save_model(self):
        """Save the trained model and data"""
        try:
            model_data = {
                'classifier': self.classifier,
                'vectorizer': self.vectorizer,
                'training_data': self.training_data,
                'is_trained': self.is_trained,
                'name_rejection': NAME_REJECTION
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            logger.warning(f"Could not save name classifier model: {e}")
    
    def load_model(self):
        """Load existing model and data"""
        global NAME_REJECTION
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.vectorizer = model_data['vectorizer']
                self.training_data = model_data.get('training_data', [])
                self.is_trained = model_data.get('is_trained', False)
                if 'name_rejection' in model_data:
                    NAME_REJECTION.update(model_data['name_rejection'])
                logger.info(f"Loaded name classifier with {len(self.training_data)} training examples")
        except Exception as e:
            logger.warning(f"Could not load name classifier model: {e}")


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
        features: Dict[str, Any] = {}
        # Number-based features
        try:
            num_val = int(number)
            features['number_value'] = num_val
            features['number_length'] = len(number)
            features['is_two_digit'] = int(len(number) == 2)
            features['is_four_digit'] = int(len(number) == 4)
            features['in_birth_range'] = int(1900 <= num_val <= 2000)
            features['in_school_range'] = int(1900 <= num_val <= 2000)
            # Convert 2-digit years to 4-digit (heuristic) - map to 1900-1999
            if len(number) == 2:
                full_year = 1900 + num_val
                features['converted_year'] = full_year
                features['converted_in_range'] = int(1900 <= full_year <= 2000)
            else:
                features['converted_year'] = num_val
                features['converted_in_range'] = int(1900 <= num_val <= 2000)
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
            ("36", "DATE OF BIRTH 8/11/36", 1),
            ("1936", "DATE OF BIRTH 8/11/1936", 1),
            ("31", "DATE OF BIRTH 6.12.31", 1),
            ("1931", "DATE OF BIRTH 6.12.1931", 1),
            ("45", "Born 15/3/45", 1),
            ("1945", "Born 15/3/1945", 1),
            ("23", "YEAR: 23", 1),
            ("1923", "YEAR: 1923", 1),
            ("42", "DOB: 12-8-42", 1),
            ("1942", "DOB: 12-8-1942", 1),
            ("55", "Birth date: 22.5.55", 1),
            ("1955", "Birth date: 22.5.1955", 1),
            ("68", "Born: 1/1/68", 1),
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
        X_combined = np.hstack([X_numeric, X_text.toarray()])
        
        self.classifier.fit(X_combined, y)
        self.is_trained = True
        self.save_model()
        logger.info(f"Initial year model trained with {len(self.training_data)} examples")
    
    def predict_year_probability(self, number: str, context: str) -> float:
        """Predict probability that a number is a year"""
        if not self.is_trained:
            self.train_initial_model()
        if not self.is_trained:
            return self._heuristic_year_probability(number, context)
        try:
            features = self.extract_features(number, context)
            X_numeric = np.array([list(features.values())])
            X_text = self.vectorizer.transform([context])
            X_combined = np.hstack([X_numeric, X_text.toarray()])
            probabilities = self.classifier.predict_proba(X_combined)[0]
            return float(probabilities[1]) if len(probabilities) > 1 else 0.0
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
                return 0.1
            # Range-based scoring
            if len(number) == 2:
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
            return float(min(score, 1.0))
        except ValueError:
            return 0.0
    
    def add_training_example(self, number: str, context: str, is_year: bool):
        """Add a new training example and retrain if needed"""
        self.training_data.append((number, context, int(is_year)))
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
            logger.warning(f"Could not save year classifier model: {e}")
    
    def load_model(self):
        """Load existing model and data"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.vectorizer = model_data['vectorizer']
                self.training_data = model_data.get('training_data', [])
                self.is_trained = model_data.get('is_trained', False)
                logger.info(f"Loaded year classifier with {len(self.training_data)} training examples")
        except Exception as e:
            logger.warning(f"Could not load year classifier model: {e}")


class StudentInfoExtractor:
    def __init__(self, vic_suburbs: Set[str]):
        self.vic_suburbs = set(s.lower() for s in vic_suburbs)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Lazy fallback if the model is not installed
            logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = spacy.blank("en")
        self.year_classifier = YearClassifier()
        self.name_classifier = NameClassifier()
        
        # Initialize ML models
        if not self.year_classifier.is_trained:
            self.year_classifier.train_initial_model()
        if not self.name_classifier.is_trained:
            self.name_classifier.train_initial_model()

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
        """Enhanced name cleaning with name bank validation first, then ML"""
        orig = name.strip()
        if not orig:
            return None
        
        low = orig.lower()
        
        # Basic preprocessing
        # Check against suburbs first
        for sub in self.vic_suburbs:
            if sub in low:
                logger.info(f"❌ SUBURB REJECTION: '{orig}' contains suburb '{sub}'")
                return None
        
        # Filter allowed characters
        cleaned = re.sub(r"[^A-Za-z\-\s']", " ", orig)
        cleaned_parts = [p.strip() for p in cleaned.split() if p.strip()]
        if not cleaned_parts:
            return None
        
        # Length / structure validation
        if len(cleaned_parts) == 1:
            part = cleaned_parts[0]
            if len(part) < 2:
                return None
            candidate = part.capitalize()
        else:
            if any(len(p) <= 1 for p in cleaned_parts):
                return None
            candidate = " ".join(p.capitalize() for p in cleaned_parts)
        
        # Enhanced validation with name bank + ML
        is_real_name, confidence = self.name_classifier.predict_is_real_name(candidate)
        
        if is_real_name and confidence > 0.5:
            return candidate
        else:
            # Only add to training if we're confident it's not a name
            if confidence < 0.3:
                self.name_classifier.add_training_example(candidate, False)
            return None

    def extract_name_from_bbox(self, doc: fitz.Document) -> Optional[str]:
        candidates = self.extract_bbox_candidates(doc, NAME_BBOXES)
        cleaned = []
        for c in candidates:
            # Strip label text
            c2 = re.sub(r'(christian name|first name|surname|name)\s*:?', '', c, flags=re.I).strip()
            cn = self.clean_name_candidate(c2)
            if cn:
                cleaned.append(cn)
        if cleaned:
            return max(cleaned, key=lambda x: len(x))
        return None

    def extract_names_from_ner(self, text: str) -> List[str]:
        if not hasattr(self.nlp, "pipe_names") or "ner" not in self.nlp.pipe_names:
            return []
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
            if "christian name" in low or "first name" in low or "surname" in low or re.search(r'\bname\b', low):
                # Next line candidate
                if i + 1 < len(lines):
                    candidate = lines[i+1].strip()
                    cn = self.clean_name_candidate(candidate)
                    if cn:
                        return cn
                # Same line after label
                m = re.search(r'(?:christian name|first name|surname|name)\s*[:\-]\s*(.+)', line, flags=re.I)
                if m:
                    cn = self.clean_name_candidate(m.group(1).strip())
                    if cn:
                        return cn
        return None

    def get_most_frequent_name(self, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        cnt = Counter(candidates)
        most, _ = cnt.most_common(1)[0]
        return most

    def extract_suburb_from_bbox(self, doc: fitz.Document) -> Optional[str]:
        candidates = self.extract_bbox_candidates(doc, SUBURB_BBOXES)
        return self._suburb_best_match(candidates)

    def extract_suburb_from_text(self, text: str) -> Optional[str]:
        # Exact substring first
        for sub in self.vic_suburbs:
            if sub in text.lower():
                return sub.title()
        # Fuzzy match over 2-3 word substrings
        words = [w for w in re.findall(r"[A-Za-z']+", text)]
        substrings = []
        for idx in range(len(words)):
            for length in (2, 3):
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
        if best and best_score > 80:
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
        all_numbers: List[str] = []
        contexts: List[str] = []
        
        # Extract all numbers from full text with context
        lines = text.splitlines()
        for i, line in enumerate(lines):
            numbers = ALL_NUMBERS_PATTERN.findall(line)
            for num in numbers:
                # Context window (current line + surrounding lines)
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
        year_candidates: List[Tuple[int, float, str]] = []
        for num, context in zip(all_numbers, contexts):
            probability = self.year_classifier.predict_year_probability(num, context)
            if probability > self.year_classifier.confidence_threshold:
                try:
                    num_val = int(num)
                    if len(num) == 2:
                        year = 1900 + num_val
                    else:
                        year = num_val
                    if 1900 <= year <= 2000:
                        year_candidates.append((year, probability, context))
                except ValueError:
                    continue
        
        # Sort and take top ones
        year_candidates.sort(key=lambda x: x[1], reverse=True)
        
        final_years: Set[int] = set()
        for year, prob, context in year_candidates[:5]:
            final_years.add(year)
            if prob > 0.8:
                self.year_classifier.add_training_example(str(year), context, True)
        
        return final_years

    def extract_years_from_bbox(self, doc: fitz.Document) -> List[str]:
        """Extract text from year bounding boxes"""
        return self.extract_bbox_candidates(doc, YEAR_BBOXES)

    def extract_all(self, pdf_path: str) -> Tuple[Optional[str], str, Set[int]]:
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)

        # Names
        name_after_label = self.extract_name_after_label(full_text)
        name_bbox = self.extract_name_from_bbox(doc)
        names_ner = self.extract_names_from_ner(full_text)
        name_ner = self.get_most_frequent_name(names_ner)
        final_name = name_after_label or name_bbox or name_ner or "UNKNOWN"

        # Suburbs
        suburb_bbox = self.extract_suburb_from_bbox(doc)
        suburb_text = self.extract_suburb_from_text(full_text)
        final_suburb = suburb_bbox or suburb_text or "Melbourne"

        # Years
        bbox_candidates = self.extract_years_from_bbox(doc)
        final_years = self.extract_years_with_ml(full_text, bbox_candidates)

        doc.close()
        return final_name, final_suburb.title(), final_years


def test_name_validation():
    """Test the name validation system with various examples"""
    print("\n" + "=" * 60)
    print("NAME VALIDATION TESTING")
    print("=" * 60)
    
    # Initialize name classifier
    name_classifier = NameClassifier()
    
    # Test cases
    test_cases = [
        # Should be accepted (real names)
        "John Smith",
        "Mary Johnson", 
        "David Brown",
        "Sarah Wilson",
        "O'Connor",
        "Mary-Jane",
        "De Silva",
        "Van Der Berg",
        "James",
        "Elizabeth",
        "Michael",
        "Jennifer",
        
        # Should be rejected (non-names)
        "Barjna Road",
        "Melbourne Grammar", 
        "Smith Street",
        "Primary School",
        "Shopping Centre",
        "Myer",
        "Grammar School",
        "High School",
        "University",
        "Hospital",
        
        # Edge cases
        "Smith",  # Just surname
        "John",   # Just first name
        "St Kilda Road",  # Contains suburb/street
        "John 123",  # Contains numbers
    ]
    
    print("Testing name validation...")
    print("-" * 60)
    
    for test_name in test_cases:
        is_real_name, confidence = name_classifier.predict_is_real_name(test_name)
        status = "✅ ACCEPTED" if is_real_name else "❌ REJECTED"
        print(f"{status}: '{test_name}' (confidence: {confidence:.2f})")
    
    print("\nName bank statistics:")
    bank = name_classifier.name_bank
    if NLTK_AVAILABLE:
        print(f"NLTK names loaded: {len(bank.nltk_first_names)}")
    else:
        print("NLTK not available")
    print(f"Backup first names: {len(bank.backup_first_names)}")
    print(f"Common surnames: {len(bank.common_surnames)}")


def interactive_name_feedback(extractor: StudentInfoExtractor):
    """Interactive mode for providing feedback on name classification"""
    print("\n" + "=" * 60)
    print("Interactive Name Classification Improvement")
    print("Help improve name detection by providing feedback on extracted names.")
    print("The system now uses name banks (NLTK) + ML for better accuracy.")
    print("Format: 'NAME is/isn't a real name'")
    print("Examples:")
    print("  'John Smith is a real name'")
    print("  'Barjna Road is not a real name'")
    print("  'Melbourne Grammar is not a real name'")
    print("Enter 'done' to finish.")
    
    while True:
        feedback = input("\nName feedback (or 'done'): ").strip()
        if feedback.lower() in ['done', 'quit', 'exit', '']:
            break
        try:
            low = feedback.lower()
            if ' is a real name' in low:
                name = low.replace(' is a real name', '').strip()
                is_real_name = True
            elif ' is not a real name' in low:
                name = low.replace(' is not a real name', '').strip()
                is_real_name = False
            elif " isn't a real name" in low:
                name = low.replace(" isn't a real name", '').strip()
                is_real_name = False
            else:
                print("Invalid format. Use: 'NAME is/isn't a real name'")
                continue
            # Capitalize
            name = ' '.join(word.capitalize() for word in name.split())
            
            # Show current prediction before adding feedback
            current_prediction, current_confidence = extractor.name_classifier.predict_is_real_name(name)
            print(f"Current prediction: {'Real Name' if current_prediction else 'Not Real Name'} (confidence: {current_confidence:.2f})")
            
            extractor.name_classifier.add_training_example(name, is_real_name)
            print(f"Added training example: '{name}' -> {'Real Name' if is_real_name else 'Not a Real Name'}")
        except Exception as e:
            print(f"Error processing feedback: {e}")


if __name__ == "__main__":
    # Test name validation first
    test_name_validation()
    
    folder = os.path.abspath(os.path.dirname(__file__))
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]

    extractor = StudentInfoExtractor(VIC_SUBURBS)

    print("\n\n" + "=" * 70)
    print("Enhanced Student Info Extractor with Name Bank + ML Detection")
    print("=" * 70)
    
    if not pdfs:
        print("No PDF files found in current directory.")
        print("The system is ready and name bank validation is working!")
        print("Place PDF files in the script directory and run again.")
        
        # Interactive name feedback even without PDFs
        interactive_name_feedback(extractor)
        exit()
    
    # Track all extracted names for feedback
    all_extracted_names: List[str] = []
    
    for pdf in pdfs:
        path = os.path.join(folder, pdf)
        try:
            name, suburb, years = extractor.extract_all(path)
            print(f"File: {pdf}")
            print(f"  Name: {name}")
            print(f"  Suburb: {suburb}")
            print(f"  Years Detected: {sorted(years) if years else 'None'}")
            if name != "UNKNOWN":
                all_extracted_names.append(name)
            if years:
                print(f"  Total Years Found: {len(years)}")
                birth_years = [y for y in years if 1900 <= y <= 2000]
                if birth_years:
                    print(f"  Likely Birth Years: {sorted(birth_years)}")
            print()
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
            print()
    
    # Show model training statistics
    print("Model Statistics:")
    print(f"  Name Classifier Training Examples: {len(extractor.name_classifier.training_data)}")
    print(f"  Year Classifier Training Examples: {len(extractor.year_classifier.training_data)}")
    print(f"  Name Model Trained: {extractor.name_classifier.is_trained}")
    print(f"  Year Model Trained: {extractor.year_classifier.is_trained}")
    print(f"  Current NAME_REJECTION size: {len(NAME_REJECTION)}")
    
    # Show extracted names for review
    if all_extracted_names:
        print(f"\nExtracted Names Summary:")
        unique_names = set(all_extracted_names)
        for name in sorted(unique_names):
            is_real, confidence = extractor.name_classifier.predict_is_real_name(name)
            count = all_extracted_names.count(name)
            print(f"  {name} (found {count}x, confidence: {confidence:.2f}, classified as: {'Real Name' if is_real else 'Not Real Name'})")
    
    # Interactive name feedback
    interactive_name_feedback(extractor)
    
    # Interactive year improvement mode  
    print("\n" + "=" * 60)
    print("Interactive Year Classification Improvement")
    print("Review the results above. You can provide feedback to improve the year model.")
    print("Enter 'quit' to exit, or provide feedback in format:")
    print("filename.pdf: year YYYY is/isn't a birth year")
    print("Example: 'document1.pdf: year 1936 is a birth year'")
    print("Example: 'document1.pdf: year 123 is not a birth year'")
    
    while True:
        feedback = input("\nYear feedback (or 'quit'): ").strip()
        if feedback.lower() in ['quit', 'exit', '']:
            break
        try:
            if ':' not in feedback:
                print("Invalid format. Use: filename.pdf: year YYYY is/isn't a birth year")
                continue
            file_part, year_part = feedback.split(':', 1)
            file_part = file_part.strip()
            year_part = year_part.strip().lower()
            year_match = re.search(r'year\s+(\d+)', year_part)
            if not year_match:
                print("Could not find year in feedback. Use format: year YYYY is/isn't a birth year")
                continue
            year_str = year_match.group(1)
            is_year = ('is a' in year_part) or ("isn't a" not in year_part and 'not a' not in year_part and 'is not a' not in year_part)
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
            extractor.year_classifier.add_training_example(year_str, context, is_year)
            print(f"Added training example: {year_str} -> {'Year' if is_year else 'Not Year'}")
            print("Model will be retrained with new examples automatically.")
        except Exception as e:
            print(f"Error processing feedback: {e}")
            print("Please use format: filename.pdf: year YYYY is/isn't a birth year")
    
    print("\nThank you! The enhanced models have been improved with your feedback.")
    print("The name bank validation + ML models will be saved and used for future extractions.")
    print(f"Final NAME_REJECTION list now contains {len(NAME_REJECTION)} entries.")
    
    # Save the updated NAME_REJECTION list
    try:
        with open('name_rejection_list.json', 'w') as f:
            json.dump(sorted(list(NAME_REJECTION)), f, indent=2)
        print("Updated NAME_REJECTION list saved to 'name_rejection_list.json'")
    except Exception as e:
        print(f"Failed to save NAME_REJECTION list: {e}")