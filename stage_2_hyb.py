import fitz  # PyMuPDF
import re
import spacy
from collections import Counter
from typing import List, Optional, Tuple, Set
import os

# You can update this list or load from a file if needed
VIC_SUBURBS = set([
    'abbotsford', 'aberdeen', 'airport west', 'albany', 'albanvale', 'alberton', 'albert park',
    'albion', 'altona', 'altona east', 'altona meadows', 'altona north', 'amelia park', 'anderson',
    'angle vale', 'anglesea', 'armadale', 'arnold', 'ashburton', 'ashwood',
    'aspendale', 'aspendale gardens', 'attwood', 'auburn', 'avalanche', 'avondale heights', 'avonsleigh',
    'baddaginnie', 'badger creek', 'balaclava', 'balcombe', 'balwyn', 'balwyn east', 'balwyn north',
    'bangholme', 'bayswater', 'bayswater north', 'beaconsfield', 'beaconsfield upper', 'beaumaris',
    'bellfield', 'belgrave', 'belgrave heights', 'bentleigh', 'bentleigh east', 'berwick',
    'briar hill', 'brighton', 'brighton east', 'broadmeadows', 'brookfield', 'brooklyn', 'brunswick',
    'brunswick east', 'brunswick west', 'bulleen', 'bundoora', 'burnley', 'burwood', 'burwood east',
    'cairnlea', 'camberwell', 'carnegie', 'carlton', 'carlton north', 'caroline springs',
    'carrum', 'carrum downs', 'caulfield', 'caulfield east', 'caulfield north', 'caulfield south',
    'chelsea', 'chelsea heights', 'cheltenham', 'cheshire', 'chadstone', 'chantilly', 'chapel street',
    'chatsworth', 'clifton hill', 'clayton', 'clayton south', 'clyde', 'clyde north',
    'coburg', 'coburg north', 'coogee', 'cockatoo', 'coldstream', 'collingwood',
    'coolaroo', 'cranbourne', 'cranbourne east', 'cranbourne north', 'cranbourne south', 'cranbourne west',
    'dandenong', 'dandenong north', 'dandenong south', 'dee why', 'deer park', 'delahey',
    'diamond creek', 'diggers rest', 'doncaster', 'doncaster east', 'donvale', 'doreen', 'doveton',
    'elm street', 'elsternwick', 'eltham', 'eltham north', 'elwood', 'endeavour hills', 'epping', 'essendon',
    'essendon north', 'essendon west', 'evening bent', 'exford', 'ferntree gully', 'fieldstone', 'fitzroy',
    'fitzroy north', 'flemington', 'floods creek', 'footscray', 'gardenvale', 'garfield', 'glen huntly',
    'glen iris', 'glenroy', 'glebe', 'glen waverley', 'glenwood', 'greendale', 'grangefields', 'greensborough',
    'greenvale', 'hallam', 'hampton', 'hampton east', 'hampton park', 'harcourt', 'hawthorn', 'hawthorn east',
    'hendersons', 'heatherton', 'heidelberg', 'heidelberg heights', 'heidelberg west', 'higher ridges', 'highett',
    'hillside', 'hoppers crossing', 'huntingdale', 'humevale', 'hurlstone park', 'highett', 'ivanhoe',
    'ivanhoe east', 'jacana', 'junction village', 'kalkallo', 'kallista', 'kangaroo ground',
    'kealba', 'keilor', 'keilor east', 'keilor north', 'keilor lodge', 'keen', 'kew', 'kew east',
    'kilsyth', 'kilsyth south', 'kings park', 'kingsville', 'knoxfield', 'kooyong', 'lalor', 'lang lang',
    'langwarrin', 'laverton', 'laverton north', 'lilydale', 'little river', 'lower plenty', 'lysterfield',
    'macclesfield', 'macleod', 'maidstone', 'malvern', 'malvern east', 'mentone', 'mentone east',
    'melbourne airport', 'melton', 'melton south', 'melton west', 'mentone', 'merton', 'mentone', 'mentone gardens',
    'mernda', 'mentone', 'middle park', 'mile end', 'mill park', 'mitcham', 'monbulk', 'mont albert north',
    'montmorency', 'moonee ponds', 'mooroolbark', 'mordialloc', 'mornington', 'mount eliza', 'mount martha',
    'mount waverley', 'mulgrave', 'murundindi', 'narre warren', 'narre warren north', 'noble park', 'noble park north',
    'north melbourne', 'northcote', 'north warrandyte', 'notting hill', 'nunawading', 'oak park',
    'oakleigh', 'oakleigh east', 'oakleigh south', 'old orchard', 'olinda', 'ormond', 'pakenham', 'park orchards',
    'pascoe vale', 'pascoe vale south', 'point cook', 'port melbourne', 'prahran', 'preston', 'queenscliff', 'richmond',
    'ringwood', 'ringwood east', 'ringwood north', 'rosanna', 'rowville', 'rye', 'sandringham', 'seaford', 'seaholme',
    'south yarra', 'southbank', 'south melbourne', 'south morang', 'south yarra', 'spotswood', 'springvale', 'st albans',
    'st andrews', 'st kilda', 'st kilda east', 'st kilda west', 'sunbury', 'sunshine', 'sunshine north', 'sunshine west',
    'surrey hills', 'sydenham', 'tarneit', 'templeton', 'templeton lower', 'templeton upper', 'templeton east', 'the basin',
    'thomastown', 'thornbury', 'toorak', 'tooradin', 'truganina', 'tullamarine', 'tyabb', 'uparfax', 'vermont', 'vermont south',
    'viewbank', 'wantirna', 'wantirna south', 'warrandyte', 'warrandyte south', 'warranwood', 'waterways', 'werribee', 'werribee south',
    'west melbourne', 'west footscray', 'west acton', 'wheelers hill', 'whittlesea', 'williamstown', 'williamstown north', 'windsor',
    'wollert', 'wonga park', 'woolamai', 'yarra junction', 'yarra glen', 'yarraville' 
])

YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')

# Bounding box coordinates are given as (x0, y0, x1, y1)
# Page numbering in PyMuPDF starts at 0

NAME_BBOXES = {
    0: [  # first page
        (290, 510, 716, 510),
        (290, 530, 716, 530),
    ],
    1: [  # second page
        (15, 500, 170, 500),
        (15, 525, 170, 525),
    ],
}

SUBURB_BBOXES = {
    0: [
        (115, 485, 275, 485),
        (115, 515, 275, 515),
    ]
}

YEAR_BBOXES = {
    0: [
        (100, 172, 160, 172),
        (100, 381, 160, 381),
    ]
}

class StudentInfoExtractor:
    def __init__(self, vic_suburbs: Set[str]):
        self.vic_suburbs = set(s.lower() for s in vic_suburbs)
        self.nlp = spacy.load("en_core_web_sm")

    def extract_text_from_bbox(self, doc: fitz.Document, page_num: int, bbox: Tuple[int, int, int, int]) -> str:
        """Extract text within the bbox rectangle on a specific page."""
        page = doc[page_num]
        rect = fitz.Rect(bbox)
        words = page.get_text("words")  # list of words on the page
        # Select words inside bbox by checking if the bottom-left corner of the word is inside the rect
        selected_words = [w[4] for w in words if rect.contains(fitz.Point(w[0], w[1]))]
        return " ".join(selected_words).strip()

    def extract_bbox_candidates(self, doc: fitz.Document, bboxes: dict) -> List[str]:
        candidates = []
        for page_num, boxes in bboxes.items():
            for bbox in boxes:
                text = self.extract_text_from_bbox(doc, page_num, bbox)
                if text:
                    # Debug print
                    # print(f"[DEBUG] Page {page_num+1} bbox {bbox} extracted text: '{text}'")
                    candidates.append(text)
        return candidates

    def clean_name_candidate(self, name: str) -> Optional[str]:
        """
        Clean and validate extracted candidate names.
        Rules:
        - Only English letters and spaces
        - No single letter parts
        - Capitalize properly
        """
        name = name.strip()
        if not name:
            return None
        parts = name.split()
        if len(parts) == 0:
            return None
        for p in parts:
            if not p.isalpha() or len(p) <= 1:
                return None
        return " ".join(p.capitalize() for p in parts)

    def extract_years_from_text(self, texts: List[str]) -> Set[int]:
        years = set()
        for text in texts:
            found = YEAR_PATTERN.findall(text)
            for y in found:
                years.add(int(y))
        return years

    def extract_suburb_from_candidates(self, candidates: List[str]) -> List[str]:
        """
        Return all candidates that match a known Victorian suburb.
        """
        matched_suburbs = []
        for candidate in candidates:
            c = candidate.lower().strip()
            for suburb in self.vic_suburbs:
                if suburb in c:
                    matched_suburbs.append(suburb.title())
        return matched_suburbs

    def extract_name_from_bbox(self, doc: fitz.Document) -> List[str]:
        candidates = self.extract_bbox_candidates(doc, NAME_BBOXES)
        cleaned_candidates = []
        for c in candidates:
            c_clean = re.sub(r'(christian name|first name|surname|:)', '', c, flags=re.I).strip()
            cleaned = self.clean_name_candidate(c_clean)
            if cleaned:
                cleaned_candidates.append(cleaned)
        return cleaned_candidates

    def extract_suburb_from_bbox(self, doc: fitz.Document) -> List[str]:
        candidates = self.extract_bbox_candidates(doc, SUBURB_BBOXES)
        return self.extract_suburb_from_candidates(candidates)

    def extract_years_from_bbox(self, doc: fitz.Document) -> Set[int]:
        candidates = self.extract_bbox_candidates(doc, YEAR_BBOXES)
        return self.extract_years_from_text(candidates)

    def extract_text_full(self, doc: fitz.Document) -> str:
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n".join(texts).lower()

    def extract_names_from_full_text(self, text: str) -> List[str]:
        doc = self.nlp(text)
        names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate = ent.text.strip()
                cleaned = self.clean_name_candidate(candidate)
                if cleaned:
                    names.append(cleaned)
        return names

    def extract_suburbs_from_full_text(self, text: str) -> List[str]:
        found = []
        for suburb in self.vic_suburbs:
            if suburb in text:
                found.append(suburb.title())
        return found

    def extract_years_from_full_text(self, text: str) -> Set[int]:
        return self.extract_years_from_text([text])

    # --- Combination functions ---
    @staticmethod
    def combine_name_candidates(bbox_names: List[str], ner_names: List[str]) -> Optional[str]:
        combined = bbox_names + ner_names
        if not combined:
            return None
        count = Counter(combined)
        # Pick the most common candidate
        most_common, freq = count.most_common(1)[0]

        # Prefer bbox candidate if it appears more than once (confidence)
        for name in bbox_names:
            if name in count and count[name] > 1:
                return name
        return most_common

    @staticmethod
    def combine_suburb_candidates(bbox_suburbs: List[str], full_text_suburbs: List[str]) -> Optional[str]:
        bbox_set = set(s.lower() for s in bbox_suburbs)
        full_set = set(s.lower() for s in full_text_suburbs)
        intersection = bbox_set.intersection(full_set)
        if intersection:
            # Pick longest suburb name (to avoid partial matches)
            return max(intersection, key=len).title()
        if bbox_set:
            return max(bbox_set, key=len).title()
        if full_set:
            return max(full_set, key=len).title()
        return None

    @staticmethod
    def combine_years(bbox_years: Set[int], full_years: Set[int]) -> Set[int]:
        # Union of both sets
        return bbox_years.union(full_years)

    # --- Main extraction method ---
    def extract_all(self, pdf_path: str) -> Tuple[Optional[str], Optional[str], Set[int]]:
        doc = fitz.open(pdf_path)

        # Extract from bbox approach
        bbox_names = self.extract_name_from_bbox(doc)
        bbox_suburbs = self.extract_suburb_from_bbox(doc)
        bbox_years = self.extract_years_from_bbox(doc)

        # Extract from full text / NER approach
        full_text = self.extract_text_full(doc)
        ner_names = self.extract_names_from_full_text(full_text)
        full_text_suburbs = self.extract_suburbs_from_full_text(full_text)
        full_years = self.extract_years_from_full_text(full_text)

        doc.close()

        # Combine all using hybrid logic
        final_name = self.combine_name_candidates(bbox_names, ner_names)
        final_suburb = self.combine_suburb_candidates(bbox_suburbs, full_text_suburbs)
        final_years = self.combine_years(bbox_years, full_years)

        return final_name, final_suburb, final_years

# --- Example usage ---
if __name__ == "__main__":
    import sys

    folder_path = os.path.abspath(os.path.dirname(__file__)) if len(sys.argv) == 1 else sys.argv[1]
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    extractor = StudentInfoExtractor(VIC_SUBURBS)

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        student_name, suburb, years = extractor.extract_all(full_path)
        print(f"File: {pdf_file}")
        print(f"  Student Name: {student_name}")
        print(f"  Suburb: {suburb}")
        print(f"  Years Mentioned: {sorted(years)}\n")

