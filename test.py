import fitz  # PyMuPDF

import fitz  # PyMuPDF
import re
import spacy
from collections import Counter
from typing import List, Optional, Tuple, Set
import os
from rapidfuzz import process, fuzz

# Rejection and threshold definitions
NAME_REJECTION = {
    "year", "special comments", "occupation", "leadership", "mathematics",
    "id", "activities", "behaviour", "report", "teacher", "class", "subject",
    "remarks", "date", "school", "name", "comments", "address", "parent", "surname", "road", "farmer", "street", "strcct", "birth", "mother's", "avenue", "anticipated", "jeweller", "mile", "hawskburn", "manufacturer"
}

# VIC Suburbs (full list you’ve got)
VIC_SUBURBS = {
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
    'melton', 'melton south', 'melton west', 'mentone', 'merton', 'mentone', 'mentone gardens',
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
}

YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')

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

class StudentInfoExtractor:
    def __init__(self, vic_suburbs: Set[str]):
        self.vic_suburbs = set(s.lower() for s in vic_suburbs)
        self.nlp = spacy.load("en_core_web_sm")

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
        if best and best_score > 80:
            return best.title()
        return None

    def extract_years_from_bbox(self, doc: fitz.Document) -> Set[int]:
        candidates = self.extract_bbox_candidates(doc, YEAR_BBOXES)
        # join them
        return set(int(y) for t in candidates for y in YEAR_PATTERN.findall(t))

    def extract_years_from_text(self, text: str) -> Set[int]:
        return set(int(y) for y in YEAR_PATTERN.findall(text))

    def extract_all(self, pdf_path: str) -> Tuple[Optional[str], str, Set[int]]:
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)

        # Names
        name_after_label = self.extract_name_after_label(full_text)
        name_bbox = self.extract_name_from_bbox(doc)
        names_ner = self.extract_names_from_ner(full_text)
        name_ner = self.get_most_frequent_name(names_ner)

        # pick name in order
        final_name = name_after_label or name_bbox or name_ner or "UNKNOWN"

        # Suburbs
        suburb_bbox = self.extract_suburb_from_bbox(doc)
        suburb_text = self.extract_suburb_from_text(full_text)
        final_suburb = suburb_bbox or suburb_text or "Melbourne"  # default if absolutely nothing

        # Years
        years_bbox = self.extract_years_from_bbox(doc)
        years_full = self.extract_years_from_text(full_text)
        final_years = years_bbox.union(years_full)

        doc.close()
        return final_name, final_suburb.title(), final_years


if __name__ == "__main__":
    folder = os.path.abspath(os.path.dirname(__file__))
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]

    extractor = StudentInfoExtractor(VIC_SUBURBS)

    for pdf in pdfs:
        path = os.path.join(folder, pdf)
        name, suburb, years = extractor.extract_all(path)
        print(f"File: {pdf}")
        print(f"  Name: {name}")
        print(f"  Suburb: {suburb}")
        print(f"  Years Mentioned: {sorted(years)}\n")
