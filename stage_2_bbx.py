import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
import os

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Bounding boxes: (x0, y0, x1, y1)
# Coordinates normalized as rectangles with some height for single-line coords
BBOXES = {
    "name": [
        # First page (0-based indexing)
        (288, 508, 716, 510),
        (290, 526, 703, 530),
        # Second page
        (15, 496, 168, 498),
        (28, 519, 171, 522),
    ],
    "suburb": [
        # First page only
        (111, 480, 272, 490),
        (114, 514, 266, 516),
    ],
    "years": [
        # First page only
        (97, 170, 159, 172),
        (99, 376, 158, 386),
    ]
}

# Helper: clean and normalize OCR text lines
def clean_text(text):
    return text.strip().replace('\n', ' ').replace('\r', '').strip()

def extract_text_from_bbox(page, bbox):
    """
    Extract image from bbox area, run OCR, return text.
    """
    x0, y0, x1, y1 = bbox
    # Crop the pixmap (page.get_pixmap()) to bbox region
    clip = fitz.Rect(x0, y0, x1, y1)
    pix = page.get_pixmap(clip=clip, dpi=300)  # higher dpi for better OCR
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # OCR on cropped image
    text = pytesseract.image_to_string(img, config='--psm 6')  # assume a single uniform block of text
    
    return clean_text(text)

def parse_name_text(lines):
    """
    Given list of lines (strings), apply heuristics to extract best student full name.
    
    Heuristics:
    - Look for 'Christian Name:', 'First Name:', 'Surname:' labels and parse accordingly
    - If no labels, accept lines with 2+ English-letter words (likely full names)
    - Reject 1-letter or non-alpha entries
    - Combine christian/first name + surname if both found
    """
    christian_name = None
    surname = None
    loose_names = []
    
    label_patterns = {
        'christian': re.compile(r'^(Christian Name|First Name|Christian|Firstname|First):\s*(.*)', re.I),
        'surname': re.compile(r'^Surname:\s*(.*)', re.I)
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        matched = False
        for key, pattern in label_patterns.items():
            m = pattern.match(line)
            if m:
                name_part = m.group(2).strip()
                # Validate name part (alpha + spaces only, no single letters)
                if name_part and all(len(w) > 1 for w in name_part.split()) and all(w.isalpha() for w in name_part.split()):
                    if key == 'christian':
                        christian_name = name_part
                    elif key == 'surname':
                        surname = name_part
                matched = True
                break
        if not matched:
            # If no label, check if looks like a name line: only letters, 2+ words
            words = line.split()
            if len(words) >= 2 and all(w.isalpha() and len(w) > 1 for w in words):
                loose_names.append(line)
    
    # Decide on final name
    if christian_name and surname:
        return f"{christian_name} {surname}"
    elif christian_name:
        return christian_name
    elif surname:
        return surname
    elif loose_names:
        # Return first loose name candidate
        return loose_names[0]
    else:
        return None

def extract_names_from_bboxes(doc):
    """
    Extract candidate names from name bboxes on pages.
    Returns the best candidate or None.
    """
    candidate_lines = []
    for i, bbox in enumerate(BBOXES['name']):
        # first two bboxes on page 0, next two on page 1
        page_num = 0 if i < 2 else 1
        if page_num >= len(doc):
            continue
        
        page = doc[page_num]
        text = extract_text_from_bbox(page, bbox)
        if text:
            # Sometimes OCR returns multiple lines
            candidate_lines.extend(text.splitlines())
    
    return parse_name_text(candidate_lines)

def extract_suburb_from_bboxes(doc):
    """
    Extract suburb text from first page suburb bboxes.
    Return most common or concatenated suburb strings.
    """
    if len(doc) == 0:
        return None
    
    page = doc[0]
    suburb_candidates = []
    for bbox in BBOXES['suburb']:
        text = extract_text_from_bbox(page, bbox)
        if text:
            suburb_candidates.append(text.strip())
    
    if not suburb_candidates:
        return None
    
    # Simple heuristic: return longest suburb text (assuming it's full)
    return max(suburb_candidates, key=len)

def extract_years_from_bboxes(doc):
    """
    Extract years mentioned from first page years bboxes.
    Return a set of unique years (as integers).
    """
    if len(doc) == 0:
        return set()
    
    page = doc[0]
    years_found = set()
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    
    for bbox in BBOXES['years']:
        text = extract_text_from_bbox(page, bbox)
        if text:
            matches = year_pattern.findall(text)
            # Since regex only captures prefix, full match is group(0)
            # Let's find all 4-digit years in the text
            all_years = re.findall(r'\b(19|20)\d{2}\b', text)
            # Actually above returns only 2 digit prefixes, better to find all matches
            all_years = re.findall(r'\b(19|20)\d{2}\b', text)
            for m in re.findall(r'\b(19|20)\d{2}\b', text):
                try:
                    y = int(m)
                    years_found.add(y)
                except:
                    pass

            # More robust way:
            all_years = re.findall(r'\b(19|20)\d{2}\b', text)
            for y in all_years:
                years_found.add(int(y))
    
    return years_found

class StudentInfoExtractor:
    def extract_all(self, pdf_path):
        doc = fitz.open(pdf_path)
        student_name = extract_names_from_bboxes(doc)
        suburb = extract_suburb_from_bboxes(doc)
        years = extract_years_from_bboxes(doc)
        return student_name, suburb, years


if __name__ == "__main__":
    extractor = StudentInfoExtractor()
    folder_path = os.path.abspath(os.path.dirname(__file__))
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        student_name, suburb, years = extractor.extract_all(full_path)
        print(f"File: {pdf_file}")
        print(f"  Student Name: {student_name}")
        print(f"  Suburb: {suburb}")
        print(f"  Years Mentioned: {sorted(years)}\n")
