import os
import sys
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Set
import logging

# --- NLTK Installation and Setup ---
try:
    import nltk
    from nltk.corpus import names
except ImportError:
    print("NLTK library not found. Please install it by running: pip install nltk")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StudentReportSplitter:
    def __init__(self, output_dir: str = "split_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.name_bank = self._load_name_bank()

        # Build or import a list of Victorian suburbs, for rejecting names
        self.vic_suburbs = self._load_vic_suburbs()

        # Expanded list of common form labels / terms to reject
        # This includes:
        #   - original list,
        #   - new terms (“AI”, “date of birth”, etc),
        #   - anything with “high school”,
        #   - anything with non‐English letters or special chars (approx),
        #   - Victorian suburb names
        rejection_terms = [
            'surname', 'class', 'date', 'year', 'number', 'form', 'student',
            'christian', 'first', 'address', 'activities', 'occupation',
            'scholastic', 'special comments', 'subsequent career', 'remarks',
            'telephone', 'guardian', 'father', 'mother', 'employer',

            # New terms
            'ai', 'date of birth', 'form teacher', 'individual',
            # Anything with “high school”:
            'high school',

            # Non-English / foreign / odd character indicators (approximate pattern):
            # We will detect names with accent letters or non‑ASCII letters via regex

            # Victorian suburbs (list below)
        ]

        # Combine the suburbs into the rejection list
        # Lowercase all for uniformity
        self.rejection_suburbs = { suburb.lower() for suburb in self.vic_suburbs }

        # We combine into a regex safe pattern
        # Escape suburb names in case of special regex chars
        suburbs_pattern = '|'.join(re.escape(s) for s in self.rejection_suburbs if s)

        # Build the full REJECTION_WORDS regex
        # This will match if candidate exactly matches one of these terms (case‐insensitive),
        # OR if it contains “high school”, or if the name matches a suburb, or if it contains non‐ASCII letters / weird chars.
        self.REJECTION_WORDS = (
            rf'^(?:{ "|".join(re.escape(t) for t in rejection_terms) }(?:\s+|$)'  # <-- FIXED: closing paren moved here
            rf'|.*\bhigh school\b'
            rf'|.*(?:{ suburbs_pattern })\b'
            rf')$'
        )

        # Regex to detect non-ASCII / foreign / special characters
        # E.g. accents, diacritics, etc.
        self.NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]')

    def _load_name_bank(self) -> Set[str]:
        try:
            logger.info("Loading name bank...")
            male_names = {name.lower() for name in names.words('male.txt')}
            female_names = {name.lower() for name in names.words('female.txt')}
            name_bank = male_names.union(female_names)
            logger.info(f"Successfully loaded {len(name_bank)} unique names.")
            return name_bank
        except LookupError:
            logger.info("NLTK 'names' corpus not found. Attempting to download...")
            nltk.download('names')
            return self._load_name_bank()
        except Exception as e:
            logger.error(f"Could not load or download the NLTK names corpus: {e}")
            return set()

    def _load_vic_suburbs(self) -> Set[str]:
        """
        Returns a set of lowercased Victorian suburbs.
        This list is not absolutely complete but covers many suburbs in Greater Melbourne.
        You can replace or extend it with your own source if needed.
        """
        suburbs = [
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
            'macclesfield', 'macleod', 'maidstone', 'malvern', 'malvern east', 'mentone', 'mentone east', 'melbourne',
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
        ]
        # Normalize to lower case and strip extra whitespace
        return { suburb.strip().lower() for suburb in suburbs if suburb and isinstance(suburb, str) }


    def _normalize_text(self, text: str) -> str:
        """Cleans text by replacing non-standard unicode and whitespace."""
        if not text:
            return ""
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = re.sub(r'[’‘´`]', "'", text)  # Fancy quotes to apostrophe
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _is_in_name_bank(self, candidate: str) -> bool:
        if not candidate or not self.name_bank:
            return False
        first_word = candidate.split(' ')[0].lower()
        return first_word in self.name_bank

    def _is_plausible_format(self, candidate: str) -> bool:
        """Checks if a string has the format of a name and isn't a known label."""
        if not candidate:
            return False

        low = candidate.lower().strip()

        # Reject if matches the rejection words exactly or matches containing pattern
        if re.fullmatch(self.REJECTION_WORDS, low, re.IGNORECASE):
            return False

        # Reject if contains non-ASCII / foreign‑looking characters
        if self.NON_ASCII_PATTERN.search(candidate):
            return False

        # Otherwise check plausible name format: only English letters, spaces, hyphens, apostrophes; length limits
        return re.fullmatch(r"[A-Za-z\s\-']{2,50}", candidate) is not None

    def extract_text_from_page(self, page) -> str:
        return page.get_text("text")

    def detect_student_name(self, text: str) -> str:
        LABEL_PATTERN = r'CHRISTIAN\s+NAME|FIRST\s+NAMES'
        medium_confidence_name = None
        candidates = []

        # 1. From the same line as the label
        same_line_pattern = re.compile(
            rf'{LABEL_PATTERN}.*?:\s*([A-Za-z\s\-\'"]+?)(?:\n|$)', re.IGNORECASE
        )
        match = same_line_pattern.search(text)
        if match and match.group(1):
            candidates.append(match.group(1))

        # 2. From the window around the label
        raw_lines = text.split('\n')
        for i, line in enumerate(raw_lines):
            if re.search(LABEL_PATTERN, line, re.IGNORECASE):
                start = max(0, i - 5)
                end = min(len(raw_lines), i + 6)
                for j in range(start, end):
                    if i == j:
                        continue
                    candidates.append(raw_lines[j])
                break

        for raw_candidate in candidates:
            candidate = self._normalize_text(raw_candidate)
            if not candidate:
                continue

            if self._is_plausible_format(candidate):
                if self._is_in_name_bank(candidate):
                    logger.debug(f"High-confidence name found: '{candidate}'")
                    return candidate
                if medium_confidence_name is None:
                    logger.debug(f"Medium-confidence name found: '{candidate}'")
                    medium_confidence_name = candidate

        return medium_confidence_name

    def sanitize_filename(self, name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
        return self._normalize_text(sanitized).replace(' ', '_')

    def _save_student_pages(self, doc, student_name: str, page_numbers: List[int]):
        if not page_numbers:
            return

        sanitized_name = self.sanitize_filename(student_name)
        if len(sanitized_name) > 50:
            sanitized_name = sanitized_name[:50]

        output_path = self.output_dir / f"{sanitized_name}.pdf"
        counter = 1
        original_stem = output_path.stem

        while output_path.exists():
            output_path = self.output_dir / f"{original_stem}_{counter}.pdf"
            counter += 1

        new_doc = fitz.open()
        for page_num in page_numbers:
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        try:
            new_doc.save(str(output_path))
            logger.info(f"Saved {len(page_numbers)} pages for '{student_name}' to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save PDF for '{student_name}': {e}")
        finally:
            new_doc.close()

    def split_pdf(self, input_path: str, report_page_length: int = 2) -> Dict[str, int]:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Processing PDF: {input_path} (Enforcing {report_page_length}-page limit)")
        doc = fitz.open(input_path)

        stats = {
            'total_pages': len(doc),
            'students_found': 0,
            'unknown_reports': 0
        }

        current_student_pages: List[int] = []
        current_student_name: str = None
        unknown_counter = 1

        for page_num, page in enumerate(doc):
            text = self.extract_text_from_page(page)
            detected_name = self.detect_student_name(text)
            if detected_name:
                detected_name = self._normalize_text(detected_name)

            is_new_student = detected_name and detected_name != current_student_name
            is_report_full = len(current_student_pages) >= report_page_length

            if current_student_pages and (is_new_student or is_report_full):
                report_name = current_student_name or f"Unknown_Student_Report_{unknown_counter}"
                if current_student_name:
                    stats['students_found'] += 1
                else:
                    stats['unknown_reports'] += 1
                    unknown_counter += 1
                self._save_student_pages(doc, report_name, current_student_pages)
                current_student_pages = []
                current_student_name = None

            if is_new_student:
                current_student_name = detected_name

            current_student_pages.append(page_num)

        # Save remaining pages
        if current_student_pages:
            report_name = current_student_name or f"Unknown_Student_Report_{unknown_counter}"
            if current_student_name:
                stats['students_found'] += 1
            else:
                stats['unknown_reports'] += 1
            self._save_student_pages(doc, report_name, current_student_pages)

        doc.close()

        total_reports = stats['students_found'] + stats['unknown_reports']
        logger.info(
            f"Split complete. Found {stats['students_found']} named students and "
            f"{stats['unknown_reports']} unknown reports, for a total of {total_reports} reports."
        )

        return stats


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI entry point could be implemented here
        pass
    else:
        print("🎯 Student Report PDF Splitter - Test Mode")
        print("-" * 50)

        pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
        if not pdf_files:
            print("❌ No PDF files found in the current directory.")
            exit(1)

        pdf_to_process = pdf_files[0]
        print(f"📄 Found PDF file: '{pdf_to_process}'")

        try:
            splitter = StudentReportSplitter(output_dir="split_reports")
            print("\n🚀 Running splitter with expanded rejection rules...")
            stats = splitter.split_pdf(pdf_to_process, report_page_length=2)

            print("\n✅ Processing complete!")
            print("=" * 50)
            print(f"📊 Total pages: {stats['total_pages']}")
            print(f"👥 Students found (named): {stats['students_found']}")
            print(f"⚠️  Unknown reports saved: {stats['unknown_reports']}")
            print(f"🔢 Total reports created: {stats['students_found'] + stats['unknown_reports']}")
            print("=" * 50)

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()
