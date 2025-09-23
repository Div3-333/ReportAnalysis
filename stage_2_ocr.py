import fitz  # PyMuPDF
import re
from collections import Counter
from typing import Tuple, Optional, Set, List
import spacy
import os

class StudentInfoExtractor:
    def __init__(self, vic_suburbs: Set[str], name_bank: Set[str]):
        self.vic_suburbs = {suburb.lower() for suburb in vic_suburbs}  # store lowercase for checks
        self.name_bank = name_bank
        self.nlp = spacy.load("en_core_web_sm")  # SpaCy small model, fast for NER
        self.year_pattern = re.compile(r'\b(19\d{2}|20\d{2})\b')
        self.word_pattern = re.compile(r'\b\w+\b')

    def extract_text(self, pdf_path: str) -> str:
        text = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text.append(page.get_text())
        doc.close()
        return "\n".join(text).lower()

    def get_adjacent_name_pairs(self, words: List[str]) -> Counter:
        # Get indices of words that appear in the name bank
        candidate_indices = [i for i, w in enumerate(words) if w in self.name_bank and w not in self.vic_suburbs]
        pairs = []
        for i in range(len(candidate_indices) - 1):
            # Check if indices are adjacent words
            if candidate_indices[i + 1] == candidate_indices[i] + 1:
                pairs.append((words[candidate_indices[i]], words[candidate_indices[i + 1]]))
        return Counter(pairs)

    def find_frequent_name_pair(self, text: str, min_occurrences: int = 2) -> Optional[str]:
        words = self.word_pattern.findall(text)
        pair_counts = self.get_adjacent_name_pairs(words)
        if not pair_counts:
            return None
        most_common_pair, count = pair_counts.most_common(1)[0]
        if count >= min_occurrences:
            # Capitalize properly and return full name
            return f"{most_common_pair[0].capitalize()} {most_common_pair[1].capitalize()}"
        return None

    def extract_names_from_text(self, text: str) -> List[str]:
        doc = self.nlp(text)
        candidates = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name_parts = ent.text.lower().split()
                # Exclude if any part is a suburb (to avoid suburb names as names)
                if any(part in self.vic_suburbs for part in name_parts):
                    continue
                # Accept if at least one part is in name_bank or all parts alphabetic & length > 2
                if (
                    any(part in self.name_bank for part in name_parts)
                    or all(part.isalpha() and len(part) > 2 for part in name_parts)
                ):
                    candidates.append(" ".join(part.capitalize() for part in name_parts))
        return candidates

    def extract_names_near_label_with_ner(self, text: str, labels: List[str], lines_before=5, lines_after=5) -> Optional[str]:
        lines = text.splitlines()
        labels_lower = [label.lower() for label in labels]

        for i, line in enumerate(lines):
            if any(label in line for label in labels_lower):
                # Check lines after the label line first (more likely)
                for offset in range(1, lines_after + 1):
                    if i + offset < len(lines):
                        names = self.extract_names_from_text(lines[i + offset])
                        if names:
                            return names[0]

                # If not found after, check lines before label line
                for offset in range(1, lines_before + 1):
                    if i - offset >= 0:
                        names = self.extract_names_from_text(lines[i - offset])
                        if names:
                            return names[0]
        return None

    def find_suburb(self, text: str) -> Optional[str]:
        for suburb in self.vic_suburbs:
            if suburb in text:
                return suburb.title()
        return None

    def find_years(self, text: str) -> Set[int]:
        return set(int(y) for y in self.year_pattern.findall(text))

    def extract_all(self, pdf_path: str) -> Tuple[Optional[str], Optional[str], Set[int]]:
        text = self.extract_text(pdf_path)

        # 1. Highest priority: Frequent full adjacent name pairs (more reliable)
        student_name = self.find_frequent_name_pair(text, min_occurrences=2)

        # 2. If no frequent pair, fallback to NER + proximity to "christian name" or "first name"
        if not student_name:
            student_name = self.extract_names_near_label_with_ner(text, ["christian name", "first name"])

        suburb = self.find_suburb(text)
        years = self.find_years(text)
        return student_name, suburb, years


if __name__ == "__main__":
    from nltk.corpus import names

    # Victorian suburbs list (already in lowercase internally)
    vic_suburbs = [
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
    ]

    name_bank = {n.lower() for n in names.words('male.txt')} | {n.lower() for n in names.words('female.txt')}

    extractor = StudentInfoExtractor(vic_suburbs, name_bank)

    folder_path = os.path.abspath(os.path.dirname(__file__))
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        text = extractor.extract_text(full_path)
        all_candidates = extractor.extract_names_from_text(text)
        print(f"File: {pdf_file}")
        print(f"  First 5 candidate names (for debug): {all_candidates[:5]}")
        student_name, suburb, years = extractor.extract_all(full_path)
        print(f"  Student Name: {student_name}")
        print(f"  Suburb: {suburb}")
        print(f"  Years Mentioned: {sorted(years)}\n")
