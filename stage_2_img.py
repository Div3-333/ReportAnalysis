import fitz  # PyMuPDF
import os
import re
from typing import Optional, List, Dict


class StudentNameExtractor:
    def __init__(self):
        # Regex patterns to detect labels with optional names after colon or newline
        self.labels = {
            "first_name": re.compile(r"(?:christian name|first name|first names)\s*[:\-]?\s*$", re.IGNORECASE),
            "surname": re.compile(r"surname\s*[:\-]?\s*$", re.IGNORECASE),
        }

    def extract_text_from_bbox(self, page, bbox):
        rect = fitz.Rect(*bbox)
        text = page.get_text("text", clip=rect)
        return text.strip()

    def clean_name(self, raw_name: str) -> str:
        # Keep only letters and spaces, capitalize properly
        cleaned = re.sub(r"[^A-Za-z\s]", "", raw_name).strip()
        parts = [p.capitalize() for p in cleaned.split() if len(p) > 1]
        return " ".join(parts)

    def is_valid_name_line(self, line: str) -> bool:
        # Accept lines with letters, spaces, and optionally one comma separating surname, firstname(s)
        # We'll be flexible here to allow comma
        if ',' in line:
            parts = line.split(',')
            if len(parts) == 2:
                surname_part = parts[0].strip()
                firstname_part = parts[1].strip()
                # Both parts should be alphabetic-ish with spaces allowed
                if all(p.isalpha() for p in surname_part.split()) and all(p.isalpha() for p in firstname_part.split()):
                    return True
            return False
        else:
            parts = line.strip().split()
            if not parts or any(len(p) <= 1 for p in parts):
                return False
            if all(p.isalpha() for p in parts):
                return True
            return False

    def parse_label_and_name(self, text: str) -> Dict[str, Optional[str]]:
        results = {}
        for key, pattern in self.labels.items():
            if pattern.match(text):
                results[key] = None
            else:
                label_pattern = re.compile(
                    rf"(?:{key.replace('_', ' ')})\s*[:\-]?\s*(.*)", re.IGNORECASE
                )
                m = label_pattern.match(text)
                if m:
                    candidate = m.group(1).strip()
                    if candidate:
                        cleaned = self.clean_name(candidate)
                        if cleaned:
                            results[key] = cleaned
        return results

    def extract_candidate_names(self, texts: List[str]) -> Dict[str, Optional[str]]:
        candidates = {"first_name": None, "surname": None}
        standalone_names = []

        lines = []
        for text in texts:
            lines.extend([line.strip() for line in text.splitlines() if line.strip()])

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for label-only lines
            label_match = None
            for key, pattern in self.labels.items():
                if pattern.match(line):
                    label_match = key
                    break

            if label_match:
                # Try next line as candidate for this label
                if i + 1 < len(lines) and self.is_valid_name_line(lines[i + 1]):
                    candidates[label_match] = self.clean_name(lines[i + 1])
                    i += 2
                    continue
                else:
                    i += 1
                    continue

            # Check for label+name on same line
            found = self.parse_label_and_name(line)
            for key in found:
                if found[key]:
                    candidates[key] = found[key]

            # Handle comma-separated names here
            if ',' in line and self.is_valid_name_line(line):
                parts = line.split(',')
                surname_raw = parts[0].strip()
                firstname_raw = parts[1].strip()
                surname_clean = self.clean_name(surname_raw)
                firstname_clean = self.clean_name(firstname_raw)
                if surname_clean:
                    candidates['surname'] = surname_clean
                if firstname_clean:
                    candidates['first_name'] = firstname_clean
                # Also add to standalone names combined
                combined_name = f"{firstname_clean} {surname_clean}".strip()
                standalone_names.append(combined_name)
                i += 1
                continue

            # Add to standalone names if valid and not a label
            if self.is_valid_name_line(line) and not any(p.match(line) for p in self.labels.values()):
                standalone_names.append(self.clean_name(line))

            i += 1

        # Remove labels from standalone names
        standalone_names = [
            n for n in standalone_names
            if n.lower() not in {"christian name", "first name", "surname", "first names"}
        ]

        first = candidates.get("first_name")
        last = candidates.get("surname")

        # Try to fill missing first/last from standalone names
        if not first and standalone_names:
            possible_firsts = [n for n in standalone_names if last is None or last.lower() not in n.lower()]
            if possible_firsts:
                first = max(possible_firsts, key=len)

        if not last and standalone_names:
            possible_lasts = [n for n in standalone_names if first is None or first.lower() not in n.lower()]
            if possible_lasts:
                last = max(possible_lasts, key=len)

        full_name = None
        if first and last:
            combined1 = f"{first} {last}"
            combined2 = f"{last} {first}"
            if combined1 in standalone_names or combined1.lower() in " ".join(standalone_names).lower():
                full_name = combined1
            elif combined2 in standalone_names or combined2.lower() in " ".join(standalone_names).lower():
                full_name = combined2
            else:
                full_name = combined1
        elif first:
            full_name = first
        elif last:
            full_name = last

        if not full_name and standalone_names:
            full_name = max(standalone_names, key=len)

        return {
            "full_name": full_name,
            "first_name": first,
            "surname": last,
            "standalone_names": standalone_names,
        }



class StudentInfoExtractor:
    def __init__(self):
        # bounding boxes: (x0, y0, x1, y1)
        self.bboxes_page1 = [
            (288, 508, 716, 510),
            (290, 526, 703, 530),
        ]
        self.bboxes_page2 = [
            (15, 496, 168, 498),
            (28, 519, 171, 522),
        ]
        self.name_extractor = StudentNameExtractor()

    def extract_name_from_pdf(self, pdf_path: str) -> Optional[str]:
        doc = fitz.open(pdf_path)
        texts = []

        if len(doc) > 0:
            page1 = doc[0]
            for i, bbox in enumerate(self.bboxes_page1):
                text = self.name_extractor.extract_text_from_bbox(page1, bbox)
                print(f"Page 1 bbox {i} raw text: '{text}'")
                texts.append(text)

        if len(doc) > 1:
            page2 = doc[1]
            for i, bbox in enumerate(self.bboxes_page2):
                text = self.name_extractor.extract_text_from_bbox(page2, bbox)
                print(f"Page 2 bbox {i} raw text: '{text}'")
                texts.append(text)

        doc.close()

        result = self.name_extractor.extract_candidate_names(texts)
        print(f"Standalone valid names found: {result['standalone_names']}")
        print(f"First name detected: {result['first_name']}")
        print(f"Surname detected: {result['surname']}")
        print(f"Full name extracted: {result['full_name']}")
        return result["full_name"]


if __name__ == "__main__":
    import os

    folder_path = os.path.abspath(os.path.dirname(__file__))
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    extractor = StudentInfoExtractor()

    if not pdf_files:
        print("No PDF files found in the folder.")
    else:
        for pdf_file in pdf_files:
            full_path = os.path.join(folder_path, pdf_file)
            print(f"Processing file: {pdf_file}")
            student_name = extractor.extract_name_from_pdf(full_path)
            print(f"  Student Name: {student_name}\n")
