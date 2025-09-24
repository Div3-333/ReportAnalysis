from PyPDF2 import PdfReader

def save_ocr_text(pdf_path, output_path="ocr_output.txt"):
    """
    Extracts the OCR text layer from a PDF (if it exists) 
    and saves it to a plain text file.
    """
    reader = PdfReader(pdf_path)
    all_text = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            all_text.append(f"\n--- Page {i} ---\n{text}")
        else:
            all_text.append(f"\n--- Page {i} ---\n[No text found]")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    print(f"OCR text saved to {output_path}")


if __name__ == "__main__":
    pdf_path = "Aa.pdf"   # replace with your PDF file
    save_ocr_text(pdf_path, "ocr_data.txt")
