from pdf2image import convert_from_path
import pytesseract


def extract_text_from_scanned_pdf(pdf_path):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path)

    # Extract text from each image
    all_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        all_text += text + "\n"

    return all_text


pdf_path = 'data/Sample_Scanned_PDF.pdf'
extracted_text = extract_text_from_scanned_pdf(pdf_path)
print(extracted_text)
