from pypdf import PdfReader

# def text_extractor_pdf(file_path):
#     pdf_file = PdfReader(file_path)
#     pdf_text= ''
#     for pages in pdf_file.pages:
#         text_only += pages.extract_text()
#         if text_only:
#             pdf_text += text_only
#     return pdf_text

def text_extractor_pdf(file_uploaded):
    if file_uploaded is None:
        return ""  # no file uploaded yet
    
    pdf_reader = PdfReader(file_uploaded)  # file_uploaded is already a file-like object
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text