from llama_index.readers.smart_pdf_loader import SmartPDFLoader

pdf_loader = SmartPDFLoader(llmsherpa_api_url="https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all")
documents = pdf_loader.load_data("path_or_url_to_pdf.pdf")
