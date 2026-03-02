import PyPDF2
import os

def extract_title_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if len(reader.pages) > 0:
                text = reader.pages[0].extract_text()
                return text[:1000] # Return the first 1000 characters
            return "Empty PDF"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_paths = [
        "/root/autodl-tmp/10321-supp.pdf",
        "/root/autodl-tmp/2107.01779v2 (2).pdf",
        "/root/autodl-tmp/2410.02035v1.pdf",
        "/root/autodl-tmp/2411.17473v2.pdf"
    ]
    
    for path in pdf_paths:
        if os.path.exists(path):
            print(f"--- Processing {path} ---")
            content = extract_title_from_pdf(path)
            print(content) 
            print("\n" + "="*50 + "\n")
        else:
            print(f"File not found: {path}")
