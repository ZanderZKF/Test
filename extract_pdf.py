
import pypdf
import sys

pdf_path = "/root/autodl-tmp/IGMamba__Illumination_Gradient_Decomposition_with_Mamba_Propagation_for_Underwater_Saliency_Detection (5).pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        full_text += f"\n--- Page {i+1} ---\n" + text
        
    # Search for LIQAM
    keyword = "LIQAM"
    if keyword in full_text:
        print(f"\nFound '{keyword}' in text!")
        # Print context
        start_index = full_text.find(keyword)
        # Print a large chunk around it to capture the definition
        print(full_text[max(0, start_index - 1000) : min(len(full_text), start_index + 2000)])
    else:
        print(f"\n'{keyword}' not found in extracted text.")
        print("Here is the beginning of the text to verify extraction works:")
        print(full_text[:1000])

except Exception as e:
    print(f"Error reading PDF: {e}")
