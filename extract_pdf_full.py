
import pypdf
import sys

pdf_path = "/root/autodl-tmp/IGMamba__Illumination_Gradient_Decomposition_with_Mamba_Propagation_for_Underwater_Saliency_Detection (5).pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        full_text += f"\n--- Page {i+1} ---\n" + text
    
    print(full_text)

except Exception as e:
    print(f"Error reading PDF: {e}")
