
import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_paths = [
        "/root/autodl-tmp/(44Semi-UIR)Huang_Contrastive_Semi-Supervised_Learning_for_Underwater_Image_Restoration_via_Reliable_Bank_CVPR_2023_paper.pdf",
        "/root/autodl-tmp/10321-supp.pdf"
    ]
    
    for path in pdf_paths:
        print(f"--- Processing {path} ---")
        content = extract_text_from_pdf(path)
        # Print first 2000 characters to get the gist without overwhelming context
        print(content[:5000]) 
        print("\n" + "="*50 + "\n")
