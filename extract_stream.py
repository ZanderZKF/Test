import PyPDF2
import re

def extract_text_stream(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Table I is likely on page 9 (index 8) based on previous "Page 9" output
            text = reader.pages[8].extract_text()
            # Replace newlines with spaces to treat as a stream
            text_stream = text.replace('\n', ' ')
            # Look for HEHP and subsequent numbers
            # Pattern: HEHP followed by some characters and then a sequence of 0.xxxx
            # The metrics are usually S, F_max, F_mean, F_w, E_max, E_mean, MAE (order varies)
            # Let's find "HEHP" and print the next 100 characters
            indices = [m.start() for m in re.finditer('HEHP', text_stream)]
            for idx in indices:
                print(f"Context around HEHP at {idx}:")
                print(text_stream[idx:idx+300])
                print("-" * 20)
            
            # Also look for "Ours"
            indices = [m.start() for m in re.finditer('Ours', text_stream)]
            for idx in indices:
                print(f"Context around Ours at {idx}:")
                print(text_stream[idx:idx+300])
                print("-" * 20)

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    pdf_path = "/root/autodl-tmp/Heterogeneous_Experts_and_Hierarchical_Perception_for_Underwater_Salient_Object_Detection (1).pdf"
    extract_text_stream(pdf_path)
