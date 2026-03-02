import PyPDF2
import re

def extract_numbers(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i in range(6, 11): # Check pages 7 to 11
                print(f"--- Page {i+1} ---")
                text = reader.pages[i].extract_text()
                lines = text.split('\n')
                for line in lines:
                    # Look for lines with multiple floating point numbers
                    # A typical table row might look like "HEHP 0.920 0.930 0.022 ..."
                    # Or just numbers if the method name is in a separate column/line
                    numbers = re.findall(r"0\.\d{3,4}", line)
                    if len(numbers) >= 3:
                        print(f"Row Candidate: {line}")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    pdf_path = "/root/autodl-tmp/Heterogeneous_Experts_and_Hierarchical_Perception_for_Underwater_Salient_Object_Detection (1).pdf"
    extract_numbers(pdf_path)
