import PyPDF2
import re

def extract_table_data(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Table I is likely on page 6, 7, 8, or 9 based on previous output
            # Previous output showed "TABLE I" on Page 9 context (which was actually page 7 in the file structure? No, it printed "Page 9")
            # Let's check pages 6 to 10
            for i in range(6, 11):
                print(f"--- Page {i+1} ---")
                text = reader.pages[i].extract_text()
                lines = text.split('\n')
                for line in lines:
                    # Look for lines that might contain the results
                    # HEHP or Ours, and a sequence of floating point numbers
                    if "HEHP" in line or "Ours" in line or "0." in line:
                         # Simple filter to find lines with multiple metrics
                         numbers = re.findall(r"0\.\d+", line)
                         if len(numbers) >= 3:
                             print(f"Possible Data Line: {line}")
                print("\n")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    pdf_path = "/root/autodl-tmp/Heterogeneous_Experts_and_Hierarchical_Perception_for_Underwater_Salient_Object_Detection (1).pdf"
    extract_table_data(pdf_path)
