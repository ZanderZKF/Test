import PyPDF2
import re

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

def find_ablation_section(text):
    # Look for "Ablation Study" or "Component Analysis"
    matches = []
    # simple keyword search to find the location
    keywords = ["Ablation Study", "ablation study", "Component Analysis", "contributions of different components"]
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for keyword in keywords:
            if keyword in line:
                # return a chunk of text around this line
                start = max(0, i)
                end = min(len(lines), i + 200) # Read next 200 lines
                matches.append("\n".join(lines[start:end]))
    return matches

if __name__ == "__main__":
    pdf_path = "/root/autodl-tmp/Heterogeneous_Experts_and_Hierarchical_Perception_for_Underwater_Salient_Object_Detection (1).pdf"
    print(f"--- Processing {pdf_path} ---")
    content = extract_text_from_pdf(pdf_path)
    
    # Print the full content is too much, let's try to find the ablation study section specifically
    ablation_sections = find_ablation_section(content)
    
    if ablation_sections:
        print("Found Ablation Study sections:")
        for section in ablation_sections:
            print(section)
            print("-" * 50)
    else:
        print("Could not find explicit 'Ablation Study' section. Printing first 2000 chars and Table captions to help locate.")
        print(content[:2000])
        # Try to find tables
        lines = content.split('\n')
        for line in lines:
            if "Table" in line or "TABLE" in line:
                print(line)
