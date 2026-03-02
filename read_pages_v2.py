import PyPDF2

def extract_pages(pdf_path, start_page, end_page):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            num_pages = len(reader.pages)
            
            for i in range(start_page, min(end_page, num_pages)):
                print(f"--- Page {i+1} ---")
                page_text = reader.pages[i].extract_text()
                print(page_text)
                text += page_text + "\n"
            return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = "/root/autodl-tmp/Heterogeneous_Experts_and_Hierarchical_Perception_for_Underwater_Salient_Object_Detection (1).pdf"
    # Read pages 5 to 9
    extract_pages(pdf_path, 4, 9)
