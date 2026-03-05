import fitz
import json

def analyze_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    
    for page_num in range(min(10, len(doc))):
        page = doc[page_num]
        print(f"\n--- Page {page_num + 1} ---")
        
        # 1. Detect Images
        images = page.get_images()
        if images:
            print(f"Found {len(images)} images on this page.")
            
        # 2. Detect Tables
        tables = page.find_tables()
        if tables:
            print(f"Found {len(tables.tables)} tables on this page.")

        # 3. Detect Headings (Font size > 12 or Bold)
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        if "bold" in s["font"].lower() or s["size"] > 11.5:
                            text = s["text"].strip()
                            if len(text) > 3 and not text.isdigit():
                                print(f"[HEADING?] {text} | Font: {s['font']} | Size: {s['size']:.2f}")

analyze_pdf("/home/cdac/Office Projects/Research-Paper-Summarizer/data/2004.05150v2.pdf")
