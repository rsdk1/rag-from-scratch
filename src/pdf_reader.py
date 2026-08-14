import pymupdf

pdf_path = "documents/HDFC_Bank_Research_Report.pdf"

doc = pymupdf.open(pdf_path)

print("PDF:", pdf_path)
print("Number of pages:", len(doc))

for page_number, page in enumerate(doc):
    text = page.get_text()

    print(f"\n--- Page {page_number + 1} ---")
    print(text[:1000])

doc.close()