import pymupdf

pdf_path = "documents/HDFC_Bank_Research_Report.pdf"

doc = pymupdf.open(pdf_path)

# Extract all text
full_text = ""

for page in doc:
    full_text += page.get_text() + "\n"

doc.close()

# Simple chunking
chunk_size = 1500
overlap = 200

chunks = []

start = 0

while start < len(full_text):
    end = start + chunk_size

    chunk = full_text[start:end]

    chunks.append(chunk)

    start = end - overlap

print("Total characters:", len(full_text))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)