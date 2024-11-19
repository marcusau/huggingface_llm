from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus
from pathlib import Path


image_path = "/home/marcus/Desktop/project/OCR_transformer_practices/data/testing_1/images/BM00004606/Covering_Letter/page_1_500dpi.png"

doc_path = Path(image_path)
converter = DocumentConverter()
result = converter.convert(doc_path)

if result.status != ConversionStatus.SUCCESS:
    print("Document conversion failed:", result.status)
    #return

doc_content = result.document

for item in doc_content.iterate_items():
    if hasattr(item, "label") and hasattr(item, "text"):
        print(f"Type: {item.label}, Content: {item.text[:100]}...")

json_data = doc_content.model_dump_json()