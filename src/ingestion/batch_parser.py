import os
import json
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter

# Disable HuggingFace symlinks warning for Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Configure logging to track progress and errors
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BatchScholarshipParser:
    def __init__(self, raw_pdf_dir: str, output_dir: str):
        self.raw_pdf_dir = Path(raw_pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = DocumentConverter()

    def process_all_pdfs(self):
        pdf_files = list(self.raw_pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logging.warning(f"No PDFs found in {self.raw_pdf_dir}. Add some PDFs to process!")
            return

        logging.info(f"🚀 Starting batch conversion of {len(pdf_files)} PDF(s)...")

        for pdf_path in pdf_files:
            try:
                logging.info(f"Processing: {pdf_path.name}")
                result = self.converter.convert(str(pdf_path))
                markdown_text = result.document.export_to_markdown()

                # 1. Save converted Markdown
                md_filename = self.output_dir / f"{pdf_path.stem}.md"
                with open(md_filename, "w", encoding="utf-8") as f:
                    f.write(markdown_text)

                # 2. Save initial Metadata JSON sidecar
                metadata = {
                    "source_file": pdf_path.name,
                    "document_id": pdf_path.stem,
                    "status": "processed",
                    "processed_format": "markdown"
                }
                meta_filename = self.output_dir / f"{pdf_path.stem}_meta.json"
                with open(meta_filename, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)

                logging.info(f"✅ Successfully converted: {pdf_path.name}")

            except Exception as e:
                logging.error(f"❌ Failed to process {pdf_path.name}: {str(e)}")

if __name__ == "__main__":
    RAW_DIR = "data/raw_pdfs"
    PROCESSED_DIR = "data/processed"
    
    parser = BatchScholarshipParser(raw_pdf_dir=RAW_DIR, output_dir=PROCESSED_DIR)
    parser.process_all_pdfs()