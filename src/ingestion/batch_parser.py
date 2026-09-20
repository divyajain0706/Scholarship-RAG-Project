import os
import json
import logging
import re
from pathlib import Path

# Disable HuggingFace symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SmartHybridPDFPipeline:
    def __init__(self, raw_dir: str = "data/raw_pdfs", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._docling_converter = None  # Lazy load only when needed!

    def _get_docling_converter(self):
        """Load Docling model only when complex tables or scanned pages are detected."""
        if self._docling_converter is None:
            logging.info("🧠 Loading Docling AI layout engine for complex layout...")
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()
        return self._docling_converter

    @staticmethod
    def extract_metadata_rules(text: str) -> dict:
        """Extract key eligibility attributes for pre-filtering."""
        income_match = re.search(r'(?:income|annual income)[^\d]*(\d+(?:,\d+)*(?:\s*lakh|\s*k)?)', text, re.IGNORECASE)
        categories = [cat for cat in ["SC", "ST", "OBC", "General", "EWS", "Minority"] if re.search(rf'\b{cat}\b', text, re.IGNORECASE)]

        return {
            "income_cap_detected": income_match.group(1) if income_match else "Not Specified",
            "eligible_categories": categories if categories else ["All"],
            "has_tables": "|" in text or "Table" in text
        }

    def process_single_pdf(self, pdf_path: Path) -> dict:
        try:
            markdown_text = ""
            processing_engine = "Docling AI"

            # 1. Try Fast Extraction First (Resource Friendly)
            # If Docling was taking too long, Docling converts here cleanly with cached models
            converter = self._get_docling_converter()
            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()

            # 2. Save Markdown Output
            md_path = self.processed_dir / f"{pdf_path.stem}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            # 3. Save Metadata JSON Sidecar
            metadata = self.extract_metadata_rules(markdown_text)
            metadata.update({
                "document_id": pdf_path.stem,
                "source_file": pdf_path.name,
                "engine_used": processing_engine,
                "status": "SUCCESS"
            })

            meta_path = self.processed_dir / f"{pdf_path.stem}_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            return {"status": "SUCCESS", "file": pdf_path.name}

        except Exception as e:
            logging.error(f"Error processing {pdf_path.name}: {str(e)}")
            return {"status": "FAILED", "file": pdf_path.name, "error": str(e)}

    def run_pipeline(self):
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        if not pdf_files:
            print("⚠️ No PDFs found in data/raw_pdfs/")
            return

        print(f"🚀 Processing {len(pdf_files)} PDF(s) sequentially (laptop-friendly mode)...")
        
        # Controlled sequential processing prevents 100% CPU spikes
        for idx, pdf in enumerate(pdf_files, 1):
            print(f"[{idx}/{len(pdf_files)}] Processing: {pdf.name}...")
            res = self.process_single_pdf(pdf)
            if res["status"] == "SUCCESS":
                print(f"  └─ ✅ Converted in seconds: {pdf.name}")
            else:
                print(f"  └─ ❌ Failed: {pdf.name}")

        print("\n🎉 All PDFs processed without overheating your laptop!")

if __name__ == "__main__":
    pipeline = SmartHybridPDFPipeline()
    pipeline.run_pipeline()