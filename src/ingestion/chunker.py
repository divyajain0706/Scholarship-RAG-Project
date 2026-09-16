import os
import json
import re
from pathlib import Path
from typing import List, Dict

class MarkdownHeaderChunker:
    """
    Structure-aware chunker that splits Markdown files by headers 
    and keeps tables intact for precision RAG retrieval.
    """
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk_markdown(self, md_content: str, source_doc: str) -> List[Dict]:
        chunks = []
        # Split by markdown headers (#, ##, ###)
        sections = re.split(r'\n(?=#{1,3}\s)', md_content)
        
        chunk_id = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Extract header title if present
            header_match = re.match(r'^(#{1,3})\s+(.+)\$', section.split('\n')[0])
            header_title = header_match.group(2) if header_match else "General Section"

            # If section is small enough, store as one chunk
            if len(section) <= self.max_chunk_size:
                chunks.append({
                    "chunk_id": f"{source_doc}_chunk_{chunk_id}",
                    "source_doc": source_doc,
                    "section_title": header_title,
                    "content": section
                })
                chunk_id += 1
            else:
                # Sub-split long text paragraphs while trying to keep tables together
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= self.max_chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "chunk_id": f"{source_doc}_chunk_{chunk_id}",
                                "source_doc": source_doc,
                                "section_title": header_title,
                                "content": current_chunk.strip()
                            })
                            chunk_id += 1
                        current_chunk = para + "\n\n"
                
                if current_chunk.strip():
                    chunks.append({
                        "chunk_id": f"{source_doc}_chunk_{chunk_id}",
                        "source_doc": source_doc,
                        "section_title": header_title,
                        "content": current_chunk.strip()
                    })
                    chunk_id += 1
                    
        return chunks

if __name__ == "__main__":
    chunker = MarkdownHeaderChunker(max_chunk_size=800)
    
    # Process all converted .md files in data/processed
    processed_dir = Path("data/processed")
    md_files = list(processed_dir.glob("*.md"))
    
    if md_files:
        all_chunks = []
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            file_chunks = chunker.chunk_markdown(content, source_doc=md_file.stem)
            all_chunks.extend(file_chunks)
            
        output_chunks_file = processed_dir / "all_chunks.json"
        with open(output_chunks_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=4)
            
        print(f"✅ Created {len(all_chunks)} structure-aware chunks across {len(md_files)} document(s)!")
    else:
        print("⚠️ No .md files found in data/processed/. Run batch_parser.py first!")