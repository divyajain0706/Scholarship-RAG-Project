# 🎓 Scholarship & Government Scheme Eligibility RAG Navigator

A production-grade, multi-lingual Retrieval-Augmented Generation (RAG) system designed to deliver zero-hallucination eligibility answers from complex Indian government scholarship circulars and PDF documents.

---

## 🌟 Key Features & Innovations

- **Layout-Aware PDF Ingestion:** Uses **Docling** and computer vision models to accurately parse complex multi-column PDF circulars and convert eligibility tables into Markdown.
- **Structure-Aware Chunking:** Smart chunking pipeline that preserves markdown table integrity and header hierarchies.
- **Hybrid Search Architecture (Phase 2):** Combines **BM25** keyword search with **bge-m3** multi-lingual vector embeddings for high-precision retrieval across English, Hindi, and regional languages.
- **Relevance Guardrail (Phase 3):** Uses Small Language Models (SLMs) as a double-checker to eliminate off-topic retrieved contexts before response generation.

---

## 📁 System Architecture & Directory Layout

```text
scholarship-rag/
├── data/
│   ├── raw_pdfs/           # Input government scheme circulars
│   └── processed/          # Parsed Markdown (.md) and JSON chunks
├── src/
│   ├── ingestion/
│   │   ├── batch_parser.py  # Batch PDF layout-aware converter
│   │   └── chunker.py       # Header and table-preserving chunker
│   ├── retrieval/          # Search & Embedding logic (Phase 2)
│   └── generation/         # LLM & Guardrails (Phase 3)
├── app.py                  # Streamlit web interface
├── requirements.txt        # Production dependencies
└── README.md