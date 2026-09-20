import os
import json
from pathlib import Path
import streamlit as st

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

st.set_page_config(page_title="Scholarship RAG - Production Admin Dashboard", layout="wide")

st.title("⚙️ Scholarship RAG: Phase 1 Admin & Pipeline Dashboard")

tab1, tab2, tab3 = st.tabs(["📤 Batch PDF Ingestion", "🔍 Document & Metadata Inspector", "🧩 Chunk Quality Auditor"])

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw_pdfs")
RAW_DIR.mkdir(parents=True, exist_ok=True)

with tab1:
    st.header("Batch Ingestion Pipeline")
    uploaded_files = st.file_uploader("Upload Government Circular PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = RAW_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} PDF(s) to `data/raw_pdfs/`")

    if st.button("🚀 Run Batch Ingestion Pipeline"):
        with st.spinner("Processing documents..."):
            from src.ingestion.batch_parser import SmartHybridPDFPipeline
            pipeline = SmartHybridPDFPipeline()
            pipeline.run_pipeline()
        st.success("Batch Ingestion Complete!")

with tab2:
    st.header("Metadata & Document Inspector")
    md_files = list(PROCESSED_DIR.glob("*.md"))
    if md_files:
        selected_doc = st.selectbox("Select Processed Document", [f.stem for f in md_files])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extracted Markdown Text")
            with open(PROCESSED_DIR / f"{selected_doc}.md", "r", encoding="utf-8") as f:
                st.text_area("Markdown Output", f.read(), height=400)
                
        with col2:
            st.subheader("Extracted Metadata Sidecar (.json)")
            meta_path = PROCESSED_DIR / f"{selected_doc}_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    st.json(json.load(f))
            else:
                st.warning("No metadata JSON found.")
    else:
        st.info("No processed documents found. Upload PDFs and run ingestion first.")

with tab3:
    st.header("Chunk Quality Auditor")
    chunks_file = PROCESSED_DIR / "all_chunks.json"
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        st.metric("Total Generated Chunks", len(chunks))
        st.dataframe(chunks)
    else:
        st.info("Run `python src/ingestion/chunker.py` to generate `all_chunks.json`.")