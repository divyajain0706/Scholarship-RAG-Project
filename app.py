import os

# Disable symlinks on Windows to prevent permission errors
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import tempfile
import streamlit as st
from docling.document_converter import DocumentConverter

import os

st.set_page_config(page_title="Scholarship PDF Parser", layout="wide")
st.title("📄 Scholarship PDF to Markdown Converter")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Processing PDF using Docling..."):
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        markdown_text = result.document.export_to_markdown()

    os.remove(tmp_path)

    st.success("PDF successfully converted!")
    st.subheader("Converted Markdown Output:")
    st.markdown(markdown_text)

    st.download_button(
        label="📥 Download Clean Markdown (.md)",
        data=markdown_text,
        file_name="parsed_scholarship.md",
        mime="text/markdown"
    )

