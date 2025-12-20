# Sibyl


[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

Hybrid document-to-markdown Python library with intelligent extraction routing.

Sibyl automatically selects the best extraction method for each document—using native text extraction for digital PDFs and vision-language models for scanned documents or images.

## Features

- **Intelligent Routing**: Automatically detects native vs scanned content and routes to the optimal extractor
- **Multiple PDF Engines**: Choose between Docling and MarkItDown with automatic fallback
- **Multiple Backends**: Support for local (Ollama, LMStudio) and cloud (OpenAI, Anthropic, Gemini) VLM providers
- **Hybrid Extraction**: Handle documents with mixed native and scanned pages
- **RAG-Ready Chunking**: Built-in page-based, size-based, and section-based chunking
- **Table Extraction**: Preserves tables as markdown with optional split-table merging
- **Image Handling**: Extract embedded images with optional VLM descriptions
- **Batch Processing**: Process multiple documents in parallel

## Installation

```bash
pip install git+https://github.com/M-Chimiste/sibyl.git
```

Or install from source for development:

```bash
git clone https://github.com/M-Chimiste/sibyl.git
cd sibyl
pip install -e ".[dev]"
```

### Dependencies

Sibyl uses:
- [Docling](https://github.com/DS4SD/docling) for native PDF extraction
- [MarkItDown](https://github.com/microsoft/markitdown) for Office formats (DOCX, PPTX, XLSX)
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF rendering

## Quick Start

### Basic Usage (Native PDFs)

```python
from sibyl import Sibyl

# Initialize without OCR backend (native extraction only)
sb = Sibyl()

# Process a document
result = sb.process("document.pdf")

# Access the markdown content
print(result.markdown)

# Access metadata
print(f"Pages: {result.metadata.page_count}")
print(f"Tables found: {len(result.tables)}")
```

### With OCR Support (Ollama)

```python
from sibyl import Sibyl
from sibyl.backends import OllamaBackend

# Initialize with Ollama backend
backend = OllamaBackend(
    model="deepseek-ocr",  # or "llava", "minicpm-v", etc.
    base_url="http://localhost:11434",  # default
)

sb = Sibyl(ocr_backend=backend)

# Process scanned documents or images
result = sb.process("scanned_document.pdf")
print(result.markdown)
```

### With LMStudio

```python
from sibyl import Sibyl
from sibyl.backends import LMStudioBackend

backend = LMStudioBackend(
    model="deepseek-ocr",
    base_url="http://localhost:1234",  # default LMStudio port
)

sb = Sibyl(ocr_backend=backend)
result = sb.process("document.pdf")
```

### Cloud Providers

```python
from sibyl import Sibyl
from sibyl.backends import OpenAIBackend, AnthropicBackend, GeminiBackend

# OpenAI
backend = OpenAIBackend(
    model="gpt-4o",  # default
    api_key="...",   # or set OPENAI_API_KEY env var
)

# Anthropic
backend = AnthropicBackend(
    model="claude-sonnet-4-20250514",  # default
    api_key="...",  # or set ANTHROPIC_API_KEY env var
)

# Google Gemini
backend = GeminiBackend(
    model="gemini-2.0-flash",  # default
    api_key="...",  # or set GOOGLE_API_KEY env var
)

sb = Sibyl(ocr_backend=backend)
```

## Processing Options

```python
result = sb.process(
    "document.pdf",
    extract_tables=True,      # Extract tables (default: True)
    extract_images=True,      # Extract embedded images (default: True)
    ocr_images=True,          # OCR text within images (default: True)
    describe_images=False,    # Use VLM to describe images (default: False)
    merge_tables=False,       # Merge horizontally-split tables (default: False)
    check_quality=False,      # Check text quality and re-extract with OCR (default: False)
    quality_threshold=0.7,    # Minimum quality score for text (default: 0.7)
    pages=[1, 2, 3],          # Process specific pages (default: all)
)
```

### Image Descriptions

When `describe_images=True`, Sibyl replaces `<!-- image -->` placeholders with VLM-generated descriptions:

```python
result = sb.process("document.pdf", describe_images=True)
# Markdown will contain: <!-- image: A bar chart showing quarterly sales... -->
```

### Merging Split Tables

PDFs often display tables in multiple columns to save space. For example, a reference table might appear as:

```
| Level | XP     | Level | XP      |
|-------|--------|-------|---------|
| 1     | 100    | 6     | 14000   |
| 2     | 300    | 7     | 23000   |
```

When `merge_tables=True`, Sibyl automatically detects these horizontally-split tables and merges them into a single table:

```python
result = sb.process("document.pdf", merge_tables=True)

# The table is now merged into a single 2-column table:
# | Level | XP     |
# |-------|--------|
# | 1     | 100    |
# | 2     | 300    |
# | 6     | 14000  |
# | 7     | 23000  |

# Check how many tables were merged
print(f"Tables merged: {result.stats.tables_merged}")
```

This is useful for converting documents to database-friendly formats.

### Quality Checking and Hybrid Extraction

Native PDF extraction can sometimes produce garbled text due to encoding issues, embedded fonts, or corrupted content. When `check_quality=True`, Sibyl analyzes extracted text for common issues and automatically re-extracts problem pages using OCR:

```python
result = sb.process(
    "document.pdf",
    check_quality=True,       # Enable quality checking
    quality_threshold=0.7,    # Pages below this score use OCR (0.0-1.0)
)
```

**Quality checks detect:**
- Replacement characters (�) indicating encoding failures
- Block/box drawing characters (█) from missing fonts
- Unescaped HTML entities (`&amp;` instead of `&`)
- Excessive whitespace indicating missing text
- Control characters and other anomalies

**Text cleaning is always applied:**
- HTML entity decoding (`&amp;` → `&`)
- Unicode normalization
- Control character removal
- Whitespace normalization

This hybrid approach gives you the best of both worlds: fast native extraction for clean pages and OCR fallback for problematic content.

### Progress Callbacks

Monitor processing progress with the `on_progress` callback:

```python
def on_progress(stage: str, current: int, total: int):
    """Progress callback for document processing.

    Args:
        stage: Current processing stage
            - "extracting": Native text extraction
            - "ocr": Vision OCR processing
            - "describing_images": VLM image descriptions
        current: Current item being processed
        total: Total items to process
    """
    print(f"[{stage}] {current}/{total}")

result = sb.process(
    "document.pdf",
    describe_images=True,
    on_progress=on_progress,
)
```

Example output:
```
[extracting] 0/10
[extracting] 10/10
[ocr] 0/5
[ocr] 1/5
[ocr] 2/5
...
[describing_images] 0/3
[describing_images] 1/3
...
```

## Chunking for RAG

Sibyl provides built-in chunking methods optimized for retrieval-augmented generation:

```python
# Process document
result = sb.process("document.pdf")

# Page-based chunking (default)
chunks = sb.chunk(result, method="page")

# Size-based chunking
chunks = sb.chunk(result, method="size", chunk_size=512, chunk_overlap=50)

# Section-based chunking (splits on headings)
chunks = sb.chunk(result, method="section")

# Each chunk contains text and metadata
for chunk in chunks:
    print(f"Page {chunk.page_number}: {chunk.text[:100]}...")
```

## Batch Processing

Process multiple documents in parallel:

```python
files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

def on_progress(completed, total, current_file):
    print(f"[{completed}/{total}] Processed {current_file.name}")

results = sb.process_batch(
    files,
    workers=4,
    on_progress=on_progress,
)
```

## Document Analysis

Analyze a document without extracting content:

```python
analysis = sb.analyze("document.pdf")

print(f"File type: {analysis.file_type}")
print(f"Content type: {analysis.content_type}")  # native, scanned, or hybrid
print(f"Recommended method: {analysis.primary_method}")
print(f"Page count: {analysis.page_count}")
```

## Examples

### Complete RAG Pipeline

```python
"""Build a complete RAG pipeline with Sibyl."""
from pathlib import Path
from sibyl import Sibyl
from sibyl.backends import OllamaBackend

# Initialize with local OCR
backend = OllamaBackend(model="deepseek-ocr")
sb = Sibyl(ocr_backend=backend, pdf_engine="auto")

def process_documents_for_rag(folder: Path) -> list[dict]:
    """Process all documents in a folder for RAG indexing."""
    documents = []

    # Find all supported files
    files = list(folder.glob("*.pdf")) + list(folder.glob("*.docx"))

    # Process with progress tracking
    def on_batch_progress(completed, total, current_file):
        print(f"[{completed}/{total}] {current_file.name}")

    results = sb.process_batch(files, workers=4, on_progress=on_batch_progress)

    # Chunk and prepare for indexing
    for result, file_path in zip(results, files):
        chunks = sb.chunk(result, method="page")

        for chunk in chunks:
            documents.append({
                "text": chunk.text,
                "metadata": {
                    "source": str(file_path),
                    "page": chunk.page_number,
                    "title": result.metadata.title,
                    "method": chunk.metadata.get("extraction_method"),
                },
            })

    return documents

# Usage
docs = process_documents_for_rag(Path("./documents"))
print(f"Indexed {len(docs)} chunks from {len(set(d['metadata']['source'] for d in docs))} documents")
```

### Processing Scanned Documents with Progress

```python
"""Process a scanned PDF with real-time progress updates."""
from sibyl import Sibyl
from sibyl.backends import OllamaBackend

backend = OllamaBackend(model="deepseek-ocr", timeout=300.0)
sb = Sibyl(ocr_backend=backend)

def process_with_progress(file_path: str):
    """Process a document with detailed progress reporting."""

    def on_progress(stage: str, current: int, total: int):
        if stage == "extracting":
            print(f"Extracting text... {current}/{total} pages")
        elif stage == "ocr":
            print(f"OCR processing... {current}/{total} pages")
        elif stage == "describing_images":
            print(f"Describing images... {current}/{total}")

    # First, analyze to understand what we're dealing with
    analysis = sb.analyze(file_path)
    print(f"Document type: {analysis.content_type.value}")
    print(f"Pages: {analysis.page_count}")
    print(f"Primary method: {analysis.primary_method.value}")

    # Process with progress
    result = sb.process(
        file_path,
        describe_images=True,
        on_progress=on_progress,
    )

    # Summary
    print(f"\nProcessing complete!")
    print(f"Time: {result.stats.total_time_seconds:.2f}s")
    print(f"Native pages: {result.stats.native_pages}")
    print(f"OCR pages: {result.stats.ocr_pages}")
    print(f"Tables found: {len(result.tables)}")

    return result

result = process_with_progress("scanned_report.pdf")
```

### Hybrid Document Processing

```python
"""Handle documents with mixed native and scanned pages."""
from sibyl import Sibyl
from sibyl.backends import AnthropicBackend

# Use Claude for high-quality OCR
backend = AnthropicBackend()  # Uses ANTHROPIC_API_KEY env var
sb = Sibyl(ocr_backend=backend)

def process_hybrid_document(file_path: str):
    """Process a document that may have both native and scanned pages."""

    # Analyze first to understand the document
    analysis = sb.analyze(file_path)

    if analysis.content_type.value == "hybrid":
        print("Detected hybrid document with mixed content types")

        # Show per-page analysis
        for page in analysis.pages:
            print(f"  Page {page.page_number}: {page.content_type.value} -> {page.recommended_method.value}")

    # Process - Sibyl automatically uses the right method per page
    result = sb.process(file_path)

    # Show which methods were used
    for page in result.pages:
        print(f"Page {page.page_number}: extracted with {page.extraction_method}")

    return result

result = process_hybrid_document("mixed_document.pdf")
```

### Office Document Conversion

```python
"""Convert Office documents to markdown."""
from pathlib import Path
from sibyl import Sibyl

sb = Sibyl(pdf_engine="markitdown")  # MarkItDown handles Office formats

def convert_office_docs(input_folder: Path, output_folder: Path):
    """Convert all Office documents to markdown files."""
    output_folder.mkdir(exist_ok=True)

    # Supported Office formats
    patterns = ["*.docx", "*.pptx", "*.xlsx", "*.html"]
    files = []
    for pattern in patterns:
        files.extend(input_folder.glob(pattern))

    for file_path in files:
        print(f"Converting {file_path.name}...")

        result = sb.process(file_path)

        # Save as markdown
        output_path = output_folder / f"{file_path.stem}.md"
        output_path.write_text(result.markdown)

        print(f"  -> {output_path.name} ({len(result.markdown)} chars)")

convert_office_docs(Path("./input"), Path("./output"))
```

### Extracting Tables from PDFs

```python
"""Extract and save tables from PDF documents."""
from sibyl import Sibyl

sb = Sibyl()

def extract_tables(file_path: str) -> list[str]:
    """Extract all tables from a PDF as markdown."""
    result = sb.process(file_path, extract_tables=True)

    tables = []
    for table in result.tables:
        print(f"Found table on page {table.page_number}: {table.rows}x{table.columns}")
        tables.append(table.markdown)

    return tables

# Get all tables as markdown strings
tables = extract_tables("financial_report.pdf")
for i, table_md in enumerate(tables):
    print(f"\n--- Table {i+1} ---")
    print(table_md)
```

### Size-Based Chunking for Embeddings

```python
"""Chunk documents for embedding models with token limits."""
from sibyl import Sibyl

sb = Sibyl()

def chunk_for_embeddings(file_path: str, max_chars: int = 1000, overlap: int = 100):
    """Chunk a document for embedding generation."""
    result = sb.process(file_path)

    # Use size-based chunking for consistent chunk sizes
    chunks = sb.chunk(
        result,
        method="size",
        chunk_size=max_chars,
        chunk_overlap=overlap,
    )

    print(f"Created {len(chunks)} chunks from {result.metadata.page_count} pages")

    # Prepare for embedding API
    texts_for_embedding = []
    for chunk in chunks:
        texts_for_embedding.append({
            "text": chunk.text,
            "page": chunk.page_number,
            "index": chunk.chunk_index,
        })

    return texts_for_embedding

chunks = chunk_for_embeddings("long_document.pdf", max_chars=500)
```

### Section-Based Chunking for Q&A

```python
"""Chunk by document sections for better Q&A retrieval."""
from sibyl import Sibyl

sb = Sibyl()

def chunk_by_sections(file_path: str):
    """Chunk document by markdown headings for Q&A systems."""
    result = sb.process(file_path)

    # Section-based chunking preserves logical document structure
    chunks = sb.chunk(result, method="section")

    for chunk in chunks:
        heading = chunk.metadata.get("section_heading", "No heading")
        preview = chunk.text[:100].replace("\n", " ")
        print(f"[{heading}] {preview}...")

    return chunks

sections = chunk_by_sections("user_manual.pdf")
```

## Result Structure

```python
result = sb.process("document.pdf")

# Full markdown content
result.markdown: str

# Per-page results
result.pages: list[PageResult]
for page in result.pages:
    page.page_number: int
    page.content: str
    page.extraction_method: str  # "docling", "vision_ocr", "markitdown"
    page.tables: list[TableResult]
    page.images: list[ImageResult]

# All tables
result.tables: list[TableResult]
for table in result.tables:
    table.markdown: str
    table.page_number: int
    table.rows: int
    table.columns: int

# Document metadata
result.metadata.title: str | None
result.metadata.author: str | None
result.metadata.page_count: int
result.metadata.file_type: str

# Processing stats
result.stats.total_time_seconds: float
result.stats.methods_used: list[str]
result.stats.pages_processed: int
result.stats.ocr_pages: int
result.stats.native_pages: int
result.stats.tables_merged: int  # Number of split tables merged (when merge_tables=True)
result.stats.quality_reextracted_pages: int  # Pages re-extracted due to quality issues
```

## Supported Formats

| Format | Extractor | Notes |
|--------|-----------|-------|
| PDF (native) | Docling/MarkItDown | Digital PDFs with text layer |
| PDF (scanned) | Vision OCR | Requires OCR backend |
| **Microsoft Office** | | |
| DOCX, DOC | MarkItDown | Word documents |
| PPTX, PPT | MarkItDown | PowerPoint presentations |
| XLSX, XLS | MarkItDown | Excel spreadsheets |
| **Web & Data** | | |
| HTML, HTM | MarkItDown | Web pages |
| CSV | MarkItDown | Comma-separated values |
| JSON | MarkItDown | JSON data files |
| XML | MarkItDown | XML documents |
| **Other Documents** | | |
| RTF | MarkItDown | Rich text format |
| EPUB | MarkItDown | E-books |
| **Images** | | |
| PNG, JPG, TIFF, BMP, GIF | Vision OCR | Requires OCR backend |

## Configuration

### PDF Engine Selection

Sibyl supports two PDF extraction engines with automatic fallback:

```python
from sibyl import Sibyl

# Use Docling (default) - better structure preservation
sb = Sibyl(pdf_engine="docling")

# Use MarkItDown - faster, good for simple documents
sb = Sibyl(pdf_engine="markitdown")

# Auto mode - tries docling first, falls back to markitdown, then OCR
sb = Sibyl(pdf_engine="auto", ocr_backend=backend)
```

**Engine comparison:**

| Engine | Strengths | Best for |
|--------|-----------|----------|
| Docling | Structure preservation, table detection | Complex documents, academic papers |
| MarkItDown | Speed, broad format support | Simple PDFs, batch processing |

**Fallback behavior:** If the primary engine fails or produces empty results, Sibyl automatically tries the alternate engine before falling back to OCR (if an OCR backend is configured).

### OCR Threshold

Control when OCR is triggered for PDF pages:

```python
# Lower threshold = more likely to use OCR
sb = Sibyl(ocr_backend=backend, ocr_threshold=0.1)  # default

# Override per-request
result = sb.process("document.pdf", ocr_threshold=0.05)
```

### Custom Timeout

For OCR backends processing large images:

```python
backend = OllamaBackend(
    model="deepseek-ocr",
    timeout=300.0,  # 5 minutes (default: 120s)
)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=sibyl --cov-report=term-missing

# Type checking
mypy sibyl

# Linting
ruff check sibyl
```

## License

Apache 2.0
