import os
import re
from PIL import Image
from dotenv import load_dotenv

# LangChain and RAG imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

load_dotenv()

# Global variables for RAG
DB_DIR = os.path.join(os.getcwd(), "data", "vector_db")
CORPUS_DIR = os.path.join(os.getcwd(), "data", "corpus")

# Ensure directories exist
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Cache embeddings globally
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # High quality multilingual model or standard light transformer
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings

def index_pdfs():
    """
    Reads PDFs from CORPUS_DIR, splits text, and indexes them into FAISS.
    Returns the vector store or None.
    """
    pdf_files = [f for f in os.listdir(CORPUS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None

    all_docs = []
    for pdf in pdf_files:
        path = os.path.join(CORPUS_DIR, pdf)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf}: {e}")

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(DB_DIR)
    return vector_store

def get_vector_store():
    """
    Loads FAISS index if exists, otherwise tries to index PDFs.
    """
    if os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        try:
            return FAISS.load_local(DB_DIR, get_embeddings(), allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS store: {e}")
            return index_pdfs()
    else:
        return index_pdfs()

def query_rag(query: str, k: int = 4):
    """
    Queries the RAG database and returns source documents and matched text.
    """
    store = get_vector_store()
    if not store:
        return []
    
    results = store.similarity_search_with_score(query, k=k)
    # Filter out very poor matches if needed, but return list of dicts
    docs_info = []
    for doc, score in results:
        docs_info.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Bilinmeyen Kaynak"),
            "page": doc.metadata.get("page", 0) + 1,
            "score": float(score)
        })
    return docs_info

def clean_markdown_to_text(md_text: str) -> str:
    """Removes basic markdown formatting for clean PDF text insertion."""
    text = md_text.replace("**", "").replace("*", "").replace("`", "")
    text = re.sub(r'#+\s*', '', text)
    return text.strip()

def generate_pdf_report(report_markdown: str, filename: str = "ilac_denetim_raporu.pdf") -> str:
    """
    Converts structured Markdown report into a clean, premium ReportLab PDF.
    """
    pdf_path = os.path.join(os.getcwd(), filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#2E5BFF'),
        spaceAfter=15,
        alignment=1 # Center
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        spaceBefore=4,
        spaceAfter=4,
        leading=14
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        leftIndent=15,
        spaceBefore=3,
        spaceAfter=3,
        leading=14
    )

    # Title Banner / Header
    story.append(Paragraph("Pharma-Guard AI Denetim Raporu", title_style))
    story.append(Spacer(1, 10))
    
    # Divider line
    divider = Table([[""]], colWidths=[530])
    divider.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 2, colors.HexColor('#2E5BFF')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0)
    ]))
    story.append(divider)
    story.append(Spacer(1, 15))

    # Process markdown lines
    lines = report_markdown.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Heading 1, 2, 3
        if line.startswith("#"):
            cleaned = clean_markdown_to_text(line)
            story.append(Spacer(1, 5))
            story.append(Paragraph(cleaned, h1_style))
            # Dynamic underline for headings
            h_line = Table([[""]], colWidths=[530])
            h_line.setStyle(TableStyle([
                ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0)
            ]))
            story.append(h_line)
            story.append(Spacer(1, 5))
        # Bullet list
        elif line.startswith("-") or line.startswith("*"):
            cleaned = clean_markdown_to_text(line[1:])
            story.append(Paragraph(f"&bull; {cleaned}", bullet_style))
        elif line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5."):
            cleaned = clean_markdown_to_text(line)
            story.append(Paragraph(cleaned, h1_style))
        else:
            cleaned = clean_markdown_to_text(line)
            story.append(Paragraph(cleaned, body_style))
            
    doc.build(story)
    return pdf_path
