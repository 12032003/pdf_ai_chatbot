import os
import uuid
import numpy as np
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
COLLECTION_NAME = "pdf_chatbot"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

class PDFProcessor:
    def extract_text_with_pages(self, pdf_path: str) -> List[Dict]:
        pages = []
        filename = os.path.basename(pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "page_number": page_num,
                            "text": text.strip(),
                            "source": filename
                        })
        except Exception as e:
            print(f"Error reading '{filename}': {e}")
        return pages

class TextChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        all_chunks = []
        for page in pages:
            page_text = page["text"]
            sub_chunks = self._split_text(page_text)
            for idx, chunk_text in enumerate(sub_chunks):
                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "chunk_index": idx
                })
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            return
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, batch_size=32).tolist()
        self.collection.add(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "page_number": chunk["page_number"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            } for chunk in chunks]
        )

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count())
        )
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "text": results["documents"][0][i],
                "page_number": results["metadatas"][0][i]["page_number"],
                "source": results["metadatas"][0][i]["source"],
                "relevance_score": 1 - results["distances"][0][i]
            })
        return retrieved_chunks

    def clear(self):
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

class PDFChatBot:
    SYSTEM_PROMPT = """You are an expert document analyst assistant.
Your job is to answer questions based ONLY on the provided PDF document excerpts.

Rules you must follow:
1. ONLY use information from the provided context, never use outside knowledge
2. ALWAYS cite sources: mention the document name and page number for every claim
3. If the answer is not in the provided context, say: I could not find this information in the uploaded documents
4. Be concise but complete, do not pad your answers
5. If multiple sources support a point, mention all of them
6. Use bullet points for clarity when listing multiple facts

Citation format: (Source: [filename], Page [X])"""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.chat_history = []

    def _build_context_block(self, chunks: List[Dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[EXCERPT {i}]\n"
                f"Document: {chunk['source']}\n"
                f"Page: {chunk['page_number']}\n"
                f"Relevance: {chunk['relevance_score']:.2%}\n"
                f"Content:\n{chunk['text']}"
            )
        return "\n\n" + "-" * 50 + "\n\n".join(context_parts)

    def ask(self, question: str, retrieved_chunks: List[Dict]) -> str:
        context = self._build_context_block(retrieved_chunks)
        user_message = f"""Here are the relevant excerpts from the uploaded PDF documents:

{context}

-------------------------------------------------

Based on the above excerpts, please answer this question:
{question}

Remember to cite the document name and page number for each claim."""

        self.chat_history.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT}
            ] + self.chat_history
        )
        answer = response.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):
        self.chat_history = []

# Initialize components
processor = PDFProcessor()
chunker = TextChunker()
vector_store = VectorStore()
chatbot = PDFChatBot(api_key=GROQ_API_KEY)
uploaded_file_names = []

def process_uploaded_pdfs(files) -> str:
    global uploaded_file_names
    if not files:
        return "No files selected. Please upload at least one PDF."
    
    vector_store.clear()
    chatbot.clear_history()
    uploaded_file_names = []
    all_chunks = []
    
    for file in files:
        pages = processor.extract_text_with_pages(file.name)
        if not pages:
            continue
        chunks = chunker.chunk_pages(pages)
        all_chunks.extend(chunks)
        uploaded_file_names.append(os.path.basename(file.name))
    
    if not all_chunks:
        return "Could not extract text from any uploaded PDFs."
    
    vector_store.add_chunks(all_chunks)
    status = (
        f"Ready. Processed {len(uploaded_file_names)} document(s):\n"
        + "\n".join(f"  - {name}" for name in uploaded_file_names)
        + f"\n\nTotal chunks in database: {vector_store.count()}"
        + f"\nYou can now ask questions in the chat."
    )
    return status

def chat_response(user_message: str, history: list) -> tuple:
    if not user_message.strip():
        return history, "", ""
    
    if not uploaded_file_names:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "Please upload and process a PDF first."})
        return history, "", ""
    
    retrieved = vector_store.search(user_message, top_k=TOP_K_RESULTS)
    
    if not retrieved:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "No relevant content found. Try rephrasing your question."})
        return history, "", ""
    
    answer = chatbot.ask(user_message, retrieved)
    sources_text = format_sources(retrieved)
    
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    
    return history, "", sources_text

def format_sources(chunks: List[Dict]) -> str:
    if not chunks:
        return "No sources retrieved."
    
    lines = ["### Sources Used\n"]
    seen_pages = set()
    
    for i, chunk in enumerate(chunks, 1):
        key = f"{chunk['source']}_p{chunk['page_number']}"
        if key in seen_pages:
            continue
        seen_pages.add(key)
        
        lines.append(
            f"**[{i}] {chunk['source']} - Page {chunk['page_number']}**  \n"
            f"Relevance: {chunk['relevance_score']:.0%}  \n"
            f"Excerpt:  \n"
            f"> {chunk['text'][:200].replace(chr(10), ' ')}...  \n"
        )
    
    return "\n".join(lines)

def clear_chat_history():
    chatbot.clear_history()
    return [], ""

# Create Gradio Interface
with gr.Blocks(title="PDF AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # PDF AI Chatbot
    Upload your PDF documents and ask questions. Get answers with page citations.
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## Upload Documents")
            file_input = gr.File(
                label="Select PDF files (up to 50MB each)",
                file_count="multiple",
                file_types=[".pdf"],
                height=150
            )
            process_btn = gr.Button("Process PDFs", variant="primary", size="lg")
            status_box = gr.Textbox(
                label="Processing Status",
                value="Upload PDFs and click Process PDFs to begin.",
                interactive=False,
                lines=6
            )
            gr.Markdown("---")
            gr.Markdown("""
            **Tips:**
            - Upload multiple PDFs at once
            - Ask follow-up questions naturally
            - Check Sources panel for citations
            - Use Clear Chat to start over
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("## Chat with your Documents")
            chatbot_ui = gr.Chatbot(label="Conversation", height=400)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about your PDF...",
                    label="",
                    scale=5,
                    lines=1,
                    max_lines=3
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)
    
    gr.Markdown("---")
    gr.Markdown("## Retrieved Sources and Excerpts")
    sources_display = gr.Markdown(value="Sources will appear here after you ask a question.")
    
    # Event handlers
    process_btn.click(
        fn=process_uploaded_pdfs,
        inputs=[file_input],
        outputs=[status_box]
    )
    
    send_btn.click(
        fn=chat_response,
        inputs=[msg_box, chatbot_ui],
        outputs=[chatbot_ui, msg_box, sources_display]
    )
    
    msg_box.submit(
        fn=chat_response,
        inputs=[msg_box, chatbot_ui],
        outputs=[chatbot_ui, msg_box, sources_display]
    )
    
    clear_btn.click(
        fn=clear_chat_history,
        outputs=[chatbot_ui, sources_display]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, debug=False, show_error=True)
