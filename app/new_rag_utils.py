
# new_rag_utils.py
import io
import os
import zipfile
from typing import Any, Dict, List, Tuple

import chromadb

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.readers import SimpleDirectoryReader


# -----------------------------
# Defaults & Environment
# -----------------------------
DEFAULT_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.abspath("./chroma_db"))
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "interagentx_rag")  # <- change if you re-index with OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # or "gpt-4-turbo", adjust to your access
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536-dim


# -----------------------------
# LLM / Embeddings (LlamaIndex)
# -----------------------------
# Global LLM for quick RAG-only answer (sanity checks)
llm = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.1,
)

# Global embed model for LlamaIndex (OpenAI 1536-dim)
embed_model = OpenAIEmbedding(model=OPENAI_EMBED_MODEL)
Settings.embed_model = embed_model


# -----------------------------
# Filesystem helpers
# -----------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _safe_join(base: str, *paths: str) -> str:
    base = os.path.abspath(base)
    target = os.path.abspath(os.path.join(base, *paths))
    if not (target.startswith(base + os.sep) or target == base):
        raise ValueError(f"Unsafe path detected: {target}")
    return target

def save_uploaded_zip_to_dir(zip_bytes: bytes, target_dir: str) -> str:
    ensure_dir(target_dir)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for member in zf.infolist():
            safe_path = _safe_join(target_dir, member.filename)
            if member.is_dir():
                os.makedirs(safe_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(safe_path), exist_ok=True)
                with zf.open(member, "r") as src, open(safe_path, "wb") as dst:
                    dst.write(src.read())
    return target_dir

def save_uploaded_files_to_dir(files: List[Tuple[str, bytes]], target_dir: str) -> str:
    ensure_dir(target_dir)
    for fname, data in files:
        safe_fp = _safe_join(target_dir, fname)
        os.makedirs(os.path.dirname(safe_fp), exist_ok=True)
        with open(safe_fp, "wb") as f:
            f.write(data)
    return target_dir


# -----------------------------
# LlamaIndex settings bootstrap
# -----------------------------
def build_llamaindex_settings(embed_model_name: str = OPENAI_EMBED_MODEL) -> None:
    """Update the global LlamaIndex embed model."""
    Settings.embed_model = OpenAIEmbedding(model=embed_model_name)


# -----------------------------
# Document IO & chunking
# -----------------------------
def load_documents_from_dir(input_dir: str) -> List[Document]:
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        required_exts=[".pdf", ".docx", ".txt"],
        filename_as_id=True,
    )
    return reader.load_data()

def chunk_documents(documents: List[Document], chunk_size: int = 1024, chunk_overlap: int = 200) -> List:
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes


# -----------------------------
# Chroma collection helpers
# -----------------------------
def get_chroma_collection(db_dir: str = DEFAULT_DB_DIR, collection_name: str = DEFAULT_COLLECTION):
    """
    Return a (client, collection). This does NOT attach an embedding_function because
    LlamaIndex computes embeddings itself (OpenAIEmbedding 1536-dim).
    """
    ensure_dir(db_dir)
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "InterAgentX chunked RAG corpus (LlamaIndex + OpenAIEmbeddings)"},
    )
    return client, collection

def reset_chroma_collection(db_dir: str = DEFAULT_DB_DIR, collection_name: str = DEFAULT_COLLECTION) -> None:
    """
    Delete & recreate the collection (use when switching embedding dimensions).
    """
    client = chromadb.PersistentClient(path=db_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.create_collection(name=collection_name, metadata={"description": "InterAgentX chunked RAG corpus (reset)"})


# -----------------------------
# Index creation/loading (LlamaIndex+Chroma)
# -----------------------------
def create_or_update_index(
    nodes: List,
    db_dir: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    reset_on_dim_mismatch: bool = True,
) -> VectorStoreIndex:
    """
    Create (or update) a Chroma-backed VectorStoreIndex using the current OpenAI embedding (1536-dim).
    If the target collection already holds vectors with a different dimension (e.g., legacy 768-dim),
    set reset_on_dim_mismatch=True to wipe & rebuild, or use a new collection name.
    """
    client, chroma_collection = get_chroma_collection(db_dir=db_dir, collection_name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            show_progress=True,
        )
        return index
    except Exception as e:
        msg = str(e)
        if reset_on_dim_mismatch and ("dimension" in msg or "expected" in msg):
            # Dimension mismatch likely (e.g., existing 768-dim vs new 1536-dim)
            reset_chroma_collection(db_dir=db_dir, collection_name=collection_name)
            # Re-create collection + index
            _, chroma_collection = get_chroma_collection(db_dir=db_dir, collection_name=collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=Settings.embed_model,
                show_progress=True,
            )
            return index
        raise

def load_index_from_chroma(
    db_dir: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorStoreIndex:
    """
    Load an existing Chroma-backed VectorStoreIndex. This assumes the stored vectors
    were created with the SAME embedding dimension as Settings.embed_model (OpenAI 1536-dim).
    """
    _, chroma_collection = get_chroma_collection(db_dir=db_dir, collection_name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    return index


# -----------------------------
# Retrieval → (context, nodes, sources) for A2A Coordinator
# -----------------------------
def retrieve_context(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 5,
    use_mmr: bool = False,
    mmr_diversity: float = 0.3,
) -> Tuple[str, List, List[Dict[str, Any]]]:
    """
    Return a single context string (with inline [file:pN] headers), the raw results,
    and a sources list (dicts) → usable by your A2A coordinator which embeds
    [[LOCAL_CONTEXT]]\n{context} into the message parts.
    """
    vs_kwargs = {"mmr_threshold": mmr_diversity} if use_mmr else {}

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        vector_store_query_mode="mmr" if use_mmr else "default",
        vector_store_kwargs=vs_kwargs,
    )
    results = retriever.retrieve(query)

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for r in results:
        node = r.node
        meta = node.metadata or {}
        src = meta.get("file_name") or meta.get("file_path") or meta.get("source") or "unknown"
        page = meta.get("page_label") or meta.get("page")
        snippet = node.get_content()
        header = f"[{src}{f':p{page}' if page else ''}]"
        context_parts.append(f"{header}\n{snippet}")
        sources.append({
            "source": src,
            "page": page,
            "score": getattr(r, "score", None),
            "node_id": node.node_id,
            "ref_doc_id": node.ref_doc_id,
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, results, sources


# -----------------------------
# Optional: simple RAG-only (non-A2A) answer for sanity checks
# -----------------------------
def make_rag_prompt(query: str, context: str, max_context_chars: int = 15000) -> Tuple[str, str]:
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[...trimmed...]"

    system = (
        "You are InterAgentX's RAG assistant. Answer using ONLY the provided context.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Return concise, well-structured responses with inline source hints like [filename] or [filename:pN]."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Instructions:\n"
        "- Use only the relevant parts of the context.\n"
        "- Include brief source hints in brackets.\n"
        "- If unsure, say you don't know."
    )
    return system, user

def answer_with_rag(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 5,
    use_mmr: bool = False,
    mmr_diversity: float = 0.3,
) -> Dict[str, Any]:
    """
    Quick local RAG (non-A2A) that uses ChatOpenAI on the retrieved context.
    Useful for validating your retrieval paths before running the multi-agent stack.
    """
    context, _, sources = retrieve_context(index, query, top_k=top_k, use_mmr=use_mmr, mmr_diversity=mmr_diversity)
    system, user = make_rag_prompt(query=query, context=context)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    answer_text = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer_text, "sources": sources, "context": context}
