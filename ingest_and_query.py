# ingest_and_query.py
import os
from pdf_loader import read_pdf_chunks, extract_metadata
from embeddings import embedder
from vector_store import vector_store

def ingest_pdf(file_path: str, chunk_size: int = 1000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    meta = extract_metadata(file_path)
    filename = os.path.basename(file_path)

    # 1) Create doc_id
    doc_id = vector_store.doc_manager.add_document(filename=filename, metadata=meta)

    # 2) Read chunks
    chunks = read_pdf_chunks(file_path=file_path, chunk_size=chunk_size)

    # 3) Inject doc_id + display fields into chunk metadata
    for ch in chunks:
        ch.setdefault("metadata", {})
        ch["metadata"]["document_id"] = doc_id
        ch["metadata"]["filename"] = filename
        ch["metadata"]["title"] = meta.get("title", filename)
        ch["metadata"]["author"] = meta.get("author", "Unknown Author")
        ch["metadata"]["year"] = meta.get("year", "")

    # 4) Embed (batched)
    embeddings = embedder.batch_embed_with_citations(chunks)

    # 5) Store into FAISS + chunk store
    vector_store.ingest_document(doc_id=doc_id, chunks=chunks, embeddings=embeddings)

    return doc_id

def ask(question: str, top_k: int = 5):
    q_emb = embedder.embed_query(question)
    results = vector_store.search_with_citations(q_emb, k=top_k)

    print("\n" + "="*90)
    print("Q:", question)
    print("="*90)

    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] {r['citation']}")
        print(f"    score={r['relevance_score']:.3f}  distance={r['distance']:.3f}")
        print(f"    APA: {r['full_citation']}")
        snippet = r["text"].strip().replace("\n", " ")
        print(f"    text: {snippet[:350]}{'...' if len(snippet) > 350 else ''}")

    return results

if __name__ == "__main__":
    # --- CHANGE THIS ---
    pdf_path = r"C:\Users\SEM\Downloads\intern2026\some.pdf"
    store_path = "vector_db/store"

    # Ingest
    doc_id = ingest_pdf(pdf_path, chunk_size=1000)
    print("\n✅ Ingested doc:", doc_id)
    print("📊 Stats:", vector_store.get_document_stats())

    # Query
    ask("What is this document about?", top_k=5)

    # Save
    vector_store.save(store_path)

    # Load smoke test (fresh load)
    from vector_store import CitationVectorStore
    fresh = CitationVectorStore(dimension=embedder.dimensions)
    fresh.load(store_path)

    # Query again using fresh store
    q_emb = embedder.embed_query("What is this document about?")
    results = fresh.search_with_citations(q_emb, k=5)

    print("\n" + "="*90)
    print("Smoke test: query after reload")
    print("="*90)
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['citation']} | score={r['relevance_score']:.3f}")
