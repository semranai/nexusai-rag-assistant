from answer_generator import generate_answer

fake_chunks = [
    {
        "chunk_id": "chunk_1",
        "text": "Steps: Initialization. Then iterate: E-step computes P(Z|X, θ). M-step updates θ to maximize likelihood.",
        "document_author": "Salar N. Azadani",
        "document_year": "2025",
        "document_title": "Chapter 3",
        "pages": [18],
        "relevance_score": 0.9,
        "metadata": {"page": 18}
    }
]

out = generate_answer("What are the steps of EM algorithm?", fake_chunks)
print(out["answer"])
print(out["citations"])
