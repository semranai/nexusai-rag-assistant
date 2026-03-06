// services/backendService.ts
export type QueryResponse = {
  question: string;
  answer: string;
  citations: any[];
  evidence: any[];
};

export type CompareResponse = QueryResponse & {
  used_doc_ids?: string[];
};

export type DocumentInfo = {
  document_id: string;
  filename: string;
  title: string;
  author: string;
  year: string;
  num_chunks: number;
  pages: number;
};

// IMPORTANT:
// - Local dev: set VITE_BACKEND_URL=http://127.0.0.1:8000 in .env.local
// - Production: set VITE_BACKEND_URL=https://<your-render-backend>.onrender.com
const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const BASE_URL = (import.meta as any).env?.VITE_BACKEND_URL || DEFAULT_BASE_URL;

export class BackendService {
  async listDocuments(): Promise<DocumentInfo[]> {
    const res = await fetch(`${BASE_URL}/documents`);
    if (!res.ok) throw new Error(`Failed /documents: ${res.status}`);
    return await res.json();
  }

  async uploadPdf(file: File): Promise<{ ok: boolean; saved_as: string }> {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${BASE_URL}/upload`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) throw new Error(`Failed /upload: ${res.status}`);
    return await res.json();
  }

  async deleteDocument(documentId: string): Promise<{ ok: boolean }> {
    const res = await fetch(
      `${BASE_URL}/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" }
    );

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(
        `Failed DELETE /documents/${documentId}: ${res.status} ${text}`
      );
    }

    return await res.json().catch(() => ({ ok: true }));
  }

  async clearAll(): Promise<{ ok: boolean }> {
    const res = await fetch(`${BASE_URL}/clear`, { method: "POST" });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Failed POST /clear: ${res.status} ${text}`);
    }

    return await res.json().catch(() => ({ ok: true }));
  }

  async query(question: string, top_k = 5): Promise<QueryResponse> {
    const res = await fetch(`${BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k }),
    });

    if (!res.ok) throw new Error(`Failed /query: ${res.status}`);
    return await res.json();
  }

  async compare(
    question: string,
    doc_ids: string[],
    top_k_per_doc = 3
  ): Promise<CompareResponse> {
    const res = await fetch(`${BASE_URL}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, doc_ids, top_k_per_doc }),
    });

    if (!res.ok) throw new Error(`Failed /compare: ${res.status}`);
    return await res.json();
  }
}

export const backendService = new BackendService();
export { BASE_URL };