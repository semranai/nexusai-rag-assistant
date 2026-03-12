import { AssistantConfig } from "../types";

export class GeminiService {
  async generateResponse(
    query: string,
    history: { role: string; content: string }[],
    context: any[],
    config: AssistantConfig
  ): Promise<any> {
    // Backend base URL:
    // - Local dev: set in .env (VITE_API_BASE_URL=http://127.0.0.1:8000)
    // - Production: set in your hosting dashboard env vars (VITE_API_BASE_URL=https://...)
    const API_BASE_URL =
      import.meta.env.VITE_API_BASE_URL || "https://nexusai-rag-assistant.onrender.com";

    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: query,
        top_k: 5,
      }),
    });

    if (!response.ok) {
      throw new Error("Backend request failed");
    }

    const data = await response.json();

    return {
      text: data.answer,
      citations: data.citations,
      evidence: data.evidence,
    };
  }
}