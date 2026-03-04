import { AssistantConfig } from "../types";

export class GeminiService {
  async generateResponse(
    query: string,
    history: { role: string; content: string }[],
    context: any[],
    config: AssistantConfig
  ): Promise<any> {

    const response = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: query,
        top_k: 5
      })
    });

    if (!response.ok) {
      throw new Error("Backend request failed");
    }

    const data = await response.json();

    return {
      text: data.answer,
      citations: data.citations,
      evidence: data.evidence
    };
  }
}