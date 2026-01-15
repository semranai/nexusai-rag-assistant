
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { DocumentChunk, AssistantConfig, AnalysisMode } from "../types";

export class GeminiService {
  private ai: GoogleGenAI;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  async generateResponse(
    query: string,
    history: { role: string; content: string }[],
    context: DocumentChunk[],
    config: AssistantConfig
  ): Promise<GenerateContentResponse> {
    const contextText = context.map(c => 
      `[File: ${c.fileName}, Author: ${c.author || 'Unknown'}, Year: ${c.year || 'n.d.'}, Page: ${c.pageNumber}]\n${c.text}`
    ).join('\n\n');
    
    const taskInstruction = config.mode === AnalysisMode.SUMMARIZE 
      ? "Provide a concise summary of the following documents." 
      : "Provide a rigorous academic analysis.";

    const citationInstruction = config.citeSources 
      ? "You MUST use strict APA format for all citations. Every claim derived from the context must include an in-text citation with the author, year, and the specific page number provided in the context (e.g., Smith, 2023, p. 14). If multiple sources apply, cite them all." 
      : "Incorporate the knowledge smoothly.";

    const systemPrompt = `
      ${config.systemPrompt}
      
      CRITICAL CITATION RULE: ${citationInstruction}
      
      CONTEXT FROM KNOWLEDGE BASE:
      ${contextText || "No specific documents found. Please state that you are answering from general knowledge."}
    `;

    return await this.ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: [
        ...history.map(h => ({ role: h.role === 'assistant' ? 'model' : 'user', parts: [{ text: h.content }] })),
        { role: 'user', parts: [{ text: query }] }
      ],
      config: {
        systemInstruction: systemPrompt,
        temperature: config.temperature,
      }
    });
  }
}
