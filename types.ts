// types.ts - UPDATED
export interface DocumentChunk {
  id: string;
  fileName: string;
  text: string;
  pageNumber: number;
  author?: string;
  year?: string;
  embeddingDimension?: number;  // ADD THIS LINE
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: any[]
  timestamp: number;
}

export enum AnalysisMode {
  SUMMARIZE = 'summarize',
  ANALYZE = 'analyze'
}

export interface AssistantConfig {
  temperature: number;
  citeSources: boolean;
  mode: AnalysisMode;
  systemPrompt: string;
}