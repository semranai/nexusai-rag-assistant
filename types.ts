
export interface DocumentChunk {
  id: string;
  fileName: string;
  text: string;
  pageNumber: number;
  author?: string;
  year?: string;
  embedding?: number[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
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
