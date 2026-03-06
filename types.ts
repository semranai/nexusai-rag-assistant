// types.ts - UPDATED (Phase 4: safe citations + evidence support)

export interface DocumentChunk {
  id: string;
  fileName: string;
  text: string;
  pageNumber: number;
  author?: string;
  year?: string;
  embeddingDimension?: number;
}

// A citation/source can be:
// - a plain string, or
// - an object coming from backend citations/evidence
export type SourceLike =
  | string
  | {
      author?: string;
      year?: string | number;
      title?: string;
      filename?: string;
      document_title?: string;
      document_author?: string;
      document_year?: string | number;
      pages?: number[];
      page?: number;
      citation?: string;
      chunk_id?: string;
      text?: string; // sometimes backend might include snippet text
      [key: string]: any;
    };

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;

  // Accept strings or objects (prevents crashes and TS mismatch)
  sources?: SourceLike[];

  timestamp: number;
}

export enum AnalysisMode {
  SUMMARIZE = "summarize",
  ANALYZE = "analyze",
}

export interface AssistantConfig {
  temperature: number;
  citeSources: boolean;
  mode: AnalysisMode;
  systemPrompt: string;
}