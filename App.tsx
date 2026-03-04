// App.tsx - UPDATED: doc selection + summarize_doc + query_doc support
import React, { useState, useEffect, ChangeEvent } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { Message, AssistantConfig, AnalysisMode } from "./types";

// ✅ Use env var if available, fallback to localhost
const BACKEND_URL =
  (import.meta as any).env?.VITE_BACKEND_URL || "http://127.0.0.1:8000";

interface ProcessingJob {
  id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "error";
  progress: number;
  message?: string;
  chunks_processed?: number;
  document_id?: string;
}

interface BackendDocument {
  document_id: string;
  filename: string;
  title: string;
  author: string;
  year: string;
  num_chunks: number;
  pages: number;
}

interface Document {
  id: string;
  filename: string;
  title: string;
  author: string;
  year: string;
  chunk_count: number;
  upload_time: number; // ms timestamp
  status: string;
}

type QueryResponse = {
  question?: string;
  answer?: string;
  summary?: string; // for summarize_doc
  citations?: any[];
  evidence?: any[];
  used_doc_id?: string;
};

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "🚀 **NexusAI Multi-Document RAG Ready**\n\n• Upload multiple PDFs\n• Document-level tracking\n• Cross-document search\n• Smart citations",
      timestamp: Date.now(),
    },
  ]);

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [processingJobs, setProcessingJobs] = useState<ProcessingJob[]>([]);
  const [backendStatus, setBackendStatus] = useState<
    "online" | "offline" | "checking"
  >("checking");

  // ✅ new: selected document id
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  const [config, setConfig] = useState<AssistantConfig>({
    temperature: 0.7,
    citeSources: true,
    mode: AnalysisMode.ANALYZE,
    systemPrompt: "You are an expert analyst examining multiple documents.",
  });

  // -----------------------------
  // Helpers
  // -----------------------------
  const addAssistant = (content: string, sources?: any[]) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "assistant",
        content,
        sources: sources || [],
        timestamp: Date.now(),
      },
    ]);
  };

  const checkBackendStatus = async (): Promise<void> => {
    try {
      const res = await fetch(`${BACKEND_URL}/system`);
      setBackendStatus(res.ok ? "online" : "offline");
    } catch {
      setBackendStatus("offline");
    }
  };

  const fetchDocuments = async (): Promise<void> => {
    try {
      const res = await fetch(`${BACKEND_URL}/documents`);
      if (!res.ok) return;

      const raw: BackendDocument[] = await res.json();

      const mapped: Document[] = (raw || []).map((d) => ({
        id: d.document_id,
        filename: d.filename,
        title: d.title,
        author: d.author,
        year: d.year,
        chunk_count: d.num_chunks,
        upload_time: Date.now(), // ms
        status: "ready",
      }));

      setDocuments(mapped);

      // ✅ if selected doc disappeared, clear selection
      if (selectedDocId && !mapped.some((m) => m.id === selectedDocId)) {
        setSelectedDocId(null);
      }
    } catch (err) {
      console.error("Error fetching documents:", err);
    }
  };

  useEffect(() => {
    checkBackendStatus();
    fetchDocuments();

    const interval = setInterval(() => {
      checkBackendStatus();
      fetchDocuments();
    }, 10000);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -----------------------------
  // Upload
  // -----------------------------
  const handleFileUpload = async (
    e: ChangeEvent<HTMLInputElement>
  ): Promise<void> => {
    const files = e.target.files;
    if (!files) return;

    setIsLoading(true);

    const fileArray: File[] = Array.from(files);

    for (const file of fileArray) {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        addAssistant(`⚠️ Skipping non-PDF: ${file.name}`);
        continue;
      }

      const jobId = `${Date.now()}_${file.name}`;
      setProcessingJobs((prev) => [
        ...prev,
        {
          id: jobId,
          filename: file.name,
          status: "processing",
          progress: 35,
          message: "Uploading…",
        },
      ]);

      const formData = new FormData();
      formData.append("file", file);

      try {
        // ✅ be explicit: replace if same filename
        const res = await fetch(`${BACKEND_URL}/upload?force=true`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);

        const data = await res.json(); // { ok: true, saved_as: "file.pdf" }

        setProcessingJobs((prev) =>
          prev.map((j) =>
            j.id === jobId
              ? {
                  ...j,
                  status: "completed",
                  progress: 100,
                  message: "Uploaded & ingested",
                }
              : j
          )
        );

        addAssistant(`✅ **Uploaded & indexed**: ${data.saved_as || file.name}`);

        await fetchDocuments();
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setProcessingJobs((prev) =>
          prev.map((j) =>
            j.id === jobId
              ? { ...j, status: "error", progress: 0, message: msg }
              : j
          )
        );
        addAssistant(`❌ **Upload failed for ${file.name}**: ${msg}`);
      }
    }

    setIsLoading(false);
  };

  // -----------------------------
  // Summarize selected doc
  // -----------------------------
  const handleSummarizeSelected = async (): Promise<void> => {
    if (!selectedDocId || isLoading) {
      addAssistant("⚠️ Select a document first to summarize.");
      return;
    }

    setIsLoading(true);

    const doc = documents.find((d) => d.id === selectedDocId);
    const docLabel = doc?.title || doc?.filename || selectedDocId;

    addAssistant(`🧾 Summarizing **${docLabel}**…`);

    try {
      const res = await fetch(`${BACKEND_URL}/summarize_doc`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: selectedDocId,
          max_chars: 1200,
        }),
      });

      if (!res.ok) throw new Error(`Backend error: ${res.statusText}`);

      const data: QueryResponse = await res.json();

      const summaryText = data.summary || "No summary returned.";

      let extra = "";
      if (config.citeSources && data.citations?.length) {
        const citeLines = data.citations
          .slice(0, 6)
          .map(
            (c: any) =>
              `• ${c.citation || `${c.author || "Unknown"}, ${c.year || "n.d."}`}`
          )
          .join("\n");
        extra += `\n\n**Citations:**\n${citeLines}`;
      }

      addAssistant(`**Summary — ${docLabel}**\n\n${summaryText}${extra}`, data.citations || []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Connection failed";
      addAssistant(`❌ Summarize error: ${msg}`);
    } finally {
      setIsLoading(false);
    }
  };

  // -----------------------------
  // Ask (Query / Query_doc)
  // -----------------------------
  const handleSend = async (): Promise<void> => {
    if (!input.trim() || isLoading) return;

    const question = input;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: question,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      // ✅ if a doc is selected, query ONLY that doc
      const endpoint = selectedDocId ? "/query_doc" : "/query";

      const body = selectedDocId
        ? { question, document_id: selectedDocId, top_k: 8 }
        : { question, top_k: 8 };

      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(`Backend error: ${res.statusText}`);

      const data: QueryResponse = await res.json();

      let extra = "";
      if (config.citeSources && data.citations?.length) {
        const citeLines = data.citations
          .slice(0, 6)
          .map(
            (c: any) =>
              `• ${c.citation || `${c.author || "Unknown"}, ${c.year || "n.d."}`}`
          )
          .join("\n");
        extra += `\n\n**Citations:**\n${citeLines}`;
      }

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `${data.answer || "No response received"}${extra}`,
        sources: data.citations || [],
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Connection failed";
      addAssistant(`❌ Error: ${msg}`);
    } finally {
      setIsLoading(false);
    }
  };

  // -----------------------------
  // Delete doc (not supported)
  // -----------------------------
  const deleteDocument = async (_docId: string): Promise<void> => {
    addAssistant("⚠️ Delete is not implemented on the backend yet.");
  };

  const handleEditMessage = (id: string): void => {
    const messageToEdit = messages.find((m) => m.id === id);
    if (!messageToEdit || messageToEdit.role !== "user") return;

    setInput(messageToEdit.content);
    const index = messages.findIndex((m) => m.id === id);
    setMessages(messages.slice(0, index));
  };

  return (
    <div className="flex h-screen w-full bg-gray-50">
      <Sidebar
        config={config}
        setConfig={setConfig}
        onFileUpload={handleFileUpload}
        documents={documents}
        processingJobs={processingJobs}
        backendStatus={backendStatus}
        onDeleteDocument={deleteDocument}
        onRefreshDocuments={fetchDocuments}
        // ✅ new props
        selectedDocId={selectedDocId}
        onSelectDocument={setSelectedDocId}
        onSummarizeSelected={handleSummarizeSelected}
      />

      <main className="flex-1 overflow-hidden relative">
        <ChatWindow
          messages={messages}
          input={input}
          setInput={setInput}
          onSend={handleSend}
          onEditMessage={handleEditMessage}
          isLoading={isLoading}
        />

        <div
          className={`absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-bold
          ${
            backendStatus === "online"
              ? "bg-green-100 border-green-300 text-green-700"
              : "bg-red-100 border-red-300 text-red-700"
          }`}
        >
          <div
            className={`w-2 h-2 rounded-full ${
              backendStatus === "online" ? "bg-green-500" : "bg-red-500"
            }`}
          ></div>
          {backendStatus === "online"
            ? `Online (${documents.length} docs)`
            : "Offline"}
        </div>
      </main>
    </div>
  );
};

export default App;