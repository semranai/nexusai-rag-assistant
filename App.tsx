// App.tsx
import React, { useState, useEffect, ChangeEvent } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { Message, AssistantConfig, AnalysisMode } from "./types";
import { backendService } from "./services/backendService";

interface ProcessingJob {
  id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "error";
  progress: number;
  message?: string;
  chunks_processed?: number;
  document_id?: string;
}

interface Document {
  id: string;
  filename: string;
  title: string;
  author: string;
  year: string;
  chunk_count: number;
  upload_time: number;
  status: string;
}

type QueryResponse = {
  question: string;
  answer: string;
  citations?: any[];
  evidence?: any[];
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

  const [config, setConfig] = useState<AssistantConfig>({
    temperature: 0.7,
    citeSources: true,
    mode: AnalysisMode.ANALYZE,
    systemPrompt: "You are an expert analyst examining multiple documents.",
  });

  const addAssistant = (content: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "assistant",
        content,
        timestamp: Date.now(),
      },
    ]);
  };

  const checkBackendStatus = async (): Promise<void> => {
    try {
      await backendService.listDocuments();
      setBackendStatus("online");
    } catch {
      setBackendStatus("offline");
    }
  };

  const fetchDocuments = async (): Promise<void> => {
    try {
      const raw = await backendService.listDocuments();
      const mapped: Document[] = (raw || []).map((d) => ({
        id: d.document_id,
        filename: d.filename,
        title: d.title,
        author: d.author,
        year: d.year,
        chunk_count: d.num_chunks,
        upload_time: Date.now(),
        status: "ready",
      }));
      setDocuments(mapped);
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

      try {
        const data = await backendService.uploadPdf(file);

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
  // Ask (Query)
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
      const data: QueryResponse = await backendService.query(question, 5);

      let extra = "";
      if (config.citeSources && data.citations?.length) {
        const citeLines = data.citations
          .slice(0, 6)
          .map(
            (c: any) =>
              `• ${
                c.citation || `${c.author || "Unknown"}, ${c.year || "n.d."}`
              }`
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
  // Delete doc (×)
  // -----------------------------
  const deleteDocument = async (docId: string): Promise<void> => {
    if (!docId) return;
    try {
      await backendService.deleteDocument(docId);
      addAssistant(`🗑️ Deleted document: ${docId}`);
      await fetchDocuments();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Delete failed";
      addAssistant(`❌ Delete error: ${msg}`);
    }
  };

  // -----------------------------
  // Clear all
  // -----------------------------
  const clearAllDocuments = async (): Promise<void> => {
    try {
      await backendService.clearAll();
      addAssistant(`🧹 Cleared all documents.`);
      await fetchDocuments();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Clear failed";
      addAssistant(`❌ Clear error: ${msg}`);
    }
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
        onClearAll={clearAllDocuments}
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