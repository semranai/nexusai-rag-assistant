// App.tsx - COMPLETE FIXED VERSION
import React, { useState, useEffect, ChangeEvent } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatWindow } from './components/ChatWindow';
import { Message, AssistantConfig, AnalysisMode } from './types';

const BACKEND_URL = import.meta.env.VITE_API_BASE_URL ?? "https://nexusai-rag-assistant.onrender.com";

interface ProcessingJob {
  id: string;
  filename: string;
  status: 'queued' | 'processing' | 'completed' | 'error';
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

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: '🚀 **NexusAI Multi-Document RAG Ready**\n\n• Upload multiple PDFs\n• Document-level tracking\n• Cross-document search\n• Smart citations',
      timestamp: Date.now()
    }
  ]);
  
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [processingJobs, setProcessingJobs] = useState<ProcessingJob[]>([]);
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [config, setConfig] = useState<AssistantConfig>({
    temperature: 0.7,
    citeSources: true,
    mode: AnalysisMode.ANALYZE,
    systemPrompt: "You are an expert analyst examining multiple documents."
  });

  // Check system status
  useEffect(() => {
    checkBackendStatus();
    fetchDocuments();
    const interval = setInterval(() => {
      checkBackendStatus();
      fetchDocuments();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async (): Promise<void> => {
    try {
      const response = await fetch(`${BACKEND_URL}/system`);
      setBackendStatus(response.ok ? 'online' : 'offline');
    } catch {
      setBackendStatus('offline');
    }
  };

  const fetchDocuments = async (): Promise<void> => {
    try {
      const response = await fetch(`${BACKEND_URL}/documents`);
      if (!response.ok) {
        throw new Error("Failed to fetch documents");
      }
  
      const data = await response.json();
  
      // Map backend fields to frontend format
      const mapped = data.map((doc: any) => ({
        id: doc.document_id,
        filename: doc.filename,
        title: doc.title,
        author: doc.author,
        year: doc.year,
        chunk_count: doc.num_chunks,
        upload_time: Math.floor(Date.now() / 1000),
        status: "loaded"
      }));
  
      setDocuments(mapped);
    } catch (error) {
      console.error("Error fetching documents:", error);
    }
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const files = e.target.files;
    if (!files) return;

    setIsLoading(true);
    
    // Convert FileList to Array with proper typing
    const fileArray: File[] = Array.from(files);
    
    for (const file of fileArray) {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'assistant',
          content: `⚠️ Skipping non-PDF: ${file.name}`,
          timestamp: Date.now()
        }]);
        continue;
      }

      const formData = new FormData();
      formData.append('file', file); // File is a Blob type

      try {
        const response = await fetch(`${BACKEND_URL}/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.job_id) {
          const newJob: ProcessingJob = {
            id: data.job_id,
            filename: data.filename,
            status: 'queued',
            progress: 0,
            message: data.message
          };
          
          setProcessingJobs(prev => [...prev, newJob]);
          pollJobStatus(data.job_id, data.filename);
          
          setMessages(prev => [...prev, {
            id: Date.now().toString(),
            role: 'assistant',
            content: `📤 **Document queued**: ${data.filename}`,
            timestamp: Date.now()
          }]);
        }
      } catch (error) {
        console.error('Upload error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'assistant',
          content: `❌ **Upload failed for ${file.name}**: ${errorMessage}`,
          timestamp: Date.now()
        }]);
      }
    }
    
    setIsLoading(false);
  };

  const pollJobStatus = async (jobId: string, filename: string): Promise<void> => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/jobs/${jobId}`);
        const data = await response.json();
        
        setProcessingJobs(prev => 
          prev.map(job => 
            job.id === jobId ? { ...job, ...data, filename } : job
          )
        );
        
        if (data.status === 'completed' || data.status === 'error') {
          clearInterval(interval);
          fetchDocuments();
          
          setMessages(prev => [...prev, {
            id: Date.now().toString(),
            role: 'assistant',
            content: data.status === 'completed' 
              ? `✅ **Document loaded**: ${filename} (${data.chunks_processed || 0} chunks)`
              : `❌ **Failed**: ${filename} - ${data.message || 'Unknown error'}`,
            timestamp: Date.now()
          }]);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 2000);
  };

  console.log("Sending question:", input);
  const handleSend = async (): Promise<void> => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          top_k: 5
        })
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.statusText}`);
      }

      const data = await response.json();

const answer = typeof data?.answer === "string"
  ? data.answer
  : JSON.stringify(data?.answer ?? "");

const citations = Array.isArray(data?.citations) ? data.citations : [];

const assistantMsg: Message = {
  id: (Date.now() + 1).toString(),
  role: "assistant",
  content: answer || "No response received",
  citations: citations,
  timestamp: Date.now()
};

      setMessages(prev => [...prev, assistantMsg]);

    } catch (error) {
      console.error("API Error:", error);
      const errorMessage = error instanceof Error ? error.message : 'Connection failed';
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: `❌ Error: ${errorMessage}`,
        timestamp: Date.now()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const deleteDocument = async (docId: string): Promise<void> => {
    if (!window.confirm('Delete this document?')) return;
    
    try {
      const response = await fetch(`${BACKEND_URL}/documents/${docId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        fetchDocuments();
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'assistant',
          content: '🗑️ Document deleted',
          timestamp: Date.now()
        }]);
      }
    } catch (error) {
      console.error('Delete error:', error);
    }
  };

  const handleEditMessage = (id: string): void => {
    const messageToEdit = messages.find(m => m.id === id);
    if (!messageToEdit || messageToEdit.role !== 'user') return;

    setInput(messageToEdit.content);
    const index = messages.findIndex(m => m.id === id);
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
        
        <div className={`absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-bold
          ${backendStatus === 'online' 
            ? 'bg-green-100 border-green-300 text-green-700' 
            : 'bg-red-100 border-red-300 text-red-700'}`}>
          <div className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-500' : 'bg-red-500'}`}></div>
          {backendStatus === 'online' 
            ? `Online (${documents.length} docs)` 
            : 'Offline'}
        </div>
      </main>
    </div>
  );
};

export default App;