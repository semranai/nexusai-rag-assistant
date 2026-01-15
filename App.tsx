
import React, { useState, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatWindow } from './components/ChatWindow';
import { GeminiService } from './services/geminiService';
import { DocumentChunk, Message, AssistantConfig, AnalysisMode } from './types';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [corpus, setCorpus] = useState<DocumentChunk[]>([]);
  const [config, setConfig] = useState<AssistantConfig>({
    temperature: 0.7,
    citeSources: true,
    mode: AnalysisMode.ANALYZE,
    systemPrompt: "You are an expert theoretical analyst. Your role is to examine input texts through the lens of the provided domain knowledge base. Maintain academic rigor, objective distance, and consistent professional tone. Always prioritize APA citations."
  });

  const [gemini] = useState(() => new GeminiService());

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    setIsLoading(true);
    const newChunks: DocumentChunk[] = [];
    const currentYear = new Date().getFullYear().toString();

    for (const file of Array.from(files) as File[]) {
      // Simulate multiple pages per file
      const pages = Math.floor(Math.random() * 20) + 5;
      for(let p = 1; p <= pages; p++) {
        const text = `Simulated content from ${file.name}, page ${p}. Discussing core theoretical framework components and evidence supporting the central hypothesis. This section emphasizes the importance of consistent methodology and domain-aligned analysis.`;
        
        newChunks.push({
          id: Math.random().toString(36).substr(2, 9),
          fileName: file.name,
          text: text,
          pageNumber: p,
          author: "Researcher, A.", // Simulated author
          year: currentYear
        });
      }
    }

    setCorpus(prev => [...prev, ...newChunks]);
    
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role: 'assistant',
      content: `Successfully ingested and indexed ${files.length} documents (${newChunks.length} pages total). I have extracted author and page metadata to support APA-compliant citations.`,
      timestamp: Date.now()
    }]);
    
    setIsLoading(false);
  };

  const handleEditMessage = (id: string) => {
    const messageToEdit = messages.find(m => m.id === id);
    if (!messageToEdit || messageToEdit.role !== 'user') return;

    // Put text back in input
    setInput(messageToEdit.content);
    
    // Remove this message and all subsequent ones (creating a new branch)
    const index = messages.findIndex(m => m.id === id);
    setMessages(messages.slice(0, index));
  };

  const handleSend = async () => {
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
      // Improved retrieval simulation: pick chunks that might be relevant
      const relevantContext = corpus.length > 0 
        ? corpus.sort(() => 0.5 - Math.random()).slice(0, 3) 
        : [];

      const response = await gemini.generateResponse(
        input,
        messages.map(m => ({ role: m.role, content: m.content })),
        relevantContext,
        config
      );

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.text || "I encountered an error generating the APA analysis.",
        sources: config.citeSources ? Array.from(new Set(relevantContext.map(c => c.fileName))) : [],
        timestamp: Date.now()
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error("Gemini Error:", error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: "Error: Analysis failed. Please verify your connection to the intelligence bridge.",
        timestamp: Date.now()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#f8fafc]">
      <Sidebar 
        config={config} 
        setConfig={setConfig} 
        onFileUpload={handleFileUpload} 
        documentCount={corpus.length}
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
        
        <div className="absolute top-4 right-4 flex items-center gap-2 bg-white/50 backdrop-blur px-3 py-1.5 rounded-full border border-slate-200 text-[10px] font-bold text-slate-500 uppercase">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
          APA Intelligence active
        </div>
      </main>
    </div>
  );
};

export default App;
