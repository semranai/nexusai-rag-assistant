
import React, { useRef, useEffect } from 'react';
import { Message } from '../types';

interface ChatWindowProps {
  messages: Message[];
  input: string;
  setInput: (val: string) => void;
  onSend: () => void;
  onEditMessage: (id: string) => void;
  isLoading: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ messages, input, setInput, onSend, onEditMessage, isLoading }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#f8fafc]">
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-8 custom-scrollbar">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto">
            <div className="w-16 h-16 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center mb-6">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </div>
            <h2 className="text-xl font-bold text-slate-800 mb-2">Ready to analyze</h2>
            <p className="text-slate-500 text-sm">Upload documents to build your APA-compliant knowledge base or ask a question.</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`flex group ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`relative max-w-[85%] rounded-2xl p-5 shadow-sm transition-all ${
              msg.role === 'user' 
                ? 'bg-indigo-600 text-white rounded-tr-none' 
                : 'bg-white border border-slate-200 text-slate-800 rounded-tl-none'
            }`}>
              {msg.role === 'user' && (
                <button 
                  onClick={() => onEditMessage(msg.id)}
                  className="absolute -left-10 top-2 p-2 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-indigo-600"
                  title="Edit and Re-run"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path></svg>
                </button>
              )}
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {msg.content}
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap gap-2">
                  <span className="text-[10px] font-bold text-slate-400 mr-1 self-center uppercase">APA Sources:</span>
                  {msg.sources.map((src, i) => (
                    <span key={i} className="text-[10px] bg-indigo-50 text-indigo-600 font-bold px-2 py-0.5 rounded-full border border-indigo-100 uppercase">
                      {src}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-slate-200 p-5 rounded-2xl rounded-tl-none flex items-center gap-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-.3s]"></div>
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-.5s]"></div>
              </div>
              <span className="text-xs font-medium text-slate-400 uppercase tracking-widest text-[10px]">Consulting Knowledge Base</span>
            </div>
          </div>
        )}
      </div>

      <div className="p-8 pt-0">
        <div className="relative glass-panel rounded-2xl shadow-xl p-2 transition-all focus-within:ring-2 focus-within:ring-indigo-500/20 border-slate-200">
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your corpus a question..."
            className="w-full bg-transparent border-none focus:ring-0 p-4 pr-16 text-sm resize-none custom-scrollbar"
            style={{ maxHeight: '200px' }}
          />
          <button 
            onClick={onSend}
            disabled={!input.trim() || isLoading}
            className={`absolute right-4 bottom-4 w-10 h-10 rounded-xl flex items-center justify-center transition-all ${
              !input.trim() || isLoading ? 'bg-slate-100 text-slate-400' : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg'
            }`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
          </button>
        </div>
      </div>
    </div>
  );
};
