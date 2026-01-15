
import React from 'react';
import { AnalysisMode, AssistantConfig } from '../types';

interface SidebarProps {
  config: AssistantConfig;
  setConfig: (config: AssistantConfig) => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  documentCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({ config, setConfig, onFileUpload, documentCount }) => {
  return (
    <div className="w-80 h-full border-r border-slate-200 bg-white p-6 flex flex-col gap-8 overflow-y-auto custom-scrollbar">
      <div>
        <h1 className="text-xl font-bold text-indigo-600 mb-1">NexusAI</h1>
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Custom Knowledge Interface</p>
      </div>

      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
          Knowledge Base
        </h3>
        <label className="block w-full cursor-pointer group">
          <div className="border-2 border-dashed border-slate-200 group-hover:border-indigo-400 rounded-xl p-6 transition-all bg-slate-50 text-center">
            <svg className="w-8 h-8 text-slate-400 group-hover:text-indigo-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
            <span className="text-xs font-medium text-slate-600 block">Upload PDFs/Texts</span>
            <input type="file" className="hidden" multiple onChange={onFileUpload} accept=".pdf,.txt,.docx" />
          </div>
        </label>
        <p className="mt-3 text-xs text-slate-500 italic">
          {documentCount} chunks indexed in vector store.
        </p>
      </section>

      <section className="space-y-6">
        <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
          Model Parameters
        </h3>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-xs font-medium text-slate-600 uppercase">Temperature</label>
            <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">{config.temperature}</span>
          </div>
          <input 
            type="range" 
            min="0" max="1" step="0.1" 
            value={config.temperature} 
            onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
          />
        </div>

        <div className="space-y-3">
          <label className="text-xs font-medium text-slate-600 uppercase">Response Mode</label>
          <div className="grid grid-cols-2 gap-2">
            <button 
              onClick={() => setConfig({ ...config, mode: AnalysisMode.SUMMARIZE })}
              className={`px-3 py-2 text-xs font-semibold rounded-lg border transition-all ${config.mode === AnalysisMode.SUMMARIZE ? 'bg-indigo-600 text-white border-indigo-600 shadow-md' : 'bg-white text-slate-600 border-slate-200 hover:border-indigo-300'}`}
            >
              Summarize
            </button>
            <button 
              onClick={() => setConfig({ ...config, mode: AnalysisMode.ANALYZE })}
              className={`px-3 py-2 text-xs font-semibold rounded-lg border transition-all ${config.mode === AnalysisMode.ANALYZE ? 'bg-indigo-600 text-white border-indigo-600 shadow-md' : 'bg-white text-slate-600 border-slate-200 hover:border-indigo-300'}`}
            >
              Analyze
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between py-2">
          <label className="text-xs font-medium text-slate-600 uppercase">Cite Sources</label>
          <button 
            onClick={() => setConfig({ ...config, citeSources: !config.citeSources })}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none ${config.citeSources ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${config.citeSources ? 'translate-x-5' : 'translate-x-1'}`} />
          </button>
        </div>
      </section>

      <div className="mt-auto">
        <div className="p-4 rounded-xl bg-slate-900 text-white text-[10px] leading-relaxed opacity-80">
          <p className="font-bold mb-1 text-indigo-400">RAG ARCHITECTURE ACTIVE</p>
          Using Gemini-3-Pro for generation and multi-part context window for domain alignment.
        </div>
      </div>
    </div>
  );
};
