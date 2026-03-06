// components/Sidebar.tsx
import React, { ChangeEvent } from "react";
import { AssistantConfig, AnalysisMode } from "../types";

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

interface SidebarProps {
  config: AssistantConfig;
  setConfig: (config: AssistantConfig) => void;
  onFileUpload: (e: ChangeEvent<HTMLInputElement>) => void;
  documents: Document[];
  processingJobs: ProcessingJob[];
  backendStatus: "online" | "offline" | "checking";
  onDeleteDocument: (docId: string) => void;
  onRefreshDocuments: () => void;
  onClearAll: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  config,
  setConfig,
  onFileUpload,
  documents,
  processingJobs,
  backendStatus,
  onDeleteDocument,
  onRefreshDocuments,
  onClearAll,
}) => {
  const confirmDeleteOne = (doc: Document) => {
    const name = doc.title || doc.filename || doc.id;
    const ok = window.confirm(`Delete this document?\n\n${name}`);
    if (ok) onDeleteDocument(doc.id);
  };

  const confirmClearAll = () => {
    const ok = window.confirm(
      `Clear ALL documents?\n\nThis will delete all documents + PDFs from the system.`
    );
    if (ok) onClearAll();
  };

  return (
    <aside className="w-80 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b">
        <h1 className="text-lg font-bold text-gray-800">NexusAI</h1>
        <p className="text-sm text-gray-500">Multi-Document RAG</p>

        <div className="mt-2 flex items-center gap-2">
          <span
            className={`inline-flex items-center px-2 py-1 rounded text-xs ${
              backendStatus === "online"
                ? "bg-green-100 text-green-700"
                : backendStatus === "offline"
                ? "bg-red-100 text-red-700"
                : "bg-yellow-100 text-yellow-700"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full mr-1 ${
                backendStatus === "online"
                  ? "bg-green-500"
                  : backendStatus === "offline"
                  ? "bg-red-500"
                  : "bg-yellow-500"
              }`}
            ></span>
            {backendStatus === "online"
              ? "Online"
              : backendStatus === "offline"
              ? "Offline"
              : "Checking..."}
          </span>

          <span className="text-xs text-gray-500">
            {documents.length} docs • {processingJobs.length} jobs
          </span>
        </div>
      </div>

      {/* Upload Section */}
      <div className="p-4 border-b">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          📄 Upload PDFs
        </label>

        <div className="relative">
          <input
            type="file"
            accept=".pdf"
            multiple
            onChange={onFileUpload}
            className="hidden"
            id="file-upload"
            disabled={backendStatus !== "online"}
          />
          <label
            htmlFor="file-upload"
            className={`block w-full text-center py-2 px-3 rounded border cursor-pointer text-sm font-medium transition-colors ${
              backendStatus === "online"
                ? "bg-blue-50 hover:bg-blue-100 text-blue-700 border-blue-200"
                : "bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed"
            }`}
          >
            Select PDF Files
          </label>

          <p className="text-xs text-gray-500 mt-2">
            Supports multiple large PDFs simultaneously
          </p>
        </div>
      </div>

      {/* Processing Jobs */}
      {processingJobs.length > 0 && (
        <div className="p-4 border-b">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-medium text-gray-700">
              🔄 Processing Jobs
            </h3>
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
              {processingJobs.length}
            </span>
          </div>

          <div className="space-y-2 max-h-40 overflow-y-auto">
            {processingJobs.slice(0, 5).map((job) => (
              <div
                key={job.id}
                className="bg-gray-50 p-3 rounded border border-gray-200"
              >
                <div className="text-xs font-medium truncate mb-1">
                  {job.filename}
                </div>

                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`text-xs font-medium ${
                      job.status === "processing"
                        ? "text-blue-600"
                        : job.status === "completed"
                        ? "text-green-600"
                        : "text-red-600"
                    }`}
                  >
                    {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                  </span>
                  <span className="text-xs font-bold">{job.progress}%</span>
                </div>

                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${
                      job.status === "processing"
                        ? "bg-blue-500"
                        : job.status === "completed"
                        ? "bg-green-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${job.progress}%` }}
                  ></div>
                </div>

                {job.message && (
                  <div className="text-xs text-gray-500 mt-1 truncate">
                    {job.message}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Documents List */}
      <div className="p-4 border-b flex-1 overflow-y-auto">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-sm font-medium text-gray-700">📚 Documents</h3>

          <div className="flex gap-2 items-center">
            <button
              onClick={onRefreshDocuments}
              className="text-xs text-blue-600 hover:text-blue-800"
              title="Refresh documents"
            >
              ↻
            </button>

            <button
              onClick={confirmClearAll}
              disabled={documents.length === 0 || backendStatus !== "online"}
              className={`text-xs px-2 py-1 rounded border ${
                documents.length === 0 || backendStatus !== "online"
                  ? "bg-gray-100 text-gray-400 border-gray-200"
                  : "bg-red-50 text-red-700 border-red-200 hover:bg-red-100"
              }`}
              title="Delete all documents"
            >
              Clear All
            </button>
          </div>
        </div>

        <div className="space-y-3">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="bg-gray-50 p-3 rounded border border-gray-200 hover:bg-gray-100 transition-colors"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-800 truncate">
                    {doc.title || doc.filename}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {doc.author && <span>{doc.author} • </span>}
                    {doc.chunk_count} chunks
                  </div>
                  {doc.year && (
                    <div className="text-xs text-gray-400 mt-1">{doc.year}</div>
                  )}
                </div>

                <button
                  onClick={() => confirmDeleteOne(doc)}
                  className="text-xs text-red-500 hover:text-red-700 ml-2 p-1"
                  title="Delete document"
                >
                  ×
                </button>
              </div>

              <div className="flex items-center justify-between mt-2">
                <span
                  className={`text-xs px-2 py-0.5 rounded ${
                    doc.status === "loaded"
                      ? "bg-green-100 text-green-700"
                      : doc.status === "processing"
                      ? "bg-blue-100 text-blue-700"
                      : "bg-gray-100 text-gray-700"
                  }`}
                >
                  {doc.status}
                </span>

                <span className="text-xs text-gray-500">
                  {new Date(doc.upload_time).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}

          {documents.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              <div className="text-3xl mb-2">📄</div>
              <p className="text-sm">No documents uploaded yet</p>
              <p className="text-xs mt-1">Upload PDFs to get started</p>
            </div>
          )}
        </div>
      </div>

      {/* Configuration */}
      <div className="p-4 border-t">
        <h3 className="text-sm font-medium text-gray-700 mb-3">
          ⚙️ Configuration
        </h3>

        <div className="space-y-4">
          {/* Citations Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-gray-600">Citations</span>
              <p className="text-xs text-gray-500">Show page references</p>
            </div>
            <button
              onClick={() =>
                setConfig({ ...config, citeSources: !config.citeSources })
              }
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                config.citeSources ? "bg-blue-500" : "bg-gray-300"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  config.citeSources ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          {/* Analysis Mode */}
          <div>
            <label className="block text-sm text-gray-600 mb-2">
              Analysis Mode
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setConfig({ ...config, mode: AnalysisMode.SUMMARIZE })}
                className={`flex-1 text-xs py-2 rounded border ${
                  config.mode === AnalysisMode.SUMMARIZE
                    ? "bg-blue-50 border-blue-300 text-blue-700 font-medium"
                    : "bg-gray-50 border-gray-200 text-gray-600"
                }`}
              >
                Summarize
              </button>
              <button
                onClick={() => setConfig({ ...config, mode: AnalysisMode.ANALYZE })}
                className={`flex-1 text-xs py-2 rounded border ${
                  config.mode === AnalysisMode.ANALYZE
                    ? "bg-blue-50 border-blue-300 text-blue-700 font-medium"
                    : "bg-gray-50 border-gray-200 text-gray-600"
                }`}
              >
                Analyze
              </button>
            </div>
          </div>

          {/* Temperature */}
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm text-gray-600">Temperature</span>
              <span className="text-xs text-gray-500">
                {config.temperature.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.temperature}
              onChange={(e) =>
                setConfig({
                  ...config,
                  temperature: parseFloat(e.target.value),
                })
              }
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Precise</span>
              <span>Creative</span>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="text-xs text-gray-500">
            <div className="font-medium">NexusAI v5.0</div>
            <div>Multi-Document RAG System</div>
            <div className="mt-2">
              Total chunks:{" "}
              {documents.reduce((sum, doc) => sum + doc.chunk_count, 0)}
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};