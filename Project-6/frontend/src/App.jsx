import React, { useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Copy,
  Download,
  FileCode,
  FileText,
  Info,
  Languages,
  Loader2,
  RefreshCw,
  Settings2,
  Sparkles,
  Trash2,
  Upload
} from "lucide-react";

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || "http://localhost:5000/api/summarize";
function App() {
  const [inputText, setInputText] = useState("");
  const [file, setFile] = useState(null);

  const [summaryLength, setSummaryLength] = useState("medium");
  const [format, setFormat] = useState("paragraph");
  const [language, setLanguage] = useState("English");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copySuccess, setCopySuccess] = useState(false);

  const fileInputRef = useRef(null);

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    if (selectedFile.type !== "application/pdf") {
      setError("Please select a valid PDF file.");
      setFile(null);
      event.target.value = "";
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("PDF file is too large. Please upload a PDF smaller than 10MB.");
      setFile(null);
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
    setError("");
  }

  async function generateSummary() {
    if (!inputText.trim() && !file) {
      setError("Please paste text or upload a PDF document.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("text", inputText);
    formData.append("summaryLength", summaryLength);
    formData.append("format", format);
    formData.append("language", language);

    if (file) {
      formData.append("pdf", file);
    }

    try {
      const response = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData
      });

      let data = null;

      try {
        data = await response.json();
      } catch {
        throw new Error("Backend returned an invalid response.");
      }

      if (!response.ok) {
        throw new Error(data?.error || "Unable to generate summary.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
        "Failed to connect to backend. Make sure the backend server is running."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleCopy() {
    if (!result?.summary) {
      return;
    }

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(result.summary);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = result.summary;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
      }

      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch {
      setError("Unable to copy summary.");
    }
  }

  function downloadTxt() {
    if (!result?.summary) {
      return;
    }

    const keywords = Array.isArray(result.keywords)
      ? result.keywords.join(", ")
      : "No keywords";

    const content = [
      `Summary`,
      `Language: ${language}`,
      `Length: ${summaryLength}`,
      `Format: ${format}`,
      `Source: ${result.meta?.source || "unknown"}`,
      `Keywords: ${keywords}`,
      "",
      result.summary
    ].join("\n");

    const blob = new Blob([content], {
      type: "text/plain;charset=utf-8"
    });

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `summary-${Date.now()}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  function clearAll() {
    setInputText("");
    setFile(null);
    setResult(null);
    setError("");
    setCopySuccess(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function renderSummary() {
    if (!result?.summary) {
      return null;
    }

    if (format === "bullets") {
      let bulletLines = result.summary
        .split(/\n|(?=\s*[-*•]\s)|(?=\s*\d+\.\s)/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => line.replace(/^[-*•]\s?/, "").replace(/^\d+\.\s?/, ""));

      if (bulletLines.length <= 1) {
        bulletLines = result.summary
          .split(/(?<=[.!?])\s+/)
          .map((line) => line.trim())
          .filter(Boolean);
      }

      return (
        <ul className="list-disc space-y-3 pl-6">
          {bulletLines.map((line, index) => (
            <li key={index} className="leading-relaxed text-slate-700">
              {line}
            </li>
          ))}
        </ul>
      );
    }

    return result.summary
      .split("\n")
      .map((paragraph) => paragraph.trim())
      .filter(Boolean)
      .map((paragraph, index) => (
        <p key={index} className="mb-4 leading-relaxed text-slate-700">
          {paragraph}
        </p>
      ));
  }

  const isGenerateDisabled = loading || (!inputText.trim() && !file);

  return (
    <div className="min-h-screen bg-slate-50 px-4 py-6 text-slate-900 sm:px-6 md:px-8">
      <header className="mx-auto mb-8 max-w-6xl text-center">
        <div className="mb-3 flex items-center justify-center gap-3">
          <div className="rounded-2xl bg-indigo-600 p-3 text-white shadow-lg shadow-indigo-200">
            <Sparkles size={28} />
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-800 sm:text-4xl">
            Summity AI
          </h1>
        </div>
        <p className="mx-auto max-w-2xl text-sm font-medium text-slate-500 sm:text-base">
          Smart AI summarization for academic, business, and professional documents.
        </p>
      </header>

      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-8 lg:grid-cols-2">
        <section className="space-y-6">
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-8">
            <div className="mb-4 flex items-center justify-between gap-4">
              <h2 className="flex items-center gap-2 text-lg font-bold text-slate-700">
                <FileText size={20} className="text-indigo-600" />
                Source Content
              </h2>

              <button
                type="button"
                onClick={clearAll}
                className="rounded-full p-2 text-slate-400 transition hover:bg-red-50 hover:text-red-500"
                aria-label="Clear all"
              >
                <Trash2 size={18} />
              </button>
            </div>

            <textarea
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              placeholder="Paste your article, notes, essay, report, or research content here..."
              className="h-64 w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm leading-relaxed outline-none transition focus:border-transparent focus:ring-2 focus:ring-indigo-500"
            />

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={handleFileChange}
                  className="hidden"
                />

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  className={`flex w-full items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-4 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${file
                    ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                    : "border-slate-200 text-slate-500 hover:border-indigo-300 hover:bg-slate-50"
                    }`}
                >
                  <Upload size={18} />
                  {file ? "Change PDF" : "Upload PDF"}
                </button>
              </div>

              <button
                type="button"
                onClick={generateSummary}
                disabled={isGenerateDisabled}
                className="flex w-full items-center justify-center gap-2 rounded-2xl bg-indigo-600 px-6 py-3 text-sm font-bold text-white shadow-xl shadow-indigo-100 transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
              >
                {loading ? (
                  <Loader2 className="animate-spin" size={20} />
                ) : (
                  <Sparkles size={20} />
                )}
                {loading ? "Processing..." : "Generate Summary"}
              </button>
            </div>

            {file && (
              <div className="mt-4 flex items-center gap-2 rounded-2xl border border-indigo-100 bg-indigo-50 p-3 text-xs font-semibold text-indigo-700">
                <FileCode size={15} />
                <span className="truncate">{file.name}</span>
              </div>
            )}
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-8">
            <h2 className="mb-6 flex items-center gap-2 text-lg font-bold text-slate-700">
              <Settings2 size={20} className="text-indigo-600" />
              Summary Options
            </h2>

            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-xs font-bold uppercase tracking-widest text-slate-400">
                  Length
                </label>
                <select
                  value={summaryLength}
                  onChange={(event) => setSummaryLength(event.target.value)}
                  disabled={loading}
                  className="w-full rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm font-medium outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60"
                >
                  <option value="short">Short</option>
                  <option value="medium">Medium</option>
                  <option value="long">Long</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-bold uppercase tracking-widest text-slate-400">
                  Language
                </label>
                <div className="relative">
                  <Languages
                    size={16}
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
                  />
                  <select
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                    disabled={loading}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 p-3 pl-10 text-sm font-medium outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60"
                  >
                    <option value="English">English</option>
                    <option value="Hindi">Hindi</option>
                    <option value="Spanish">Spanish</option>
                    <option value="French">French</option>
                    <option value="German">German</option>
                    <option value="Chinese">Chinese</option>
                  </select>
                </div>
              </div>

              <div className="space-y-2 sm:col-span-2">
                <label className="text-xs font-bold uppercase tracking-widest text-slate-400">
                  Output Style
                </label>
                <div className="flex rounded-xl bg-slate-100 p-1.5">
                  <button
                    type="button"
                    onClick={() => setFormat("paragraph")}
                    disabled={loading}
                    className={`flex-1 rounded-lg py-2 text-sm font-bold transition disabled:opacity-60 ${format === "paragraph"
                      ? "bg-white text-indigo-600 shadow-sm"
                      : "text-slate-500 hover:text-slate-700"
                      }`}
                  >
                    Paragraphs
                  </button>
                  <button
                    type="button"
                    onClick={() => setFormat("bullets")}
                    disabled={loading}
                    className={`flex-1 rounded-lg py-2 text-sm font-bold transition disabled:opacity-60 ${format === "bullets"
                      ? "bg-white text-indigo-600 shadow-sm"
                      : "text-slate-500 hover:text-slate-700"
                      }`}
                  >
                    Bullet Points
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section>
          <div className="flex min-h-[620px] flex-col rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-8">
            <div className="mb-6 flex items-center justify-between gap-4">
              <h2 className="flex items-center gap-2 text-lg font-bold text-slate-700">
                <CheckCircle2 size={20} className="text-emerald-500" />
                AI Summary
              </h2>

              {result && (
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={handleCopy}
                    className="rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-slate-500 transition hover:text-indigo-600"
                    aria-label="Copy summary"
                  >
                    {copySuccess ? (
                      <CheckCircle2 size={18} className="text-emerald-500" />
                    ) : (
                      <Copy size={18} />
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={downloadTxt}
                    className="rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-slate-500 transition hover:text-indigo-600"
                    aria-label="Download summary"
                  >
                    <Download size={18} />
                  </button>

                  <button
                    type="button"
                    onClick={generateSummary}
                    disabled={loading}
                    className="rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-slate-500 transition hover:text-indigo-600 disabled:cursor-not-allowed disabled:opacity-60"
                    aria-label="Regenerate summary"
                  >
                    <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
                  </button>
                </div>
              )}
            </div>

            {error && (
              <div className="mb-6 flex items-start gap-3 rounded-2xl border border-red-100 bg-red-50 p-4 text-red-700">
                <AlertCircle size={20} className="mt-0.5 flex-shrink-0" />
                <div className="text-sm font-medium">{error}</div>
              </div>
            )}

            {!result && !loading && (
              <div className="flex flex-1 flex-col items-center justify-center rounded-3xl border-2 border-dashed border-slate-100 p-8 text-center">
                <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-slate-50">
                  <FileCode className="text-slate-300" size={32} />
                </div>
                <h3 className="mb-1 font-bold text-slate-800">
                  Waiting for content
                </h3>
                <p className="max-w-xs text-sm text-slate-400">
                  Paste text or upload a PDF, then generate your AI summary.
                </p>
              </div>
            )}

            {loading && (
              <div className="flex flex-1 flex-col items-center justify-center space-y-4 p-8">
                <div className="h-16 w-16 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" />
                <div className="text-center">
                  <p className="animate-pulse font-bold text-slate-800">
                    Analyzing content...
                  </p>
                  <p className="mt-1 text-xs font-semibold uppercase tracking-widest text-slate-400">
                    Generating summary
                  </p>
                </div>
              </div>
            )}

            {result && !loading && (
              <div className="custom-scrollbar flex-1 space-y-6 overflow-y-auto pr-1">
                {result.meta?.chunked && (
                  <div className="flex items-center gap-2 rounded-xl border border-amber-100 bg-amber-50 p-3 text-xs font-semibold text-amber-700">
                    <Info size={14} />
                    <span>
                      Long document detected. Processed in{" "}
                      {result.meta.chunkCount} parts.
                    </span>
                  </div>
                )}

                {Array.isArray(result.keywords) && result.keywords.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {result.keywords.map((keyword, index) => (
                      <span
                        key={`${keyword}-${index}`}
                        className="rounded-lg border border-indigo-100 bg-indigo-50 px-3 py-1.5 text-[10px] font-black uppercase tracking-wider text-indigo-600"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}

                <div className="max-w-none text-sm sm:text-base">
                  {renderSummary()}
                </div>
              </div>
            )}

            <div className="mt-8 border-t border-slate-100 pt-6 text-center text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">
              Powered by Summity AI
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;