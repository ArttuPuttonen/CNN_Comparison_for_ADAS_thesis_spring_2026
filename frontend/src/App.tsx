import { useEffect, useMemo, useState } from "react";
import { predictImage, type PredictResponse } from "./api/client";
import ImageUpload from "./components/ImageUpload";
import InferenceCharts from "./components/InferenceCharts";
import ResultsTable from "./components/ResultsTable";

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const previewUrl = useMemo(() => {
    if (!selectedFile) {
      return null;
    }
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileChange = (file: File | null) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
  };

  const handleRunInference = async () => {
    if (!selectedFile) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await predictImage(selectedFile);
      setResult(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-10">
      <div className="mx-auto w-full max-w-5xl space-y-6">
        <header className="text-center">
          <h1 className="text-2xl font-bold text-slate-900 sm:text-3xl">
            CNN Model Comparison for Traffic Sign Recognition
          </h1>
        </header>

        <ImageUpload
          previewUrl={previewUrl}
          selectedFile={selectedFile}
          loading={loading}
          onFileChange={handleFileChange}
          onRunInference={handleRunInference}
        />

        {loading && (
          <section className="rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm text-blue-900">
            Running inference, please wait...
          </section>
        )}

        {error && (
          <section className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </section>
        )}

        {result && (
          <section className="space-y-5">
            <ResultsTable predictions={result.predictions} />
            <InferenceCharts predictions={result.predictions} />
          </section>
        )}
      </div>
    </main>
  );
}
