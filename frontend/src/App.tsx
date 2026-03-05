import { useEffect, useMemo, useState } from "react";
import { predictImage, type PredictResponse } from "./api/client";
import ImageUpload, { type CropArea } from "./components/ImageUpload";
import InferenceCharts from "./components/InferenceCharts";
import ResultsTable from "./components/ResultsTable";

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cropArea, setCropArea] = useState<CropArea | null>(null);

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
    setCropArea(null);
    setResult(null);
    setError(null);
  };

  const createCroppedFile = async (sourceFile: File, crop: CropArea): Promise<File> => {
    const imageUrl = URL.createObjectURL(sourceFile);
    try {
      const imageElement = await new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("Failed to load selected image for cropping."));
        image.src = imageUrl;
      });

      const sx = Math.round(crop.x * imageElement.naturalWidth);
      const sy = Math.round(crop.y * imageElement.naturalHeight);
      const sw = Math.max(1, Math.round(crop.width * imageElement.naturalWidth));
      const sh = Math.max(1, Math.round(crop.height * imageElement.naturalHeight));

      const canvas = document.createElement("canvas");
      canvas.width = sw;
      canvas.height = sh;
      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Failed to create canvas context for crop.");
      }

      context.drawImage(imageElement, sx, sy, sw, sh, 0, 0, sw, sh);

      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((resultBlob) => {
          if (!resultBlob) {
            reject(new Error("Failed to export cropped image."));
            return;
          }
          resolve(resultBlob);
        }, sourceFile.type || "image/jpeg");
      });

      return new File([blob], `crop_${sourceFile.name}`, {
        type: blob.type || sourceFile.type,
        lastModified: Date.now()
      });
    } finally {
      URL.revokeObjectURL(imageUrl);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const inferenceFile = cropArea
        ? await createCroppedFile(selectedFile, cropArea)
        : selectedFile;
      const data = await predictImage(inferenceFile);
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
          cropArea={cropArea}
          onFileChange={handleFileChange}
          onCropAreaChange={setCropArea}
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
