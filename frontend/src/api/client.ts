export type PredictionItem = {
  model: string;
  class_id: string;
  confidence: string;
  inference_time_ms: string;
};

export type PredictResponse = {
  filename: string;
  predictions: PredictionItem[];
};

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Inference request failed");
  }

  return (await response.json()) as PredictResponse;
}
