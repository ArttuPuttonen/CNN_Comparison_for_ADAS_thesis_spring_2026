import type { PredictionItem } from "../api/client";

type ResultsTableProps = {
  predictions: PredictionItem[];
};

const GTSRB_CLASS_LABELS: Record<number, string> = {
  0: "Speed limit (20km/h)",
  1: "Speed limit (30km/h)",
  2: "Speed limit (50km/h)",
  3: "Speed limit (60km/h)",
  4: "Speed limit (70km/h)",
  5: "Speed limit (80km/h)",
  6: "End of speed limit (80km/h)",
  7: "Speed limit (100km/h)",
  8: "Speed limit (120km/h)",
  9: "No passing",
  10: "No passing for vehicles over 3.5 tons",
  11: "Right-of-way at intersection",
  12: "Priority road",
  13: "Yield",
  14: "Stop",
  15: "No vehicles",
  16: "Vehicles over 3.5 tons prohibited",
  17: "No entry",
  18: "General caution",
  19: "Dangerous curve to the left",
  20: "Dangerous curve to the right",
  21: "Double curve",
  22: "Bumpy road",
  23: "Slippery road",
  24: "Road narrows on the right",
  25: "Road work",
  26: "Traffic signals",
  27: "Pedestrians",
  28: "Children crossing",
  29: "Bicycles crossing",
  30: "Beware of ice/snow",
  31: "Wild animals crossing",
  32: "End of all speed and passing limits",
  33: "Turn right ahead",
  34: "Turn left ahead",
  35: "Ahead only",
  36: "Go straight or right",
  37: "Go straight or left",
  38: "Keep right",
  39: "Keep left",
  40: "Roundabout mandatory",
  41: "End of no passing",
  42: "End of no passing by vehicles over 3.5 tons"
};

export default function ResultsTable({ predictions }: ResultsTableProps) {
  const formatConfidence = (value: string): string => {
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed)) {
      return value;
    }
    const percent = parsed <= 1 ? parsed * 100 : parsed;
    return `${percent.toFixed(2)}%`;
  };

  const resolveClassName = (item: PredictionItem): string => {
    const provided = item.class_name?.trim();
    if (provided) {
      return provided;
    }

    const classId = Number.parseInt(item.class_id, 10);
    if (!Number.isFinite(classId)) {
      return `Class ${item.class_id}`;
    }

    if (classId in GTSRB_CLASS_LABELS) {
      return GTSRB_CLASS_LABELS[classId];
    }

    if (classId >= 1 && classId <= 43 && classId - 1 in GTSRB_CLASS_LABELS) {
      return GTSRB_CLASS_LABELS[classId - 1];
    }

    return `Class ${item.class_id}`;
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="mb-4 text-lg font-semibold text-slate-900">Results</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm text-slate-700">
          <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2">Model</th>
              <th className="px-3 py-2">Class ID</th>
              <th className="px-3 py-2">Predicted Sign</th>
              <th className="px-3 py-2">Confidence</th>
              <th className="px-3 py-2">Inference Time (ms)</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((item) => (
              <tr key={item.model} className="border-b border-slate-100 last:border-b-0">
                <td className="px-3 py-2 font-medium text-slate-900">{item.model}</td>
                <td className="px-3 py-2">{item.class_id}</td>
                <td className="px-3 py-2">{resolveClassName(item)}</td>
                <td className="px-3 py-2">{formatConfidence(item.confidence)}</td>
                <td className="px-3 py-2">{item.inference_time_ms}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
