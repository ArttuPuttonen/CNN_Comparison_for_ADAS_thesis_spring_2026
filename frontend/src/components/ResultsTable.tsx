import type { PredictionItem } from "../api/client";

type ResultsTableProps = {
  predictions: PredictionItem[];
};

export default function ResultsTable({ predictions }: ResultsTableProps) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="mb-4 text-lg font-semibold text-slate-900">Results</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm text-slate-700">
          <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2">Model</th>
              <th className="px-3 py-2">Class ID</th>
              <th className="px-3 py-2">Confidence</th>
              <th className="px-3 py-2">Inference Time (ms)</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((item) => (
              <tr key={item.model} className="border-b border-slate-100 last:border-b-0">
                <td className="px-3 py-2 font-medium text-slate-900">{item.model}</td>
                <td className="px-3 py-2">{item.class_id}</td>
                <td className="px-3 py-2">{item.confidence}</td>
                <td className="px-3 py-2">{item.inference_time_ms}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
