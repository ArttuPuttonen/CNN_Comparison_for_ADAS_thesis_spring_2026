import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type { PredictionItem } from "../api/client";

type InferenceChartsProps = {
  predictions: PredictionItem[];
};

type ChartDataPoint = {
  model: string;
  confidence: number;
  inferenceTimeMs: number;
};

const BAR_COLORS = ["#2563eb", "#16a34a", "#ea580c"];

function toNumber(value: string): number {
  const normalized = value.trim().replace("%", "");
  const parsed = Number.parseFloat(normalized);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatConfidence(value: number): string {
  return `${value.toFixed(2)}%`;
}

export default function InferenceCharts({ predictions }: InferenceChartsProps) {
  const chartData: ChartDataPoint[] = predictions.map((item) => ({
    model: item.model,
    confidence: toNumber(item.confidence),
    inferenceTimeMs: toNumber(item.inference_time_ms)
  }));

  return (
    <section className="grid gap-5 rounded-xl border border-slate-200 bg-white p-5 shadow-sm lg:grid-cols-2">
      <div>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-600">
          Inference Time (ms)
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip formatter={(value) => `${value} ms`} />
              <Bar dataKey="inferenceTimeMs" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`time-bar-${entry.model}`}
                    fill={BAR_COLORS[index % BAR_COLORS.length]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-600">
          Confidence
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip formatter={(value) => formatConfidence(Number(value))} />
              <Bar dataKey="confidence" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`confidence-bar-${entry.model}`}
                    fill={BAR_COLORS[index % BAR_COLORS.length]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
