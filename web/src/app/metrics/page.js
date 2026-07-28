"use client";

import { useState, useEffect } from "react";
import { X, Download, Maximize, ChevronDown } from "lucide-react";
import {
  BarChart as RechartsBarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ─── Static Data from Paper ───
const PAPER_TABLE_I = [
  { model: "YOLOv8s", mAP50: 97.1, mAP5095: 87.9, precision: 92.2, recall: 96.5 },
  { model: "YOLOv9s", mAP50: 96.6, mAP5095: 87.9, precision: 92.9, recall: 96.2 },
  { model: "YOLOv11s", mAP50: 97.4, mAP5095: 89.3, precision: 93.7, recall: 96.7 },
];

const PAPER_TABLE_II = [
  { cls: "Money Plant Bacterial Wilt", p: 0.938, r: 1.0, mAP50: 0.991, mAP5095: 0.961 },
  { cls: "Money Plant Healthy", p: 0.986, r: 1.0, mAP50: 0.995, mAP5095: 0.96 },
  { cls: "Money Plant Manganese Toxicity", p: 0.986, r: 1.0, mAP50: 0.995, mAP5095: 0.966 },
  { cls: "Snake Plant Anthracnose", p: 0.689, r: 1.0, mAP50: 0.933, mAP5095: 0.881 },
  { cls: "Snake Plant Healthy", p: 0.993, r: 0.944, mAP50: 0.99, mAP5095: 0.952 },
  { cls: "Snake Plant Leaf Withering", p: 0.919, r: 0.857, mAP50: 0.966, mAP5095: 0.799 },
  { cls: "Spider Plant Fungal Leaf Spot", p: 1.0, r: 1.0, mAP50: 0.995, mAP5095: 0.87 },
  { cls: "Spider Plant Healthy", p: 0.926, r: 0.897, mAP50: 0.907, mAP5095: 0.789 },
  { cls: "Spider Plant Leaf Tip Necrosis", p: 0.994, r: 1.0, mAP50: 0.995, mAP5095: 0.856 },
];

// ─── Model Config ───
const MODEL_CONFIG = {
  yolov8s: {
    label: "YOLOv8s",
    color: "#3b82f6",
    csvPath: "/YOLOv8s_100/YOLOv8.csv",
    imgBase: "/YOLOv8s_100/leaf_disease_yolo11",
  },
  yolov9s: {
    label: "YOLOv9s",
    color: "#f59e0b",
    csvPath: "/YOLOv9s_100/YOLOv9.csv",
    imgBase: "/YOLOv9s_100/exp_v9",
  },
  yolov11s: {
    label: "YOLOv11s",
    color: "#10b981",
    csvPath: "/YOLOv11s_100/leaf_disease_yolo112/results.csv",
    imgBase: "/YOLOv11s_100/leaf_disease_yolo112",
  },
};

const MODEL_KEYS = ["yolov8s", "yolov9s", "yolov11s"];

// ─── CSV Parser ───
function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = parseFloat(vals[i]);
    });
    return row;
  });
}

// ─── Metric cell color helper ───
function metricColor(val, thresholds = [0.95, 0.9]) {
  if (val >= thresholds[0]) return "text-emerald-600 font-semibold";
  if (val >= thresholds[1]) return "text-amber-600 font-medium";
  return "text-red-500 font-medium";
}

function bestInCol(rows, key) {
  return Math.max(...rows.map((r) => r[key]));
}

// ─── Reusable chart tooltip ───
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 text-xs">
      <p className="font-semibold text-slate-700 mb-1">Epoch {label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(4)}
        </p>
      ))}
    </div>
  );
};

// ─── Main Page Component ───
export default function MetricsPage() {
  const [csvData, setCsvData] = useState({});
  const [activeTab, setActiveTab] = useState("yolov8s");
  const [expandedImg, setExpandedImg] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch and parse all CSVs on mount
  useEffect(() => {
    const fetchAll = async () => {
      const data = {};
      for (const key of MODEL_KEYS) {
        try {
          const res = await fetch(MODEL_CONFIG[key].csvPath);
          const text = await res.text();
          data[key] = parseCSV(text);
        } catch (e) {
          console.error(`Failed to load CSV for ${key}`, e);
          data[key] = [];
        }
      }
      setCsvData(data);
      setLoading(false);
    };
    fetchAll();
  }, []);

  // ─── Merge CSV data for overlay charts ───
  const mergedEpochs = [];
  if (!loading && csvData.yolov8s) {
    const len = Math.min(
      csvData.yolov8s?.length || 0,
      csvData.yolov9s?.length || 0,
      csvData.yolov11s?.length || 0
    );
    for (let i = 0; i < len; i++) {
      const row = { epoch: i + 1 };
      MODEL_KEYS.forEach((k) => {
        const d = csvData[k]?.[i];
        if (d) {
          row[`${k}_train_box`] = d["train/box_loss"];
          row[`${k}_train_cls`] = d["train/cls_loss"];
          row[`${k}_train_dfl`] = d["train/dfl_loss"];
          row[`${k}_val_box`] = d["val/box_loss"];
          row[`${k}_val_cls`] = d["val/cls_loss"];
          row[`${k}_val_dfl`] = d["val/dfl_loss"];
          row[`${k}_precision`] = d["metrics/precision(B)"];
          row[`${k}_recall`] = d["metrics/recall(B)"];
          row[`${k}_mAP50`] = d["metrics/mAP50(B)"];
          row[`${k}_mAP5095`] = d["metrics/mAP50-95(B)"];
        }
      });
      mergedEpochs.push(row);
    }
  }

  // ─── Helpers ───
  const handleDownload = (src, name) => {
    const a = document.createElement("a");
    a.href = src;
    a.download = name;
    a.click();
  };

  const OverlayLineChart = ({ title, metricSuffix, yLabel, domain }) => (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <h4 className="text-sm font-semibold text-slate-700 mb-3">{title}</h4>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={mergedEpochs}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="epoch" tick={{ fontSize: 10 }} label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11 }} />
          <YAxis tick={{ fontSize: 10 }} domain={domain || ["auto", "auto"]} label={{ value: yLabel || "", angle: -90, position: "insideLeft", fontSize: 11 }} />
          <Tooltip content={<ChartTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {MODEL_KEYS.map((k) => (
            <Line
              key={k}
              type="monotone"
              dataKey={`${k}_${metricSuffix}`}
              name={MODEL_CONFIG[k].label}
              stroke={MODEL_CONFIG[k].color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );

  const ImageCard = ({ src, label }) => (
    <div
      className="relative group cursor-pointer rounded-xl overflow-hidden border border-slate-200 bg-white"
      onClick={() => setExpandedImg({ src, name: label })}
    >
      <img src={src} alt={label} className="w-full h-auto object-contain" />
      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all flex items-center justify-center">
        <Maximize className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
      <p className="text-xs text-center text-slate-500 py-2 border-t border-slate-100">{label}</p>
    </div>
  );

  const cfg = MODEL_CONFIG[activeTab];

  return (
    <div className="container mx-auto px-3 py-4 md:px-4 md:py-8 space-y-12">
      {/* ─── PAGE HEADER ─── */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Training Metrics & Results
        </h1>
        <p className="text-sm text-slate-500 max-w-2xl mx-auto">
          Comprehensive evaluation of YOLOv8s, YOLOv9s, and YOLOv11s on the 9-class indoor ornamental plant leaf disease dataset (3,935 images, 100 epochs).
        </p>
      </div>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* SECTION 1 — MODEL COMPARISON OVERVIEW                      */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <section className="space-y-6">
        <h2 className="text-xl font-bold text-slate-800 border-b border-slate-200 pb-2">
          1. Model Comparison Overview
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Table I */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
              Test Performance Comparison <span className="text-slate-400 font-normal">(Table I from Paper)</span>
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                    <th className="text-center py-2 px-3 text-slate-500 font-medium">mAP@50</th>
                    <th className="text-center py-2 px-3 text-slate-500 font-medium">mAP@50-95</th>
                    <th className="text-center py-2 px-3 text-slate-500 font-medium">Precision</th>
                    <th className="text-center py-2 px-3 text-slate-500 font-medium">Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {PAPER_TABLE_I.map((row) => (
                    <tr key={row.model} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-2.5 px-3 font-semibold text-slate-800">{row.model}</td>
                      <td className={`text-center py-2.5 px-3 ${row.mAP50 === bestInCol(PAPER_TABLE_I, "mAP50") ? "text-emerald-600 font-bold" : "text-slate-600"}`}>
                        {row.mAP50}%
                      </td>
                      <td className={`text-center py-2.5 px-3 ${row.mAP5095 === bestInCol(PAPER_TABLE_I, "mAP5095") ? "text-emerald-600 font-bold" : "text-slate-600"}`}>
                        {row.mAP5095}%
                      </td>
                      <td className={`text-center py-2.5 px-3 ${row.precision === bestInCol(PAPER_TABLE_I, "precision") ? "text-emerald-600 font-bold" : "text-slate-600"}`}>
                        {row.precision}%
                      </td>
                      <td className={`text-center py-2.5 px-3 ${row.recall === bestInCol(PAPER_TABLE_I, "recall") ? "text-emerald-600 font-bold" : "text-slate-600"}`}>
                        {row.recall}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* mAP Bar Chart */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
              mAP Comparison
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <RechartsBarChart data={PAPER_TABLE_I} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                <YAxis domain={[80, 100]} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="mAP50" name="mAP@50 (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="mAP5095" name="mAP@50-95 (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </RechartsBarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* SECTION 2 — TRAINING DYNAMICS (CSV → Recharts)             */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <section className="space-y-6">
        <h2 className="text-xl font-bold text-slate-800 border-b border-slate-200 pb-2">
          2. Training Dynamics <span className="text-sm font-normal text-slate-400">(100 Epochs)</span>
        </h2>

        {loading ? (
          <div className="text-center py-12 text-slate-400">Loading CSV data...</div>
        ) : (
          <>
            {/* Training Losses */}
            <div>
              <h3 className="text-sm font-semibold text-slate-600 mb-3">Training Loss Curves</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <OverlayLineChart title="Box Loss" metricSuffix="train_box" yLabel="Loss" />
                <OverlayLineChart title="Classification Loss" metricSuffix="train_cls" yLabel="Loss" />
                <OverlayLineChart title="DFL Loss" metricSuffix="train_dfl" yLabel="Loss" />
              </div>
            </div>

            {/* Validation Losses */}
            <div>
              <h3 className="text-sm font-semibold text-slate-600 mb-3">Validation Loss Curves</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <OverlayLineChart title="Val Box Loss" metricSuffix="val_box" yLabel="Loss" />
                <OverlayLineChart title="Val Classification Loss" metricSuffix="val_cls" yLabel="Loss" />
                <OverlayLineChart title="Val DFL Loss" metricSuffix="val_dfl" yLabel="Loss" />
              </div>
            </div>

            {/* Validation Metrics */}
            <div>
              <h3 className="text-sm font-semibold text-slate-600 mb-3">Validation Performance Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <OverlayLineChart title="Precision" metricSuffix="precision" yLabel="Score" domain={[0, 1]} />
                <OverlayLineChart title="Recall" metricSuffix="recall" yLabel="Score" domain={[0, 1]} />
                <OverlayLineChart title="mAP@50" metricSuffix="mAP50" yLabel="Score" domain={[0, 1]} />
                <OverlayLineChart title="mAP@50-95" metricSuffix="mAP5095" yLabel="Score" domain={[0, 1]} />
              </div>
            </div>
          </>
        )}
      </section>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* SECTION 3 — PER-MODEL DEEP DIVE                           */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <section className="space-y-6">
        <h2 className="text-xl font-bold text-slate-800 border-b border-slate-200 pb-2">
          3. Per-Model Deep Dive
        </h2>

        {/* Tabs */}
        <div className="flex gap-2">
          {MODEL_KEYS.map((k) => (
            <button
              key={k}
              onClick={() => setActiveTab(k)}
              className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${
                activeTab === k
                  ? "text-white"
                  : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
              }`}
              style={activeTab === k ? { backgroundColor: MODEL_CONFIG[k].color } : {}}
            >
              {MODEL_CONFIG[k].label}
            </button>
          ))}
        </div>

        {/* Confusion Matrices */}
        <div>
          <h3 className="text-sm font-semibold text-slate-600 mb-3">Confusion Matrices</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ImageCard
              src={`${cfg.imgBase}/confusion_matrix.png`}
              label={`${cfg.label} — Confusion Matrix`}
            />
            <ImageCard
              src={`${cfg.imgBase}/confusion_matrix_normalized.png`}
              label={`${cfg.label} — Normalized Confusion Matrix`}
            />
          </div>
        </div>

        {/* Performance Curves */}
        <div>
          <h3 className="text-sm font-semibold text-slate-600 mb-3">Performance Curves</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <ImageCard src={`${cfg.imgBase}/BoxF1_curve.png`} label="F1 Curve" />
            <ImageCard src={`${cfg.imgBase}/BoxPR_curve.png`} label="PR Curve" />
            <ImageCard src={`${cfg.imgBase}/BoxP_curve.png`} label="Precision Curve" />
            <ImageCard src={`${cfg.imgBase}/BoxR_curve.png`} label="Recall Curve" />
          </div>
        </div>

        {/* Validation Predictions */}
        <div>
          <h3 className="text-sm font-semibold text-slate-600 mb-3">Validation Predictions vs Ground Truth</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[0, 1, 2].map((i) => (
              <div key={i} className="grid grid-cols-2 gap-2">
                <ImageCard src={`${cfg.imgBase}/val_batch${i}_labels.jpg`} label={`Batch ${i} — Ground Truth`} />
                <ImageCard src={`${cfg.imgBase}/val_batch${i}_pred.jpg`} label={`Batch ${i} — Predictions`} />
              </div>
            ))}
          </div>
        </div>

        {/* Results Summary */}
        <div>
          <h3 className="text-sm font-semibold text-slate-600 mb-3">Training Results Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ImageCard src={`${cfg.imgBase}/results.png`} label={`${cfg.label} — Results Overview`} />
            <ImageCard src={`${cfg.imgBase}/labels.jpg`} label={`${cfg.label} — Label Distribution`} />
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* SECTION 4 — PER-CLASS PERFORMANCE (YOLOv11s)               */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <section className="space-y-6 pb-12">
        <h2 className="text-xl font-bold text-slate-800 border-b border-slate-200 pb-2">
          4. Per-Class Performance — YOLOv11s <span className="text-sm font-normal text-slate-400">(Table II from Paper)</span>
        </h2>

        <div className="bg-white rounded-xl border border-slate-200 p-5 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Disease Class</th>
                <th className="text-center py-2 px-3 text-slate-500 font-medium">Precision</th>
                <th className="text-center py-2 px-3 text-slate-500 font-medium">Recall</th>
                <th className="text-center py-2 px-3 text-slate-500 font-medium">mAP@50</th>
                <th className="text-center py-2 px-3 text-slate-500 font-medium">mAP@50-95</th>
              </tr>
            </thead>
            <tbody>
              {PAPER_TABLE_II.map((row) => (
                <tr key={row.cls} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="py-2.5 px-3 font-medium text-slate-800 whitespace-nowrap">{row.cls}</td>
                  <td className={`text-center py-2.5 px-3 ${metricColor(row.p)}`}>{(row.p * 100).toFixed(1)}%</td>
                  <td className={`text-center py-2.5 px-3 ${metricColor(row.r)}`}>{(row.r * 100).toFixed(1)}%</td>
                  <td className={`text-center py-2.5 px-3 ${metricColor(row.mAP50)}`}>{(row.mAP50 * 100).toFixed(1)}%</td>
                  <td className={`text-center py-2.5 px-3 ${metricColor(row.mAP5095)}`}>{(row.mAP5095 * 100).toFixed(1)}%</td>
                </tr>
              ))}
              {/* Overall row */}
              <tr className="bg-slate-50 font-bold border-t-2 border-slate-300">
                <td className="py-2.5 px-3 text-slate-900">Overall Average</td>
                <td className="text-center py-2.5 px-3 text-slate-900">93.7%</td>
                <td className="text-center py-2.5 px-3 text-slate-900">96.7%</td>
                <td className="text-center py-2.5 px-3 text-slate-900">97.4%</td>
                <td className="text-center py-2.5 px-3 text-slate-900">89.3%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ─── FULLSCREEN LIGHTBOX MODAL ─── */}
      {expandedImg && (
        <div
          className="fixed inset-0 z-[9999] bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center p-4"
          onClick={() => setExpandedImg(null)}
        >
          <div className="relative max-w-5xl w-full max-h-[85vh] flex flex-col items-center" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setExpandedImg(null)}
              className="absolute -top-3 -right-3 z-10 bg-white rounded-full p-1.5 shadow-lg hover:bg-slate-100"
            >
              <X className="w-5 h-5 text-slate-700" />
            </button>
            <img
              src={expandedImg.src}
              alt={expandedImg.name}
              className="max-w-full max-h-[75vh] object-contain rounded-lg"
            />
            <button
              onClick={() => handleDownload(expandedImg.src, expandedImg.name + ".png")}
              className="mt-4 flex items-center gap-2 px-5 py-2.5 bg-white text-slate-800 rounded-lg font-semibold text-sm hover:bg-slate-100 shadow-lg"
            >
              <Download className="w-4 h-4" /> Download Full Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
