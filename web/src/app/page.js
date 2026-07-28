"use client";

import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, Image as ImageIcon, Loader2, Maximize, BarChart3, AlertCircle, X, Download, ChevronDown, ChevronUp, Shuffle, Leaf } from "lucide-react";
import { compareModels, benchmarkModels, explainModel, getModels } from "@/services/api";

const SAMPLE_CATEGORIES = [
  { id: "Money_Plant_Bacterial_wilt_disease", label: "Bacterial Wilt", plant: "Money", color: "bg-emerald-500" },
  { id: "Money_Plant_Healthy", label: "Healthy", plant: "Money", color: "bg-emerald-500" },
  { id: "Money_Plant_Manganese_Toxicity", label: "Mn Toxicity", plant: "Money", color: "bg-emerald-500" },
  { id: "Snake_Plant_Anthracnose", label: "Anthracnose", plant: "Snake", color: "bg-blue-500" },
  { id: "Snake_Plant_Healthy", label: "Healthy", plant: "Snake", color: "bg-blue-500" },
  { id: "Snake_Plant_Leaf_Withering", label: "Withering", plant: "Snake", color: "bg-blue-500" },
  { id: "Spider_Plant_Fungal_leaf_spot", label: "Fungal Spot", plant: "Spider", color: "bg-amber-500" },
  { id: "Spider_Plant_Healthy", label: "Healthy", plant: "Spider", color: "bg-amber-500" },
  { id: "Spider_Plant_Leaf_Tip_Necrosis", label: "Tip Necrosis", plant: "Spider", color: "bg-amber-500" },
];

function pickRandom(arr, n) {
  const shuffled = [...arr].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, n);
}

import { 
  BarChart as RechartsBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from "recharts";

export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  
  // Controls
  const [selectedModels, setSelectedModels] = useState(["yolov8s", "yolov11s"]);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [xaiMethod, setXaiMethod] = useState("occlusion");
  
  // State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExplaining, setIsExplaining] = useState(false);
  const [error, setError] = useState(null);
  
  // Results
  const [detectResults, setDetectResults] = useState(null);
  const [benchmarkResults, setBenchmarkResults] = useState(null);
  const [xaiOcclusion, setXaiOcclusion] = useState(null);
  const [xaiRise, setXaiRise] = useState(null);
  const [expandedImg, setExpandedImg] = useState(null);

  // Sample images
  const [sampleOpen, setSampleOpen] = useState(false);
  const [activeCategory, setActiveCategory] = useState(null);
  const [loadingSample, setLoadingSample] = useState(false);
  const [manifest, setManifest] = useState(null);
  const [sampleThumbs, setSampleThumbs] = useState([]);

  useEffect(() => {
    fetch("/test/manifest.json")
      .then((r) => r.json())
      .then(setManifest)
      .catch(() => {});
  }, []);

  const selectCategory = (catId) => {
    if (activeCategory === catId) {
      setActiveCategory(null);
      setSampleThumbs([]);
      return;
    }
    setActiveCategory(catId);
    if (manifest && manifest[catId]) {
      setSampleThumbs(pickRandom(manifest[catId], 6));
    }
  };

  const shuffleThumbs = () => {
    if (manifest && activeCategory && manifest[activeCategory]) {
      setSampleThumbs(pickRandom(manifest[activeCategory], 6));
    }
  };

  const loadSampleImage = async (filename) => {
    const path = `/test/${activeCategory}/${filename}.png`;
    setLoadingSample(true);
    try {
      const res = await fetch(path);
      const blob = await res.blob();
      const sampleFile = new File([blob], `${filename}.png`, { type: "image/png" });
      setFile(sampleFile);
      setPreview(URL.createObjectURL(sampleFile));
      setDetectResults(null);
      setBenchmarkResults(null);
      setXaiOcclusion(null);
      setXaiRise(null);
      setError(null);
    } catch (e) {
      console.error("Failed to load sample image", e);
    } finally {
      setLoadingSample(false);
    }
  };

  const [availableModels, setAvailableModels] = useState([]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModels();
        const models = data.available_models.map(m => ({
          id: m,
          name: m.toUpperCase()
        }));
        setAvailableModels(models);
        if (models.length > 0) {
          setSelectedModels([models[0].id]);
        }
      } catch (err) {
        console.error("Failed to fetch models", err);
      }
    };
    fetchModels();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setPreview(URL.createObjectURL(acceptedFiles[0]));
      // Reset all previous results
      setDetectResults(null);
      setBenchmarkResults(null);
      setXaiOcclusion(null);
      setXaiRise(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    maxFiles: 1,
  });

  const toggleModel = (modelId) => {
    setSelectedModels((prev) => {
      if (prev.includes(modelId)) {
        if (prev.length === 1) return prev; 
        return prev.filter((m) => m !== modelId);
      }
      return [...prev, modelId];
    });
  };

  const handleRunAnalysis = async () => {
    if (!file || selectedModels.length === 0) return;
    
    setIsAnalyzing(true);
    setIsExplaining(true);
    setError(null);
    setDetectResults(null);
    setBenchmarkResults(null);
    setXaiOcclusion(null);
    setXaiRise(null);

    try {
      // 1. Fetch Detection and Benchmark concurrently
      const [detectRes, benchRes] = await Promise.all([
        compareModels(file, selectedModels, conf, iou),
        benchmarkModels(file, selectedModels)
      ]);

      setDetectResults(detectRes.results);
      setBenchmarkResults(benchRes.results);
      setIsAnalyzing(false); // Unlock the UI for primary results

      // 2. Determine the "Best" Model for XAI
      // We define "best" as the model that found the highest confidence bounding box
      let bestModel = selectedModels[0];
      let maxConf = -1;

      detectRes.results.forEach((res) => {
        if (res.detections && res.detections.length > 0) {
          const highestInModel = Math.max(...res.detections.map(d => d.confidence));
          if (highestInModel > maxConf) {
            maxConf = highestInModel;
            bestModel = res.model_name;
          }
        }
      });

      // 3. Fetch Explainability for the best model
      const [occRes, riseRes] = await Promise.allSettled([
        explainModel(file, bestModel, "occlusion"),
        explainModel(file, bestModel, "rise")
      ]);
      setXaiOcclusion(occRes.status === 'fulfilled' ? occRes.value : null);
      setXaiRise(riseRes.status === 'fulfilled' ? riseRes.value : null);

    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "An error occurred during analysis.");
      setIsAnalyzing(false);
    } finally {
      setIsExplaining(false);
    }
  };

  const downloadImage = (base64Str, filename) => {
    const link = document.createElement("a");
    link.href = base64Str;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setDetectResults(null);
    setBenchmarkResults(null);
    setXaiOcclusion(null);
    setXaiRise(null);
    setError(null);
  };

  return (
    <div className="container mx-auto py-4 px-3 md:py-8 md:px-4 max-w-7xl flex flex-col gap-6">
      {/* Header / Error */}
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">PlantVision Dashboard</h1>
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 text-red-700 rounded-lg text-sm border border-red-100">
            <AlertCircle className="h-4 w-4" />
            {error}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* --- LEFT PANEL: CONTROLS --- */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wider">Input Source</h2>
            <div 
              {...getRootProps()} 
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center h-40 ${
                isDragActive ? "border-slate-800 bg-slate-50" : "border-slate-300 hover:border-slate-400 hover:bg-slate-50"
              }`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <div className="w-full h-full relative group">
                  <img src={preview} alt="Preview" className="w-full h-full object-contain rounded" />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
                    <span className="bg-white text-slate-900 text-xs px-3 py-1.5 rounded-md font-bold shadow-sm">Change Image</span>
                  </div>
                </div>
              ) : (
                <>
                  <UploadCloud className="h-8 w-8 text-slate-400 mb-2" />
                  <span className="text-sm font-medium text-slate-700">Drop leaf image here</span>
                </>
              )}
            </div>

            {/* Sample Images Trigger */}
            <div className="mt-4">
              <button
                onClick={() => setSampleOpen(true)}
                className="w-full flex items-center justify-center gap-2 text-sm font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 transition-colors py-2.5 rounded-lg border border-slate-200"
              >
                <ImageIcon className="w-4 h-4" /> Try a sample image
              </button>
            </div>
          </div>

          <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col gap-5">
            <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">Parameters</h2>
            
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-2">Architectures</label>
              <div className="flex flex-wrap gap-2">
                {availableModels.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => toggleModel(m.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                      selectedModels.includes(m.id) 
                        ? "bg-slate-900 text-white border-slate-900" 
                        : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    {m.name}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-slate-500">Confidence Threshold</label>
                <span className="text-xs text-slate-900 font-medium">{conf}</span>
              </div>
              <input 
                type="range" min="0.01" max="1.0" step="0.01" 
                value={conf} onChange={(e) => setConf(parseFloat(e.target.value))}
                className="w-full accent-slate-900 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div className="hidden">
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-slate-500">IoU Threshold</label>
                <span className="text-xs text-slate-900 font-medium">{iou}</span>
              </div>
              <input 
                type="range" min="0.01" max="1.0" step="0.01" 
                value={iou} onChange={(e) => setIou(parseFloat(e.target.value))}
                className="w-full accent-slate-900 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div className="flex gap-2 mt-2">
              <button
                onClick={handleClear}
                disabled={!file && !detectResults}
                className="px-4 py-3 bg-white text-slate-700 font-bold text-sm rounded-xl border border-slate-200 hover:bg-slate-50 transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Clear
              </button>
              <button
                onClick={handleRunAnalysis}
                disabled={!file || isAnalyzing || selectedModels.length === 0}
                className="flex-1 py-3 bg-slate-900 text-white font-bold text-sm rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800 transition-colors flex justify-center items-center shadow-sm"
              >
                {isAnalyzing ? <Loader2 className="animate-spin h-4 w-4 mr-2" /> : null}
                {isAnalyzing ? "Processing..." : "Run Analysis"}
              </button>
            </div>
          </div>
        </div>

        {/* --- RIGHT PANEL: RESULTS --- */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          
          {/* Default Empty State */}
          {!isAnalyzing && !detectResults && (
            <div className="h-full min-h-[500px] flex flex-col items-center justify-center bg-white rounded-2xl border border-slate-200 shadow-sm p-8">
              <h2 className="text-xl font-bold text-slate-900 mb-8">Supported Research Dataset</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl">
                {/* Money Plant */}
                <div className="bg-slate-50 rounded-xl p-5 border border-slate-100 flex flex-col gap-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Leaf className="w-5 h-5 text-emerald-600" />
                    <h3 className="font-bold text-slate-800 text-sm">Money Plant</h3>
                  </div>
                  <ul className="space-y-2 text-sm text-slate-600">
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>Healthy</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div>Bacterial Wilt</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div>Manganese Toxicity</li>
                  </ul>
                </div>

                {/* Snake Plant */}
                <div className="bg-slate-50 rounded-xl p-5 border border-slate-100 flex flex-col gap-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Leaf className="w-5 h-5 text-emerald-600" />
                    <h3 className="font-bold text-slate-800 text-sm">Snake Plant</h3>
                  </div>
                  <ul className="space-y-2 text-sm text-slate-600">
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>Healthy</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div>Anthracnose</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div>Leaf Withering</li>
                  </ul>
                </div>

                {/* Spider Plant */}
                <div className="bg-slate-50 rounded-xl p-5 border border-slate-100 flex flex-col gap-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Leaf className="w-5 h-5 text-emerald-600" />
                    <h3 className="font-bold text-slate-800 text-sm">Spider Plant</h3>
                  </div>
                  <ul className="space-y-2 text-sm text-slate-600">
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>Healthy</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div>Fungal Leaf Spot</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div>Leaf Tip Necrosis</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Loading State for Primary Results */}
          {isAnalyzing && (
            <div className="h-[400px] flex flex-col items-center justify-center bg-white rounded-2xl border border-slate-200 shadow-sm">
              <Loader2 className="h-8 w-8 text-slate-300 animate-spin mb-4" />
              <p className="text-sm font-medium text-slate-500">Running inference and hardware benchmarks...</p>
            </div>
          )}

          {/* Detections Grid */}
          {detectResults && !isAnalyzing && (
            <div className={`grid grid-cols-1 ${detectResults.length > 1 ? 'md:grid-cols-2' : ''} ${detectResults.length > 2 ? 'xl:grid-cols-3' : ''} gap-4`}>
              {detectResults.map((res, idx) => (
                <div key={idx} className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
                  <div className="px-4 py-3 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                    <h3 className="font-bold text-slate-900 text-sm capitalize">{res.model_name}</h3>
                    <span className="text-xs font-semibold text-slate-500 bg-white px-2 py-1 border border-slate-200 rounded shadow-sm">
                      {res.inference_time_ms.toFixed(1)} ms
                    </span>
                  </div>
                  <div 
                    className="bg-slate-100 p-2 relative h-80 cursor-pointer group"
                    onClick={() => setExpandedImg({ src: `data:image/jpeg;base64,${res.image_base64}`, name: `${res.model_name}_detection.jpg` })}
                  >
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors z-10 flex items-center justify-center rounded">
                      <Maximize className="text-white opacity-0 group-hover:opacity-100 h-8 w-8 drop-shadow-md transition-opacity" />
                    </div>
                    <img 
                      src={`data:image/jpeg;base64,${res.image_base64}`} 
                      alt="Detection" 
                      className="w-full h-full object-contain rounded" 
                    />
                  </div>
                  <div className="p-3 bg-white border-t border-slate-100 min-h-[60px] flex flex-wrap gap-1.5">
                    {res.detections.length === 0 && <span className="text-xs text-slate-400">No objects detected.</span>}
                    {res.detections.map((d, i) => (
                      <span key={i} className="inline-flex items-center rounded bg-slate-100 px-2 py-1 text-[10px] font-bold text-slate-700 uppercase tracking-wider">
                        {d.class_name} <span className="ml-1 text-slate-400">{(d.confidence * 100).toFixed(0)}%</span>
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Metrics & Explainability Row */}
          {detectResults && !isAnalyzing && (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              
              {/* Explainability (XAI) */}
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 flex flex-col">
                <div className="flex items-center gap-2 mb-4 text-slate-900">
                  <Maximize className="h-5 w-5" />
                  <h2 className="font-bold">Explainability Map (XAI)</h2>
                </div>
                
                <div className="flex-1 grid grid-cols-2 gap-4">
                  {/* Occlusion */}
                  <div className="bg-slate-50 rounded-xl border border-slate-100 flex flex-col relative overflow-hidden p-2 min-h-[250px]">
                    {isExplaining ? (
                      <div className="flex-1 flex flex-col items-center justify-center">
                        <Loader2 className="h-6 w-6 text-slate-300 animate-spin mb-3" />
                        <p className="text-[10px] font-medium text-slate-500">Occlusion...</p>
                      </div>
                    ) : xaiOcclusion ? (
                      <div 
                        className="w-full h-full flex flex-col gap-2 cursor-pointer group"
                        onClick={() => setExpandedImg({ src: `data:image/jpeg;base64,${xaiOcclusion.overlay_base64}`, name: `occlusion.jpg` })}
                      >
                        <span className="bg-white/90 backdrop-blur-sm px-2 py-1 rounded shadow-sm text-[10px] font-bold text-slate-900 absolute top-4 left-4 z-20 border border-slate-200">
                          Occlusion
                        </span>
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors z-10 flex items-center justify-center">
                          <Maximize className="text-white opacity-0 group-hover:opacity-100 h-8 w-8 drop-shadow-md transition-opacity" />
                        </div>
                        <img src={`data:image/jpeg;base64,${xaiOcclusion.overlay_base64}`} alt="Occlusion" className="w-full h-full object-contain z-0" />
                      </div>
                    ) : (
                      <p className="text-xs text-slate-400 m-auto">Failed.</p>
                    )}
                  </div>
                  
                  {/* RISE */}
                  <div className="bg-slate-50 rounded-xl border border-slate-100 flex flex-col relative overflow-hidden p-2 min-h-[250px]">
                    {isExplaining ? (
                      <div className="flex-1 flex flex-col items-center justify-center">
                        <Loader2 className="h-6 w-6 text-slate-300 animate-spin mb-3" />
                        <p className="text-[10px] font-medium text-slate-500">RISE...</p>
                      </div>
                    ) : xaiRise ? (
                      <div 
                        className="w-full h-full flex flex-col gap-2 cursor-pointer group"
                        onClick={() => setExpandedImg({ src: `data:image/jpeg;base64,${xaiRise.overlay_base64}`, name: `rise.jpg` })}
                      >
                        <span className="bg-white/90 backdrop-blur-sm px-2 py-1 rounded shadow-sm text-[10px] font-bold text-slate-900 absolute top-4 left-4 z-20 border border-slate-200">
                          RISE
                        </span>
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors z-10 flex items-center justify-center">
                          <Maximize className="text-white opacity-0 group-hover:opacity-100 h-8 w-8 drop-shadow-md transition-opacity" />
                        </div>
                        <img src={`data:image/jpeg;base64,${xaiRise.overlay_base64}`} alt="RISE" className="w-full h-full object-contain z-0" />
                      </div>
                    ) : (
                      <p className="text-xs text-slate-400 m-auto">Failed.</p>
                    )}
                  </div>
                </div>
              </div>

              {/* Benchmarks */}
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 flex flex-col">
                <div className="flex items-center gap-2 mb-4 text-slate-900">
                  <BarChart3 className="h-5 w-5" />
                  <h2 className="font-bold">Hardware Performance</h2>
                </div>
                
                <div className="flex-1 min-h-[250px]">
                  {benchmarkResults && (
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsBarChart data={benchmarkResults} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="model_name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 11}} dy={10} />
                        <YAxis yAxisId="left" orientation="left" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 11}} />
                        <YAxis yAxisId="right" orientation="right" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 11}} />
                        <Tooltip cursor={{fill: '#f8fafc'}} contentStyle={{borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', fontSize: '12px', fontWeight: 500}} />
                        <Legend iconType="circle" wrapperStyle={{fontSize: '12px', paddingTop: '10px'}} />
                        <Bar yAxisId="left" dataKey="inference_time_ms" name="Latency (ms)" fill="#0f172a" radius={[4, 4, 0, 0]} maxBarSize={40} />
                        <Bar yAxisId="right" dataKey="ram_usage_mb" name="RAM (MB)" fill="#94a3b8" radius={[4, 4, 0, 0]} maxBarSize={40} />
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

            </div>
          )}

        </div>
      </div>

      {/* SAMPLE IMAGE MODAL */}
      {sampleOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setSampleOpen(false)}>
          <div className="relative w-full max-w-2xl bg-white rounded-2xl shadow-2xl p-6 flex flex-col max-h-[85vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setSampleOpen(false)} className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 bg-slate-100 hover:bg-slate-200 p-1.5 rounded-full transition-colors">
              <X className="h-5 w-5" />
            </button>
            
            <h2 className="text-xl font-bold text-slate-900 mb-1">Sample Images</h2>
            <p className="text-sm text-slate-500 mb-6">Select a disease category to view test images.</p>

            <div className="overflow-y-auto pr-2 custom-scrollbar flex-1">
              <div className="space-y-6">
                {["Money", "Snake", "Spider"].map((plant) => (
                  <div key={plant}>
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                      {plant} Plant
                      <span className="h-px flex-1 bg-slate-100"></span>
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {SAMPLE_CATEGORIES.filter((c) => c.plant === plant).map((cat) => (
                        <button
                          key={cat.id}
                          onClick={() => selectCategory(cat.id)}
                          className={`px-3 py-2 rounded-lg text-xs font-bold transition-all border ${
                            activeCategory === cat.id
                              ? `${cat.color} text-white border-transparent shadow-md scale-105`
                              : "bg-white text-slate-600 border-slate-200 hover:border-slate-300 hover:bg-slate-50 shadow-sm"
                          }`}
                        >
                          {cat.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {activeCategory && (
                <div className="mt-8 pt-6 border-t border-slate-100 animate-in fade-in slide-in-from-bottom-4 duration-300">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-sm font-semibold text-slate-800">Select an image to load</span>
                    <button 
                      onClick={shuffleThumbs}
                      className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 transition-colors bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded-lg"
                    >
                      <Shuffle className="w-4 h-4" /> Shuffle
                    </button>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                    {sampleThumbs.map((img) => {
                      const src = `/test/${activeCategory}/${img}.png`;
                      return (
                        <button
                          key={img}
                          onClick={() => {
                            loadSampleImage(img);
                            setSampleOpen(false);
                          }}
                          disabled={loadingSample}
                          className="relative rounded-xl overflow-hidden border-2 border-transparent hover:border-slate-800 transition-all aspect-square group bg-slate-100 shadow-sm"
                        >
                          <img src={src} alt={img} className="w-full h-full object-cover" />
                          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-all" />
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* FULLSCREEN IMAGE MODAL */}
      {expandedImg && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-sm p-6" onClick={() => setExpandedImg(null)}>
          <div className="relative w-full max-w-6xl h-[85vh] flex flex-col items-center justify-center" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setExpandedImg(null)} className="absolute -top-4 -right-4 z-50 text-white hover:text-slate-300 bg-white/10 hover:bg-white/20 p-2 rounded-full transition-colors backdrop-blur-md">
              <X className="h-6 w-6" />
            </button>
            <img src={expandedImg.src} className="w-full h-full object-contain drop-shadow-2xl" />
            <button 
              onClick={() => downloadImage(expandedImg.src, expandedImg.name)}
              className="absolute bottom-4 flex items-center gap-2 px-6 py-3 bg-white text-slate-900 rounded-full font-bold hover:bg-slate-200 transition-colors shadow-2xl z-50"
            >
              <Download className="h-5 w-5" /> Download Full Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
