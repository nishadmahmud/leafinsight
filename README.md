# 🌿 LeafInsight

> **Comparative Explainable Deep Learning Platform for Indoor Plant Leaf Disease Detection.**

LeafInsight is a production-ready, full-stack AI platform designed to detect and explain diseases in indoor ornamental plants. Bridging the gap between academic machine learning research and practical software engineering, LeafInsight deploys a highly accurate **YOLOv11s** model through a modern **FastAPI** backend and an offline-capable **Next.js Progressive Web App (PWA)**.

---

## ✨ Key Features

- **🏆 State-of-the-Art Object Detection:** Powered by a custom-trained **YOLOv11s** model achieving **97.4% mAP@50** on indoor plant diseases.
- **🔍 Explainable AI (XAI):** Built-in **Occlusion Sensitivity** and **RISE** (Randomized Input Sampling for Explanation) heatmaps provide visual proof of the model's decision-making, ensuring transparency and trust.
- **📱 Progressive Web App (PWA):** Fully installable on iOS, Android, and Desktop. Features native camera integration for on-the-go plant diagnostics.
- **⚡ Client-Side Image Compression:** Automatically compresses large, high-resolution smartphone photos (10MB+) directly in the browser to under 1MB before network transmission, guaranteeing fast inference on weak networks.
- **📊 Real-Time Hardware Benchmarking:** The backend actively monitors and returns CPU/GPU latency, preprocessing time, and RAM utilization alongside inference results.

---

## 🔬 Dataset & Models

### The Dataset
A meticulously curated dataset of **787 high-resolution images** collected from various plant nurseries, capturing real-world lighting and background variations. Through rigorous photometric and spatial augmentation (Albumentations), the dataset was expanded to **3,935 images**.

**Supported Plants & Classes:**
1. **Money Plant:** Bacterial Wilt Disease, Manganese Toxicity, Healthy
2. **Snake Plant:** Anthracnose, Leaf Withering, Healthy
3. **Spider Plant:** Fungal Leaf Spot, Leaf Tip Necrosis, Healthy

### The Models
We comparatively evaluated YOLOv8s, YOLOv9s, and YOLOv11s architectures. **YOLOv11s** was selected as the primary inference engine due to its superior performance:
- **mAP@50:** 97.4%
- **Precision:** 93.7%
- **Recall:** 96.7%

---

## 🛠️ Technology Stack

### Frontend (Web Application)
- **Framework:** Next.js 16 (App Router)
- **Styling:** Tailwind CSS & shadcn/ui
- **PWA Integration:** Serwist (Service Worker caching & manifest generation)
- **Utilities:** `browser-image-compression` for rapid client-side preprocessing.

### Backend (Inference Engine)
- **Framework:** FastAPI (Python)
- **Deep Learning:** PyTorch, Ultralytics (YOLO)
- **Image Processing:** OpenCV, PIL
- **Explainability:** Custom Occlusion & RISE algorithms

---

## 🔌 API Routes

The FastAPI backend exposes the following robust RESTful endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | `POST` | Accepts an image upload, runs dynamic letterboxing, and returns YOLOv11 bounding boxes and confidence scores. |
| `/api/benchmark` | `POST` | Identical to `/detect`, but also calculates and returns real-time hardware metrics (RAM usage, inference latency). |
| `/api/explain` | `POST` | Generates interpretability heatmaps (Occlusion/RISE) for the uploaded image, returning them as base64-encoded visual overlays. |
| `/api/health` | `GET` | Simple health-check endpoint to verify backend operational status. |

---

## 🚀 Quick Start (Local Development)

### 1. Backend Setup
The backend requires Python 3.9+ and pre-trained YOLO weights inside `server/app/models/`.

```bash
cd server
python -m venv venv

# Activate Virtual Environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install Dependencies
pip install -r requirements.txt

# Run the FastAPI Server
uvicorn app.main:app --reload
```
*The backend will be available at `http://localhost:8000`.*

### 2. Frontend Setup
```bash
cd web

# Install Dependencies
npm install

# Run the Next.js Development Server
npm run dev
```
*The frontend will be available at `http://localhost:3000`.*

---
*Developed by Nishad Mahmud & Ruma*
