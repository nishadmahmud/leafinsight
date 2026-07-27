# PlantVision AI

Comparative Explainable Deep Learning Platform for Indoor Plant Leaf Disease Detection.

## Architecture
- **Frontend**: Next.js 15, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI, Ultralytics YOLO, PyTorch

## Quick Start (Local Development)

### Backend
```bash
cd server
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd web
npm install
npm run dev
```
