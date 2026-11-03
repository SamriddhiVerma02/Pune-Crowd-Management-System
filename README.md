# Pune Crowd Management System

A full-stack crowd detection and tourist capacity monitoring system for Pune landmarks. The project combines a React + Vite frontend with a FastAPI backend that analyzes uploaded videos using YOLOv8 and estimates crowd levels in real time.

## Features

- Tourist place listings for Pune landmarks
- Crowd estimation based on video analysis
- YOLOv8-based person detection and tracking
- Capacity comparison against estimated place capacity
- Supabase-based frontend authentication and database integration
- FastAPI REST API for crowd analysis and place data

## Tech Stack

- Frontend: React, Vite, JavaScript
- Backend: FastAPI, Python
- AI/Computer Vision: OpenCV, Ultralytics YOLOv8, PyTorch
- Database/Auth: Supabase

## Project Structure

```text
pune crowd management project/
├── .gitignore
├── README.md
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── test_people_count.py
│   ├── uploads/
│   └── venv/
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── .gitignore
├── uploads/
├── yolov8n.pt
├── yolov8s.pt
└── .env files (local only, not committed)
```

## Important Note

The YOLO model files are large and should not be committed to GitHub by default. Keep them local, or use Git LFS if you want them in the repository.

## Prerequisites

Before running the project, install:

- Python 3.10+
- Node.js 18+
- npm
- Git

## Backend Setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment:

```bash
cd backend
python -m venv venv
```

On Windows:

```powershell
backend\venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at:

- http://127.0.0.1:8000
- API test route: http://127.0.0.1:8000/api/test

## Frontend Setup

1. Open a new terminal.
2. Go to the frontend folder:

```bash
cd frontend
```

3. Install dependencies:

```bash
npm install
```

4. Create a local environment file named `.env.local` in the `frontend` folder:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your_publishable_key_here
```

5. Start the app:

```bash
npm run dev
```

The frontend typically runs at:

- http://localhost:5173

## How the App Works

- The frontend lets users choose a Pune tourist place.
- Users upload a video file.
- The backend uses YOLOv8 to detect people from frames.
- The system estimates crowd density and compares it to estimated visitor capacity.
- Results are shown in the React interface.

## Environment and Secret Safety

This project should never commit real secret files. Keep these local only:

- `.env`
- `.env.local`
- any production secrets or API keys

The `.gitignore` file is configured to ignore them.

## Deployment Notes

- The frontend is built with Vite.
- The backend runs as a FastAPI service.
- For production deployment, use environment variables and keep sensitive values outside the repository.

## License

This project is intended for academic demonstration and final-year project use.

## Author

Pune Crowd Management System
